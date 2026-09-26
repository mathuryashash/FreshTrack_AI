import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

import 'pipeline.dart';

/// Logs `FT_TIMING ...` lines to logcat. On in debug builds; for release
/// measurements build with `--dart-define=FT_TIMING=true`.
const bool _logTiming = kDebugMode || bool.fromEnvironment('FT_TIMING');

class ScanTiming {
  final double preprocessMs, inferenceMs, totalMs;
  const ScanTiming(this.preprocessMs, this.inferenceMs, this.totalMs);
}

class Scan {
  final Classification classification;
  final List<double> freshnessLogits, typeLogits;
  final ScanTiming timing;
  const Scan(this.classification, this.freshnessLogits, this.typeLogits, this.timing);
}

/// One thing found in a photo. [box] is null when nothing was detected and the
/// whole photo was classified instead.
class ScanItem {
  final Detection? box;
  final (int, int, int, int)? crop;
  final Classification classification;
  final bool manual;
  const ScanItem(this.box, this.crop, this.classification, {this.manual = false});
}

class Analysis {
  final List<ScanItem> items;
  final int width, height;
  final double decodeMs, detectMs, classifyMs;
  const Analysis(this.items, this.width, this.height, this.decodeMs, this.detectMs, this.classifyMs);

  bool get detected => items.any((i) => i.box != null);
  List<ScanItem> get accepted => [for (final i in items) if (i.classification.isProduce) i];
}

/// One ONNX Runtime session for the app's lifetime. Decoding and resizing run
/// in a background isolate; the plugin runs inference on a native background
/// task queue, so the UI thread never does the heavy work.
class Classifier {
  Classifier._();
  static final instance = Classifier._();

  static const _dir = 'assets/model';
  ModelMeta? _meta;
  DetectorMeta? _detMeta;
  OrtSession? _session, _detSession;
  Future<void>? _loading;
  double? loadMs;
  (String, RgbImage)? _decoded; // last photo, reused when the user draws a box on it

  ModelMeta? get meta => _meta;
  DetectorMeta? get detectorMeta => _detMeta;

  /// Idempotent. A failed load is forgotten so the next call retries.
  Future<void> load() => _loading ??= _load().catchError((Object e) {
        _loading = null;
        throw e;
      });

  Future<void> _load() async {
    final sw = Stopwatch()..start();
    final meta = ModelMeta.fromJson(
        jsonDecode(await rootBundle.loadString('$_dir/model_meta.json')) as Map<String, dynamic>);
    final det = DetectorMeta.fromJson(
        jsonDecode(await rootBundle.loadString('$_dir/detector_meta.json')) as Map<String, dynamic>);
    final ort = OnnxRuntime();
    _session = await ort.createSessionFromAsset('$_dir/${meta.onnxFile}');
    _detSession = await ort.createSessionFromAsset('$_dir/${det.onnxFile}');
    _meta = meta;
    _detMeta = det;
    loadMs = sw.elapsedMicroseconds / 1000;
    if (_logTiming) debugPrint('FT_TIMING load_ms=${loadMs!.toStringAsFixed(1)}');
    _dropStaleModels({meta.onnxFile, det.onnxFile}).ignore();
  }

  /// The ONNX plugin copies model assets into the temp dir and reuses any file
  /// with the same name. Model files are named by content hash (export scripts),
  /// so an update always extracts fresh; this deletes the copies of older models.
  static Future<void> _dropStaleModels(Set<String> keep) async {
    try {
      for (final f in (await getTemporaryDirectory()).listSync().whereType<File>()) {
        final name = f.uri.pathSegments.last;
        if (name.endsWith('.onnx') && !keep.contains(name)) await f.delete();
      }
    } catch (_) {
      // Best effort: a stale copy only wastes storage.
    }
  }

  Future<RgbImage> _decode(String path) async {
    final cached = _decoded;
    if (cached != null && cached.$1 == path) return cached.$2;
    final im = await Isolate.run(() => decodeRgb(File(path).readAsBytesSync()));
    _decoded = (path, im);
    return im;
  }

  Future<List<List<double>>> _run(OrtSession session, String input, Float32List x, List<int> shape,
      List<String> outputs) async {
    final tensor = await OrtValue.fromList(x, shape);
    Map<String, OrtValue> out = const {};
    try {
      out = await session.run({input: tensor});
      return [for (final name in outputs) await _floats(out[name]!)];
    } finally {
      await tensor.dispose();
      for (final v in out.values) {
        await v.dispose();
      }
    }
  }

  Future<Classification> _classifyTensor(Float32List x, {double? gate}) async {
    final meta = _meta!;
    final r = await _run(_session!, meta.inputName, x, [1, 3, meta.imageSize, meta.imageSize], meta.outputNames);
    return postprocess(meta, r[0], r[1], gate: gate);
  }

  /// Detect produce, then classify a square crop around each find. With no
  /// detections, the whole photo is classified (the v2.0 behaviour).
  Future<Analysis> analyse(String imagePath) async {
    await load();
    final meta = _meta!, det = _detMeta!;
    final sw = Stopwatch()..start();
    final im = await _decode(imagePath);
    final ds = det.inputSize, dm = det.mean, dsd = det.std;
    final detInput = await Isolate.run(() => toTensor(im.rgb, im.width, im.height, ds, dm, dsd));
    final t0 = sw.elapsedMicroseconds;
    final r = await _run(_detSession!, det.inputName, detInput, [1, 3, ds, ds], det.outputNames);
    final found = selectDetections(r[0], r[1], det, im.width, im.height);
    final t1 = sw.elapsedMicroseconds;

    final crops = [for (final d in found) cropBounds(d, im.width, im.height, det.cropScale)];
    final size = meta.imageSize, mean = meta.mean, std = meta.std;
    final inputs = await Isolate.run(() => crops.isEmpty
        ? [toTensor(im.rgb, im.width, im.height, size, mean, std)]
        : [for (final (x1, y1, x2, y2) in crops) cropTensor(im, x1, y1, x2, y2, size, mean, std)]);
    final items = <ScanItem>[
      for (var k = 0; k < inputs.length; k++)
        found.isEmpty
            ? ScanItem(null, null, await _classifyTensor(inputs[k]))
            : ScanItem(found[k], crops[k], await _classifyTensor(inputs[k], gate: det.cropOodThreshold)),
    ];
    final t2 = sw.elapsedMicroseconds;
    final a = Analysis(items, im.width, im.height, t0 / 1000, (t1 - t0) / 1000, (t2 - t1) / 1000);
    if (_logTiming) {
      debugPrint('FT_TIMING analyse decode_ms=${a.decodeMs.toStringAsFixed(1)} '
          'detect_ms=${a.detectMs.toStringAsFixed(1)} classify_ms=${a.classifyMs.toStringAsFixed(1)} '
          'boxes=${found.length} items=${items.length}');
    }
    return a;
  }

  /// Classifies a box the user drew (photo pixel coordinates), cropped like a detection.
  Future<ScanItem> classifyRegion(String imagePath, Detection box) async {
    await load();
    final meta = _meta!, det = _detMeta!;
    final im = await _decode(imagePath);
    final crop = cropBounds(box, im.width, im.height, det.cropScale);
    final (x1, y1, x2, y2) = crop;
    final size = meta.imageSize, mean = meta.mean, std = meta.std;
    final x = await Isolate.run(() => cropTensor(im, x1, y1, x2, y2, size, mean, std));
    return ScanItem(box, crop, await _classifyTensor(x, gate: det.cropOodThreshold), manual: true);
  }

  /// JPEGs of the items' crops in [dir] (history thumbnails), all encoded in one
  /// isolate hop; a whole-photo item keeps the photo itself.
  Future<List<String>> saveCrops(String imagePath, List<ScanItem> items, String dir, String prefix) async {
    final paths = [
      for (var k = 0; k < items.length; k++) items[k].crop == null ? imagePath : '$dir/${prefix}_$k.jpg',
    ];
    final targets = [for (var k = 0; k < items.length; k++) if (items[k].crop != null) paths[k]];
    final crops = [for (final i in items) if (i.crop != null) i.crop!];
    if (crops.isEmpty) return paths;
    final im = await _decode(imagePath);
    await Isolate.run(() {
      for (var i = 0; i < crops.length; i++) {
        final (x1, y1, x2, y2) = crops[i];
        File(targets[i]).writeAsBytesSync(encodeCropJpeg(im, x1, y1, x2, y2));
      }
    });
    return paths;
  }

  /// Runs the full pipeline on an image file.
  Future<Scan> classify(String imagePath) async {
    await load();
    final meta = _meta!;
    final sw = Stopwatch()..start();

    final size = meta.imageSize, mean = meta.mean, std = meta.std;
    final input = await Isolate.run(
        () => preprocess(File(imagePath).readAsBytesSync(), size, mean, std));
    final pre = sw.elapsedMicroseconds;

    final tensor = await OrtValue.fromList(input, [1, 3, size, size]);
    Map<String, OrtValue> out = const {};
    final List<double> fresh, type;
    try {
      out = await _session!.run({meta.inputName: tensor});
      fresh = await _floats(out[meta.outputNames[0]]!);
      type = await _floats(out[meta.outputNames[1]]!);
    } finally {
      await tensor.dispose();
      for (final v in out.values) {
        await v.dispose();
      }
    }
    final total = sw.elapsedMicroseconds;

    final timing = ScanTiming(pre / 1000, (total - pre) / 1000, total / 1000);
    if (_logTiming) {
      debugPrint('FT_TIMING pre_ms=${timing.preprocessMs.toStringAsFixed(1)} '
          'inf_ms=${timing.inferenceMs.toStringAsFixed(1)} '
          'total_ms=${timing.totalMs.toStringAsFixed(1)}');
    }
    return Scan(postprocess(meta, fresh, type), fresh, type, timing);
  }

  static Future<List<double>> _floats(OrtValue v) async =>
      [for (final x in await v.asFlattenedList()) (x as num).toDouble()];
}

/// User-facing message for anything thrown while scanning.
String describeScanError(Object e) {
  if (e is FormatException || e is FileSystemException) {
    return 'Could not read this image. Try another photo.';
  }
  return 'Something went wrong: $e';
}
