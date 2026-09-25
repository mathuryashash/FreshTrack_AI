import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

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

/// One ONNX Runtime session for the app's lifetime. Decoding and resizing run
/// in a background isolate; the plugin runs inference on a native background
/// task queue, so the UI thread never does the heavy work.
class Classifier {
  Classifier._();
  static final instance = Classifier._();

  static const _dir = 'assets/model';
  ModelMeta? _meta;
  OrtSession? _session;
  Future<void>? _loading;
  double? loadMs;

  ModelMeta? get meta => _meta;

  /// Idempotent. A failed load is forgotten so the next call retries.
  Future<void> load() => _loading ??= _load().catchError((Object e) {
        _loading = null;
        throw e;
      });

  Future<void> _load() async {
    final sw = Stopwatch()..start();
    final meta = ModelMeta.fromJson(
        jsonDecode(await rootBundle.loadString('$_dir/model_meta.json')) as Map<String, dynamic>);
    _session = await OnnxRuntime().createSessionFromAsset('$_dir/${meta.onnxFile}');
    _meta = meta;
    loadMs = sw.elapsedMicroseconds / 1000;
    if (_logTiming) debugPrint('FT_TIMING load_ms=${loadMs!.toStringAsFixed(1)}');
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
