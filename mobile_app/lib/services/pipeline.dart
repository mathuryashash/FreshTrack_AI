// Pure-Dart pre/post-processing for the on-device model. No Flutter or plugin
// imports, so it runs in a background isolate and in plain unit tests.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../models/prediction_result.dart';

/// assets/model/model_meta.json, written by the export script next to the model.
class ModelMeta {
  final String backbone, gitSha, onnxFile, inputName;
  final List<String> outputNames, freshnessLabels, produceTypes;
  final int imageSize;
  final List<double> mean, std;
  final double oodThreshold;
  final List<(double, String)> qualityFromPFresh; // descending min P(fresh)
  final Map<String, double> shelfLifeReferenceDays;

  ModelMeta.fromJson(Map<String, dynamic> j)
      : backbone = j['backbone'] as String,
        gitSha = j['git_sha'] as String? ?? '',
        onnxFile = j['onnx_file'] as String,
        inputName = j['input_name'] as String,
        outputNames = List<String>.from(j['output_names'] as List),
        freshnessLabels = List<String>.from(j['freshness_labels'] as List),
        produceTypes = List<String>.from(j['produce_types'] as List),
        imageSize = j['image_size'] as int,
        mean = _doubles(j['normalize_mean']),
        std = _doubles(j['normalize_std']),
        oodThreshold = (j['ood_threshold'] as num).toDouble(),
        qualityFromPFresh = [
          for (final e in j['quality_from_p_fresh'] as List)
            ((e[0] as num).toDouble(), e[1] as String),
        ],
        shelfLifeReferenceDays = (j['shelf_life_reference_days'] as Map)
            .map((k, v) => MapEntry(k as String, (v as num).toDouble())) {
    // The gate below implements exactly one OOD score; refuse anything else.
    if (j['ood_score'] != 'energy_produce_type') {
      throw FormatException('Unsupported ood_score ${j['ood_score']}');
    }
  }

  static List<double> _doubles(Object? l) => [for (final v in l as List) (v as num).toDouble()];

  String get version => gitSha.length >= 7 ? '$backbone @ ${gitSha.substring(0, 7)}' : backbone;
}

// ── Preprocessing ────────────────────────────────────────────────────────────

/// Encoded JPG/PNG/WebP bytes -> float32 NCHW [1,3,size,size].
/// Matches the Python reference: decode to RGB, apply EXIF orientation, squash
/// (no crop) to size x size with cv2.INTER_LINEAR, /255, (x - mean) / std.
Float32List preprocess(Uint8List encoded, int size, List<double> mean, List<double> std) {
  var im = img.decodeImage(encoded);
  if (im == null) throw const FormatException('Unsupported or corrupt image');
  if (im.exif.imageIfd.hasOrientation && im.exif.imageIfd.orientation != 1) {
    im = img.bakeOrientation(im);
  }
  return toTensor(_rgbBytes(im), im.width, im.height, size, mean, std);
}

/// Packed 8-bit RGB. Alpha is dropped (not composited), like cv2.imread.
Uint8List _rgbBytes(img.Image im) {
  if (im.format == img.Format.uint8 && !im.hasPalette && im.numChannels == 3) {
    return im.toUint8List();
  }
  final out = Uint8List(im.width * im.height * 3);
  final gray = !im.hasPalette && im.numChannels < 3;
  var i = 0;
  for (final p in im) {
    final r = (p.rNormalized * 255).round();
    out[i++] = r;
    out[i++] = gray ? r : (p.gNormalized * 255).round();
    out[i++] = gray ? r : (p.bNormalized * 255).round();
  }
  return out;
}

/// Bilinear resize of packed RGB bytes + normalisation into NCHW floats.
///
/// Replicates OpenCV's 8-bit INTER_LINEAR: half-pixel centres, edge clamping,
/// 11-bit fixed-point weights and the SIMD vertical pass rounding. On the 40
/// parity fixtures this matched cv2.resize on all but 38 of 6.0 M values
/// (each off by 1/255).
Float32List toTensor(Uint8List rgb, int w, int h, int size, List<double> mean, List<double> std) {
  final xs = _Taps(w, size), ys = _Taps(h, size);

  // Horizontal pass on every source row (ints, scaled by 2048).
  final rows = Int32List(h * size * 3);
  for (var y = 0; y < h; y++) {
    final src = y * w * 3, dst = y * size * 3;
    for (var x = 0; x < size; x++) {
      final p0 = src + xs.i0[x] * 3, p1 = src + xs.i1[x] * 3;
      final a0 = xs.a0[x], a1 = xs.a1[x];
      for (var c = 0; c < 3; c++) {
        rows[dst + x * 3 + c] = rgb[p0 + c] * a0 + rgb[p1 + c] * a1;
      }
    }
  }

  // Per-channel lookup: byte -> (v / 255 - mean) / std.
  final lut = Float32List(3 * 256);
  for (var c = 0; c < 3; c++) {
    for (var v = 0; v < 256; v++) {
      lut[c * 256 + v] = (v / 255.0 - mean[c]) / std[c];
    }
  }

  final plane = size * size;
  final out = Float32List(3 * plane);
  for (var y = 0; y < size; y++) {
    final r0 = ys.i0[y] * size * 3, r1 = ys.i1[y] * size * 3;
    final b0 = ys.a0[y], b1 = ys.a1[y];
    for (var x = 0; x < size; x++) {
      for (var c = 0; c < 3; c++) {
        final s0 = rows[r0 + x * 3 + c], s1 = rows[r1 + x * 3 + c];
        var v = (((b0 * (s0 >> 4)) >> 16) + ((b1 * (s1 >> 4)) >> 16) + 2) >> 2;
        if (v > 255) v = 255;
        out[c * plane + y * size + x] = lut[c * 256 + v];
      }
    }
  }
  return out;
}

/// Source indices and 11-bit weights for one axis (cv::resize, INTER_LINEAR).
class _Taps {
  final Int32List i0, i1, a0, a1;
  _Taps(int src, int dst)
      : i0 = Int32List(dst),
        i1 = Int32List(dst),
        a0 = Int32List(dst),
        a1 = Int32List(dst) {
    final f32 = Float32List(1); // OpenCV does this arithmetic in float32
    final scale = src / dst;
    for (var d = 0; d < dst; d++) {
      f32[0] = (d + 0.5) * scale - 0.5;
      var s = f32[0].floor();
      f32[0] = f32[0] - s;
      if (s < 0) {
        s = 0;
        f32[0] = 0;
      }
      if (s >= src - 1) {
        s = src - 1;
        f32[0] = 0;
      }
      final fx = f32[0];
      f32[0] = 1 - fx;
      i0[d] = s;
      i1[d] = math.min(s + 1, src - 1);
      a0[d] = (f32[0] * 2048).round();
      a1[d] = (fx * 2048).round();
    }
  }
}

// ── Post-processing ──────────────────────────────────────────────────────────

List<double> softmax(List<double> z) {
  final m = z.reduce(math.max);
  final e = [for (final v in z) math.exp(v - m)];
  final s = e.reduce((a, b) => a + b);
  return [for (final v in e) v / s];
}

/// logsumexp(logits): the energy OOD score (higher = more in-distribution).
double energy(List<double> z) {
  final m = z.reduce(math.max);
  return m + math.log(z.fold(0.0, (s, v) => s + math.exp(v - m)));
}

int argmax(List<double> z) {
  var best = 0;
  for (var i = 1; i < z.length; i++) {
    if (z[i] > z[best]) best = i;
  }
  return best;
}

/// Heuristic grade from P(fresh): first band whose minimum it reaches.
String qualityFromPFresh(ModelMeta meta, double pFresh) {
  for (final (minP, label) in meta.qualityFromPFresh) {
    if (pFresh >= minP) return label;
  }
  return meta.qualityFromPFresh.last.$2;
}

/// Heuristic days: reference days for the type x P(fresh), 1 decimal.
double shelfLifeDays(ModelMeta meta, String produceType, double pFresh) =>
    ((meta.shelfLifeReferenceDays[produceType] ?? 0) * pFresh * 10).round() / 10;

class Classification {
  final bool isProduce; // false = energy below the OOD threshold
  final double energy;
  final String freshness;
  final double freshnessConfidence, pFresh;
  final String produceType;
  final double produceConfidence;
  final String quality;
  final double shelfLifeDays;

  const Classification({
    required this.isProduce,
    required this.energy,
    required this.freshness,
    required this.freshnessConfidence,
    required this.pFresh,
    required this.produceType,
    required this.produceConfidence,
    required this.quality,
    required this.shelfLifeDays,
  });

  PredictionResult toResult({String? imagePath}) => PredictionResult(
        freshness: freshness,
        freshnessConfidence: freshnessConfidence,
        produceType: produceType,
        quality: quality,
        shelfLifeDays: shelfLifeDays,
        timestamp: DateTime.now(),
        imagePath: imagePath,
      );
}

/// Logits -> labels, same rules as src/api/main.py /predict.
Classification postprocess(ModelMeta meta, List<double> freshnessLogits, List<double> typeLogits) {
  final fp = softmax(freshnessLogits), tp = softmax(typeLogits);
  final fi = argmax(fp), ti = argmax(tp);
  final pFresh = fp[0]; // index 0 == "Fresh"
  final type = meta.produceTypes[ti];
  final e = energy(typeLogits);
  return Classification(
    isProduce: e >= meta.oodThreshold,
    energy: e,
    freshness: meta.freshnessLabels[fi],
    freshnessConfidence: fp[fi],
    pFresh: pFresh,
    produceType: type,
    produceConfidence: tp[ti],
    quality: qualityFromPFresh(meta, pFresh),
    shelfLifeDays: shelfLifeDays(meta, type, pFresh),
  );
}
