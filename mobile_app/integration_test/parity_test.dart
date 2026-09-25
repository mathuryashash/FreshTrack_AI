// On-device parity: real Dart preprocessing + ONNX Runtime vs PyTorch logits.
//
// The 4 MB of fixtures are not bundled as app assets (they would ship in the
// release APK). Push them to the device first (world-readable, survives the
// reinstall that `flutter test` does):
//
//   adb push test/fixtures/parity/. /data/local/tmp/parity/
//   flutter test integration_test/parity_test.dart            # debug (JIT)
//   flutter drive --profile --driver=test_driver/integration_test.dart \
//     --target=integration_test/parity_test.dart               # AOT timings
//
// Prints one `PARITY_REPORT {json}` line; `flutter drive` also writes it to
// build/parity_report.json.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:freshtrack_mobile/services/classifier.dart';
import 'package:freshtrack_mobile/services/pipeline.dart';
import 'package:integration_test/integration_test.dart';

double pct(List<double> xs, double p) {
  final s = [...xs]..sort();
  return s[math.min(s.length - 1, (p * (s.length - 1)).round())];
}

Map<String, double> stats(List<double> xs) => {
      'median': pct(xs, 0.5),
      'p95': pct(xs, 0.95),
      'min': xs.reduce(math.min),
      'max': xs.reduce(math.max),
    };

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('on-device pipeline agrees with PyTorch on the 40 parity fixtures', (tester) async {
    final dir = Directory(const String.fromEnvironment('PARITY_DIR', defaultValue: '/data/local/tmp/parity'));
    final expected = (jsonDecode(await File('${dir.path}/expected.json').readAsString()) as List)
        .cast<Map<String, dynamic>>();
    expect(expected, hasLength(40));

    final c = Classifier.instance;
    final sw = Stopwatch()..start();
    await c.load();
    final loadMs = sw.elapsedMicroseconds / 1000;
    final meta = c.meta!;

    // First scan pays one-off costs (isolate code, ORT arena); report it apart.
    final warm = await c.classify('${dir.path}/${expected.first['file']}');

    var freshAgree = 0, typeAgree = 0, freshCorrect = 0, typeCorrect = 0, gateAgree = 0;
    var sumAbs = 0.0, n = 0, maxAbs = 0.0;
    final pre = <double>[], inf = <double>[], total = <double>[];
    final perImage = <Map<String, Object>>[];
    for (final e in expected) {
      final file = e['file'] as String;
      final refF = (e['freshness_logits'] as List).map((v) => (v as num).toDouble()).toList();
      final refT = (e['produce_type_logits'] as List).map((v) => (v as num).toDouble()).toList();
      final s = await c.classify('${dir.path}/$file');

      var imgMax = 0.0;
      for (final (got, ref) in [(s.freshnessLogits, refF), (s.typeLogits, refT)]) {
        for (var i = 0; i < ref.length; i++) {
          final d = (got[i] - ref[i]).abs();
          sumAbs += d;
          n++;
          imgMax = math.max(imgMax, d);
        }
      }
      maxAbs = math.max(maxAbs, imgMax);
      final k = s.classification;
      final ref = postprocess(meta, refF, refT);
      if (k.freshness == ref.freshness) freshAgree++;
      if (k.produceType == ref.produceType) typeAgree++;
      if (k.isProduce == ref.isProduce) gateAgree++;
      if (k.freshness == e['label_freshness']) freshCorrect++;
      if (k.produceType == e['label_produce_type']) typeCorrect++;
      pre.add(s.timing.preprocessMs);
      inf.add(s.timing.inferenceMs);
      total.add(s.timing.totalMs);
      perImage.add({
        'file': file,
        'max_abs_logit_diff': imgMax,
        'p_fresh': k.pFresh,
        'p_fresh_ref': ref.pFresh,
        'freshness': k.freshness,
        'produce_type': k.produceType,
        'energy': k.energy,
        'energy_ref': ref.energy,
      });
    }

    final total40 = expected.length;
    final report = {
      'build_mode': kReleaseMode ? 'release' : (kProfileMode ? 'profile' : 'debug'),
      'n_images': total40,
      'model_load_ms': loadMs,
      'first_scan_ms': {
        'preprocess': warm.timing.preprocessMs,
        'inference': warm.timing.inferenceMs,
        'total': warm.timing.totalMs,
      },
      'top1_agreement': {'freshness': freshAgree / total40, 'produce_type': typeAgree / total40},
      'ood_gate_agreement': gateAgree / total40,
      'logit_abs_diff': {'max': maxAbs, 'mean': sumAbs / n},
      'accuracy_vs_labels': {
        'freshness': freshCorrect / total40,
        'produce_type': typeCorrect / total40,
      },
      'latency_ms': {'preprocess': stats(pre), 'inference': stats(inf), 'total': stats(total)},
      'per_image': perImage,
    };
    binding.reportData = report;
    // ignore: avoid_print
    print('PARITY_REPORT ${jsonEncode(report)}');

    expect(freshAgree / total40, greaterThanOrEqualTo(0.97));
    expect(typeAgree / total40, greaterThanOrEqualTo(0.97));
  });
}
