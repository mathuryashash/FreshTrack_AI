// On-device check of the two-stage scan (detector -> crops -> classifier)
// against the Python reference (src/detection/export.py fixtures).
//
//   adb push test/fixtures/detector/. /data/local/tmp/detector/
//   flutter test integration_test/detector_parity_test.dart
//
// Prints one `DETECTOR_REPORT {json}` line with agreement and timings.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:freshtrack_mobile/services/classifier.dart';
import 'package:freshtrack_mobile/services/pipeline.dart';
import 'package:integration_test/integration_test.dart';

double median(List<double> xs) => ([...xs]..sort())[xs.length ~/ 2];

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('on-device detector + crops agree with Python', (tester) async {
    final dir = Directory(const String.fromEnvironment('DETECTOR_DIR', defaultValue: '/data/local/tmp/detector'));
    final expected = (jsonDecode(await File('${dir.path}/expected.json').readAsString()) as List)
        .cast<Map<String, dynamic>>();
    final c = Classifier.instance;
    await c.load();
    final meta = c.meta!;
    await c.analyse('${dir.path}/${expected.first['file']}'); // warm-up

    var nItems = 0, countAgree = 0, typeAgree = 0, gateAgree = 0;
    var maxBoxDiff = 0.0, maxScoreDiff = 0.0;
    final decode = <double>[], detect = <double>[], classify = <double>[];
    for (final e in expected) {
      final a = await c.analyse('${dir.path}/${e['file']}');
      decode.add(a.decodeMs);
      detect.add(a.detectMs);
      classify.add(a.classifyMs);
      final ref = (e['items'] as List).cast<Map<String, dynamic>>();
      final got = a.items.where((i) => i.box != null).toList();
      if (got.length == ref.length) countAgree++;
      for (var k = 0; k < math.min(got.length, ref.length); k++) {
        nItems++;
        final b = got[k].box!, rb = (ref[k]['box'] as List).map((v) => (v as num).toDouble()).toList();
        for (final (x, r) in [(b.x1, rb[0]), (b.y1, rb[1]), (b.x2, rb[2]), (b.y2, rb[3])]) {
          maxBoxDiff = math.max(maxBoxDiff, (x - r).abs());
        }
        maxScoreDiff = math.max(maxScoreDiff, (b.score - (ref[k]['score'] as num).toDouble()).abs());
        final r = postprocess(
          meta,
          (ref[k]['freshness_logits'] as List).map((v) => (v as num).toDouble()).toList(),
          (ref[k]['produce_type_logits'] as List).map((v) => (v as num).toDouble()).toList(),
          gate: c.detectorMeta!.cropOodThreshold,
        );
        if (got[k].classification.produceType == r.produceType) typeAgree++;
        if (got[k].classification.isProduce == r.isProduce) gateAgree++;
      }
    }
    final report = {
      'build_mode': kReleaseMode ? 'release' : (kProfileMode ? 'profile' : 'debug'),
      'n_photos': expected.length,
      'n_items': nItems,
      'item_count_agreement': countAgree / expected.length,
      'type_agreement': nItems == 0 ? null : typeAgree / nItems,
      'gate_agreement': nItems == 0 ? null : gateAgree / nItems,
      'max_box_diff_px': maxBoxDiff,
      'max_score_diff': maxScoreDiff,
      'median_ms': {'decode': median(decode), 'detect': median(detect), 'classify_all_crops': median(classify)},
    };
    binding.reportData = report;
    // ignore: avoid_print
    print('DETECTOR_REPORT ${jsonEncode(report)}');

    expect(countAgree, expected.length);
    expect(typeAgree, nItems);
    expect(gateAgree, nItems);
    expect(maxBoxDiff, lessThan(3.0)); // JPEG decoders differ slightly between Dart and OpenCV
    expect(maxScoreDiff, lessThan(0.03));
  });
}
