import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:freshtrack_mobile/services/pipeline.dart';

DetectorMeta metaWith({double threshold = 0.5, int maxItems = 8, double nms = 0.45}) => DetectorMeta.fromJson({
      'onnx_file': 'detector.onnx',
      'input_name': 'input',
      'output_names': ['boxes', 'scores'],
      'input_size': 320,
      'max_items': maxItems,
      'normalize_mean': [0.5, 0.5, 0.5],
      'normalize_std': [0.5, 0.5, 0.5],
      'score_threshold': threshold,
      'nms_iou': nms,
      'crop_scale': 1.2,
      'crop_ood_threshold': 6.0,
    });

List<double> d(List<num> v) => [for (final x in v) x.toDouble()];

void main() {
  group('selectDetections', () {
    test('drops low scores, suppresses overlaps, keeps score order', () {
      final boxes = d([
        0.1, 0.1, 0.3, 0.3, // 0: 0.9
        0.11, 0.1, 0.31, 0.3, // 1: 0.8, IoU with 0 ~0.9 -> suppressed
        0.6, 0.6, 0.9, 0.9, // 2: 0.7
        0.5, 0.0, 0.6, 0.1, // 3: 0.4, below threshold
      ]);
      final got = selectDetections(boxes, d([0.9, 0.8, 0.7, 0.4]), metaWith(), 1000, 500);
      expect(got.map((e) => e.score), [0.9, 0.7]);
      expect(got.first.x1, closeTo(100, 1e-9));
      expect(got.first.y2, closeTo(150, 1e-9));
    });

    test('caps the number of items', () {
      final boxes = <double>[];
      final scores = <double>[];
      for (var i = 0; i < 12; i++) {
        boxes.addAll([i * 0.08, 0, i * 0.08 + 0.05, 0.05]);
        scores.add(0.9 - i * 0.01);
      }
      expect(selectDetections(boxes, scores, metaWith(maxItems: 8), 100, 100), hasLength(8));
    });

    test('threshold is inclusive', () {
      expect(selectDetections(d([0, 0, 1, 1]), d([0.5]), metaWith(threshold: 0.5), 10, 10), hasLength(1));
    });
  });

  group('crops', () {
    test('square crop, 1.2 x the longer side, clipped to the photo', () {
      expect(cropBounds(const Detection(100, 100, 200, 150, 1), 1000, 1000, 1.2), (90, 65, 210, 185));
      expect(cropBounds(const Detection(0, 0, 50, 100, 1), 300, 300, 1.2), (0, 0, 85, 110));
    });

    test('a degenerate box on the edge still gets a one-pixel crop', () {
      expect(cropBounds(const Detection(300, 200, 300, 200, 1), 300, 200, 1.2), (299, 199, 300, 200));
      expect(cropBounds(const Detection(0, 0, 0, 0, 1), 300, 200, 1.2), (0, 0, 1, 1));
      // A 1 x 1 crop still makes a classifier input.
      final t = cropTensor(RgbImage(Uint8List(300 * 200 * 3), 300, 200), 299, 199, 300, 200, 224,
          const [0.5, 0.5, 0.5], const [0.5, 0.5, 0.5]);
      expect(t, hasLength(3 * 224 * 224));
    });

    test('cropRgb copies the right rows and columns', () {
      // 4 x 3 image whose red channel is x + 10 * y.
      final rgb = Uint8List(4 * 3 * 3);
      for (var y = 0; y < 3; y++) {
        for (var x = 0; x < 4; x++) {
          rgb[(y * 4 + x) * 3] = x + 10 * y;
        }
      }
      final c = cropRgb(RgbImage(rgb, 4, 3), 1, 1, 3, 3);
      expect((c.width, c.height), (2, 2));
      expect([c.rgb[0], c.rgb[3], c.rgb[6], c.rgb[9]], [11, 12, 21, 22]);
    });
  });

  // Fixtures from `python -m src.detection.export` (the Python reference pipeline).
  final dir = Directory('test/fixtures/detector');
  final bundled = File('assets/model/detector_meta.json');
  if (!dir.existsSync() || !bundled.existsSync()) return;
  final meta = DetectorMeta.fromJson(jsonDecode(bundled.readAsStringSync()) as Map<String, dynamic>);
  final clf = ModelMeta.fromJson(
      jsonDecode(File('assets/model/model_meta.json').readAsStringSync()) as Map<String, dynamic>);
  final expected =
      (jsonDecode(File('${dir.path}/expected.json').readAsStringSync()) as List).cast<Map<String, dynamic>>();

  test('bundled detector meta is what the pipeline expects', () {
    expect(meta.inputSize, 320);
    expect(meta.inputName, 'input');
    expect(meta.outputNames, ['boxes', 'scores']);
    expect(meta.mean, [0.5, 0.5, 0.5]);
    expect(meta.std, [0.5, 0.5, 0.5]);
    expect(meta.cropScale, 1.2);
    expect(meta.scoreThreshold, inInclusiveRange(0.1, 0.9));
  });

  for (final e in expected) {
    test('same boxes and crops as Python: ${e['file']}', () {
      final w = e['width'] as int, h = e['height'] as int;
      final got = selectDetections(d((e['candidate_boxes'] as List).cast<num>()),
          d((e['candidate_scores'] as List).cast<num>()), meta, w, h);
      final items = (e['items'] as List).cast<Map<String, dynamic>>();
      expect(got, hasLength(items.length));
      final im = decodeRgb(File('${dir.path}/${e['file']}').readAsBytesSync());
      expect((im.width, im.height), (w, h));
      for (var k = 0; k < items.length; k++) {
        final ref = d((items[k]['box'] as List).cast<num>());
        expect([got[k].x1, got[k].y1, got[k].x2, got[k].y2], [for (final v in ref) closeTo(v, 0.01)]);
        expect(got[k].score, closeTo((items[k]['score'] as num).toDouble(), 1e-6));
        final (x1, y1, x2, y2) = cropBounds(got[k], w, h, meta.cropScale);
        expect([x1, y1, x2, y2], items[k]['crop']);
        // Same crop pixels as Python (bit-exact for PNG; JPEG decoders may differ by a level).
        final t = cropTensor(im, x1, y1, x2, y2, clf.imageSize, clf.mean, clf.std);
        final n = t.length;
        expect(t.fold(0.0, (s, v) => s + v) / n,
            closeTo((items[k]['crop_tensor_sum'] as num) / n, 0.01));
        expect(t.fold(0.0, (s, v) => s + v.abs()) / n,
            closeTo((items[k]['crop_tensor_abs_sum'] as num) / n, 0.01));
      }
    });
  }
}
