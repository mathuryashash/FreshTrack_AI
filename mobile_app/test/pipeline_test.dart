import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:freshtrack_mobile/services/pipeline.dart';
import 'package:image/image.dart' as img;

// Goldens from cv2.resize(src, dsize, interpolation=cv2.INTER_LINEAR),
// opencv-python 4.13.0, np.random.default_rng(7).
const downSrc = [139, 74, 229, 241, 169, 65, 6, 160, 149, 106, 38, 175, 188, 205, 175, 229, 98, 249, 10, 148, 95, 86, 147, 198, 66, 39, 106, 213, 171, 45, 167, 57, 69, 81, 55, 14, 153, 178, 215, 76, 52, 39, 250, 72, 215, 50, 161, 223, 112, 172, 161, 233, 38, 17, 89, 1, 221, 6, 242, 127, 143, 6, 60, 210, 94, 25, 166, 33, 249, 189, 12, 204, 33, 57, 124, 30, 199, 149, 202, 119, 72, 220, 3, 209, 122, 136, 147, 77, 133, 82, 115, 87, 162, 230, 70, 71, 240, 62, 47, 184, 31, 34, 63, 65, 146, 197, 142, 253, 81, 133, 240, 113, 30, 247, 103, 122, 31, 19, 42, 129, 141, 136, 33, 149, 161, 0, 178, 141, 107, 35, 107, 130, 72, 27, 217, 254, 99, 13, 195, 206, 59, 228, 235, 202, 183, 47, 69, 179, 82, 35, 71, 159, 110, 121, 77, 87, 3, 126, 44, 253, 233, 225, 90, 119, 137, 120, 30, 55, 33, 165, 87, 216, 229, 167, 3, 41, 109, 203, 129, 219, 63, 101, 207, 156, 27, 220, 93, 29, 143, 200, 63, 11, 11, 180, 219, 113, 190, 87, 34, 9, 133, 29, 61, 36, 244, 192, 207, 131, 254, 67, 99, 248, 47, 71, 89, 119, 137, 85, 246, 206, 214, 129, 203, 234, 113, 121, 206, 210, 200, 248, 20, 161, 52, 114, 255, 112, 206, 54, 157, 131, 133, 69, 42, 68, 243, 24, 51, 127, 8, 205, 30, 97, 79, 35, 93, 63, 131, 188, 69, 254, 233, 238, 4, 3, 167, 155, 225, 24, 84, 68, 65, 49, 124, 147, 17, 248, 94, 4, 41, 177, 159, 126, 192, 225, 86, 246, 90, 51, 34, 231, 145, 184, 128, 238, 153, 94, 63, 13, 48, 125, 47, 186, 244, 0, 105, 58, 23, 158, 13, 2, 126, 212, 229, 139, 231, 169, 235, 194, 138, 39, 29, 67, 167, 136, 85, 99, 129, 68, 235, 102, 3, 247, 178, 114, 93, 225, 31, 230, 172, 47, 136, 166, 129, 130, 134, 151, 147, 240, 170, 214, 222];
const downRef = [193, 113, 135, 85, 94, 170, 194, 81, 233, 100, 107, 157, 194, 114, 98, 91, 109, 107, 204, 54, 149, 90, 143, 118, 191, 55, 110, 167, 29, 47, 169, 112, 247, 89, 80, 151, 105, 120, 92, 156, 123, 79, 102, 81, 197, 196, 188, 107, 172, 49, 95, 99, 121, 100, 87, 110, 123, 171, 132, 142, 126, 242, 76, 155, 103, 231, 172, 177, 153, 168, 153, 69, 132, 106, 153, 110, 202, 45, 157, 83, 101, 83, 128, 161, 174, 73, 93, 34, 88, 92, 174, 149, 189, 111, 153, 106, 65, 222, 112, 116, 101, 182, 59, 142, 168, 123, 167, 198];
const upSrc = [17, 110, 58, 181, 22, 129, 196, 163, 116, 254, 199, 10, 104, 179, 228, 189, 201, 43, 56, 121, 135, 65, 108, 23, 236, 144, 150, 62, 199, 102, 136, 138, 180, 184, 14, 186];
const upRef = [17, 110, 58, 58, 88, 76, 140, 44, 111, 185, 57, 126, 192, 128, 119, 210, 172, 89, 239, 190, 36, 254, 199, 10, 22, 114, 69, 62, 94, 82, 142, 53, 110, 183, 65, 122, 186, 129, 119, 201, 169, 90, 228, 185, 37, 242, 193, 11, 55, 140, 132, 87, 130, 122, 152, 110, 102, 172, 111, 99, 147, 133, 116, 144, 148, 97, 162, 155, 43, 171, 159, 16, 88, 166, 196, 113, 166, 162, 163, 167, 93, 161, 158, 77, 109, 138, 113, 87, 128, 104, 96, 126, 48, 100, 125, 20, 129, 172, 213, 138, 179, 173, 156, 193, 94, 142, 181, 76, 95, 143, 121, 75, 116, 121, 83, 99, 76, 87, 90, 53, 178, 159, 184, 163, 169, 157, 133, 190, 103, 113, 182, 97, 105, 148, 139, 109, 112, 149, 124, 74, 126, 132, 55, 115, 228, 146, 155, 188, 159, 141, 109, 186, 112, 85, 184, 118, 116, 152, 157, 142, 108, 177, 165, 49, 176, 177, 20, 176, 236, 144, 150, 192, 158, 138, 105, 185, 114, 80, 184, 121, 117, 153, 160, 148, 107, 181, 172, 45, 184, 184, 14, 186];

const identity = [0.0, 0.0, 0.0], unit = [1.0, 1.0, 1.0];

/// NCHW [0,1] floats back to packed HWC bytes.
List<int> toHwcBytes(Float32List t, int size) {
  final plane = size * size;
  return [
    for (var i = 0; i < plane; i++)
      for (var c = 0; c < 3; c++) (t[c * plane + i] * 255).round(),
  ];
}

void main() {
  final meta = ModelMeta.fromJson(
      jsonDecode(File('assets/model/model_meta.json').readAsStringSync()) as Map<String, dynamic>);

  group('ModelMeta (bundled asset)', () {
    test('labels, input contract and gate are what the pipeline expects', () {
      expect(meta.freshnessLabels, ['Fresh', 'Stale']);
      expect(meta.produceTypes, ['apple', 'banana', 'bitter_gourd', 'capsicum', 'orange', 'tomato']);
      expect(meta.imageSize, 224);
      expect(meta.inputName, 'input');
      expect(meta.outputNames, ['freshness', 'produce_type']);
      expect(meta.mean, [0.485, 0.456, 0.406]);
      expect(meta.std, [0.229, 0.224, 0.225]);
      expect(meta.oodThreshold, closeTo(5.0, 1e-3));
      expect(meta.version, startsWith('mobilenetv3_large_100 @ '));
    });

    test('rejects an OOD score it does not implement', () {
      final j = jsonDecode(File('assets/model/model_meta.json').readAsStringSync()) as Map<String, dynamic>;
      expect(() => ModelMeta.fromJson({...j, 'ood_score': 'msp'}), throwsFormatException);
    });
  });

  group('resize vs cv2.INTER_LINEAR', () {
    test('downscale 13x9 -> 6x6 is bit-exact', () {
      final t = toTensor(Uint8List.fromList(downSrc), 13, 9, 6, identity, unit);
      expect(toHwcBytes(t, 6), downRef);
    });

    test('upscale 4x3 -> 8x8 is within 1/255', () {
      // OpenCV finishes rows narrower than a SIMD register with a scalar loop
      // that rounds differently, so tiny images can differ by one level.
      final got = toHwcBytes(toTensor(Uint8List.fromList(upSrc), 4, 3, 8, identity, unit), 8);
      for (var i = 0; i < got.length; i++) {
        expect((got[i] - upRef[i]).abs(), lessThanOrEqualTo(1), reason: 'index $i');
      }
    });
  });

  group('preprocess', () {
    test('normalises with ImageNet mean/std into NCHW', () {
      // 2x2 solid (255, 0, 128) -> every value of a channel is identical.
      final im = img.Image(width: 2, height: 2)..clear(img.ColorRgb8(255, 0, 128));
      final t = preprocess(img.encodePng(im), 4, meta.mean, meta.std);
      expect(t.length, 3 * 4 * 4);
      expect(t[0], closeTo((1.0 - 0.485) / 0.229, 1e-5)); // R plane
      expect(t[16], closeTo((0.0 - 0.456) / 0.224, 1e-5)); // G plane
      expect(t[32], closeTo((128 / 255 - 0.406) / 0.225, 1e-5)); // B plane
    });

    test('drops alpha without compositing, like cv2.imread', () {
      final im = img.Image(width: 2, height: 2, numChannels: 4)..clear(img.ColorRgba8(10, 20, 30, 0));
      final t = preprocess(img.encodePng(im), 2, identity, unit);
      expect(toHwcBytes(t, 2).sublist(0, 3), [10, 20, 30]);
    });

    test('grayscale is replicated to RGB', () {
      final im = img.Image(width: 2, height: 2, numChannels: 1)..clear(img.ColorUint8.rgb(77, 77, 77));
      final t = preprocess(img.encodePng(im), 2, identity, unit);
      expect(toHwcBytes(t, 2).sublist(0, 3), [77, 77, 77]);
    });

    test('applies EXIF orientation before resizing', () {
      // Stored landscape: left half red, right half blue. Orientation 6 means
      // "rotate 90 deg clockwise to display", so red must end up on top.
      final im = img.Image(width: 8, height: 4);
      for (final p in im) {
        p
          ..r = p.x < 4 ? 255 : 0
          ..g = 0
          ..b = p.x < 4 ? 0 : 255;
      }
      im.exif.imageIfd.orientation = 6;
      final t = preprocess(img.encodeJpg(im, quality: 100), 8, identity, unit);
      final px = toHwcBytes(t, 8);
      final top = px.sublist(0, 3), bottom = px.sublist(px.length - 3);
      expect(top[0], greaterThan(200));
      expect(top[2], lessThan(60));
      expect(bottom[0], lessThan(60));
      expect(bottom[2], greaterThan(200));
    });

    test('rejects bytes that are not an image', () {
      expect(() => preprocess(Uint8List.fromList(utf8.encode('not an image')), 4, identity, unit),
          throwsFormatException);
    });
  });

  group('softmax / energy / argmax', () {
    test('softmax sums to 1 and is stable for large logits', () {
      final p = softmax([1000.0, 1000.0]);
      expect(p, [0.5, 0.5]);
      final q = softmax([2.0, 1.0, 0.1]);
      expect(q.reduce((a, b) => a + b), closeTo(1, 1e-12));
      expect(q[0], closeTo(0.659001, 1e-6));
    });

    test('energy is logsumexp', () {
      expect(energy([0.0, 0.0]), closeTo(0.693147, 1e-6));
      expect(energy([1000.0, 0.0]), closeTo(1000.0, 1e-9)); // no overflow
      expect(energy([1.0, 2.0, 3.0]), closeTo(3.407606, 1e-6));
    });

    test('argmax returns the first maximum', () {
      expect(argmax([0.1, 0.7, 0.7]), 1);
      expect(argmax([3.0]), 0);
    });
  });

  group('OOD gate', () {
    // Real PyTorch logits for 00_apple_fresh.png from test/fixtures/parity/expected.json.
    const appleFresh = [11.20069408416748, -9.080162048339844];
    const appleType = [13.88315486907959, -3.259221076965332, -3.1770052909851074, -3.816282033920288,
      -2.6863279342651367, -6.014338970184326];

    test('in-distribution produce passes', () {
      final c = postprocess(meta, appleFresh, appleType);
      expect(c.isProduce, isTrue);
      expect(c.energy, closeTo(13.883, 1e-3));
      expect(c.freshness, 'Fresh');
      expect(c.produceType, 'apple');
      expect(c.quality, 'High (A)');
      expect(c.shelfLifeDays, 10.0);
    });

    test('flat low logits are rejected as not produce', () {
      final c = postprocess(meta, appleFresh, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]); // energy = 1 + ln 6
      expect(c.isProduce, isFalse);
    });

    test('threshold is inclusive (server rejects only energy < threshold)', () {
      final t = meta.oodThreshold;
      // A single dominant logit makes energy == that logit (to 1e-9).
      final at = postprocess(meta, appleFresh, [t, -1e9, -1e9, -1e9, -1e9, -1e9]);
      expect(at.isProduce, isTrue);
      final below = postprocess(meta, appleFresh, [t - 1e-6, -1e9, -1e9, -1e9, -1e9, -1e9]);
      expect(below.isProduce, isFalse);
    });
  });

  group('heuristic quality / shelf life', () {
    test('quality bands from P(fresh)', () {
      expect(qualityFromPFresh(meta, 1.0), 'High (A)');
      expect(qualityFromPFresh(meta, 0.85), 'High (A)');
      expect(qualityFromPFresh(meta, 0.8499), 'Medium (B)');
      expect(qualityFromPFresh(meta, 0.5), 'Medium (B)');
      expect(qualityFromPFresh(meta, 0.4999), 'Low (C)');
      expect(qualityFromPFresh(meta, 0.0), 'Low (C)');
    });

    test('shelf life = reference days x P(fresh), 1 decimal', () {
      expect(shelfLifeDays(meta, 'banana', 0.93), 3.7); // 3.72
      expect(shelfLifeDays(meta, 'apple', 0.456), 4.6); // 4.56
      expect(shelfLifeDays(meta, 'bitter_gourd', 0.1), 0.3);
      expect(shelfLifeDays(meta, 'tomato', 0.0), 0.0);
    });

    test('stale prediction uses P(fresh), not the winning-class confidence', () {
      // softmax([0, ln 9]) = [0.1, 0.9]: Stale at 90%, P(fresh) = 0.1.
      final c = postprocess(meta, [0.0, 2.1972245773362196], [0.0, 20.0, 0.0, 0.0, 0.0, 0.0]);
      expect(c.freshness, 'Stale');
      expect(c.freshnessConfidence, closeTo(0.9, 1e-9));
      expect(c.pFresh, closeTo(0.1, 1e-9));
      expect(c.quality, 'Low (C)');
      expect(c.shelfLifeDays, 0.4); // banana 4 d x 0.1
      final r = c.toResult(imagePath: '/x.jpg');
      expect(r.produceType, 'banana');
      expect(r.quality, 'Low (C)');
      expect(r.imagePath, '/x.jpg');
    });
  });
}
