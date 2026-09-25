import 'package:flutter_test/flutter_test.dart';
import 'package:freshtrack_mobile/models/prediction_result.dart';

PredictionResult make({String freshness = 'Fresh', double conf = 0.9, String? produce = 'tomato'}) =>
    PredictionResult(
      freshness: freshness,
      freshnessConfidence: conf,
      produceType: produce,
      quality: 'A',
      shelfLifeDays: 5,
      timestamp: DateTime(2026, 9, 25),
    );

void main() {
  group('PredictionResult', () {
    test('toDb/fromDb round-trips every field including produce_type', () {
      final r = PredictionResult(
        id: 'abc-123',
        freshness: 'Stale',
        freshnessConfidence: 0.91,
        produceType: 'bitter_gourd',
        quality: 'Low (C)',
        shelfLifeDays: 0.4,
        timestamp: DateTime(2026, 9, 25, 10, 30),
        imagePath: 'scans/abc-123.jpg',
      );
      final back = PredictionResult.fromDb(r.toDb());
      expect(back.id, r.id);
      expect(back.freshness, r.freshness);
      expect(back.freshnessConfidence, r.freshnessConfidence);
      expect(back.produceType, 'bitter_gourd');
      expect(back.quality, r.quality);
      expect(back.shelfLifeDays, r.shelfLifeDays);
      expect(back.timestamp, r.timestamp);
      expect(back.imagePath, r.imagePath);
    });

    test('fromDb tolerates v1 rows without produce_type', () {
      final row = make().toDb()..remove('produce_type');
      expect(PredictionResult.fromDb(row).produceType, isNull);
      expect(PredictionResult.fromDb(row).produceLabel, isNull);
    });

    test('produceLabel humanises snake_case labels', () {
      expect(make(produce: 'bitter_gourd').produceLabel, 'Bitter gourd');
      expect(make(produce: 'apple').produceLabel, 'Apple');
      expect(make(produce: '').produceLabel, isNull);
    });
  });

  group('heuristic / status labels', () {
    test('confident fresh and stale', () {
      expect(make(freshness: 'Fresh', conf: 0.9).statusLabel, 'Looks fresh');
      expect(make(freshness: 'Stale', conf: 0.9).statusLabel, 'Looks stale');
      expect(make(freshness: 'Fresh', conf: 0.9).looksFresh, isTrue);
      expect(make(freshness: 'Stale', conf: 0.9).looksFresh, isFalse);
    });

    test('low confidence asks for a manual check (boundary 0.70)', () {
      expect(make(conf: 0.69).statusLabel, 'Unsure: check manually');
      expect(make(conf: 0.69).looksFresh, isFalse);
      expect(make(conf: 0.70).statusLabel, 'Looks fresh');
    });

    test('no label makes a food-safety claim', () {
      final labels = [
        for (final f in ['Fresh', 'Stale', 'Unknown'])
          for (final c in [0.1, 0.7, 0.99]) make(freshness: f, conf: c).statusLabel,
        PredictionResult.heuristicDisclaimer,
      ];
      for (final l in labels) {
        expect(l.toLowerCase(), isNot(contains('safe to eat')));
        expect(l.toLowerCase(), isNot(matches(RegExp(r'\bsafe\b'))));
      }
      expect(PredictionResult.heuristicDisclaimer, contains('estimates'));
    });
  });
}
