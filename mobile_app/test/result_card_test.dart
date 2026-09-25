import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:freshtrack_mobile/models/prediction_result.dart';
import 'package:freshtrack_mobile/widgets/result_card.dart';

Future<void> pumpCard(WidgetTester tester, PredictionResult r) async {
  await tester.pumpWidget(MaterialApp(
    home: Scaffold(body: SingleChildScrollView(child: ResultCard(result: r))),
  ));
  await tester.pumpAndSettle();
}

PredictionResult make({String freshness = 'Fresh', double conf = 0.92, String? produce = 'bitter_gourd'}) =>
    PredictionResult(
      freshness: freshness,
      freshnessConfidence: conf,
      produceType: produce,
      quality: 'A',
      shelfLifeDays: 6.3,
      timestamp: DateTime(2026, 9, 25, 10, 30),
    );

void main() {
  testWidgets('shows produce type, freshness and confidence', (tester) async {
    await pumpCard(tester, make());
    expect(find.text('Produce'), findsOneWidget);
    expect(find.text('Bitter gourd'), findsOneWidget);
    expect(find.text('Fresh'), findsOneWidget);
    expect(find.text('92%'), findsOneWidget);
    expect(find.text('Looks fresh'), findsOneWidget);
  });

  testWidgets('labels quality and shelf life as estimates with a disclaimer', (tester) async {
    await pumpCard(tester, make());
    expect(find.text('Quality (est.)'), findsOneWidget);
    expect(find.text('Shelf Life (est.)'), findsOneWidget);
    expect(find.text('Shelf life remaining (est.)'), findsOneWidget);
    expect(find.text(PredictionResult.heuristicDisclaimer), findsOneWidget);
    // "not a food-safety test" is fine; "safe" / "safe to eat" claims are not.
    expect(find.textContaining(RegExp(r'\bsafe\b', caseSensitive: false)), findsNothing);
  });

  testWidgets('stale and low-confidence states', (tester) async {
    await pumpCard(tester, make(freshness: 'Stale', conf: 0.95));
    expect(find.text('Looks stale'), findsOneWidget);

    await pumpCard(tester, make(conf: 0.55));
    expect(find.text('Unsure: check manually'), findsOneWidget);
  });

  testWidgets('v1 rows without produce type hide the Produce row', (tester) async {
    await pumpCard(tester, make(produce: null));
    expect(find.text('Produce'), findsNothing);
    expect(find.text('Quality (est.)'), findsOneWidget);
  });
}
