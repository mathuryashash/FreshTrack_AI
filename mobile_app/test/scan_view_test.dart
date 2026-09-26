import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:freshtrack_mobile/services/classifier.dart';
import 'package:freshtrack_mobile/services/pipeline.dart';
import 'package:freshtrack_mobile/widgets/scan_view.dart';

Classification cls({bool produce = true, String freshness = 'Fresh'}) => Classification(
      isProduce: produce,
      energy: produce ? 8 : 2,
      freshness: freshness,
      freshnessConfidence: 0.9,
      pFresh: freshness == 'Fresh' ? 0.9 : 0.1,
      produceType: 'apple',
      produceConfidence: 0.95,
      quality: 'Good',
      shelfLifeDays: 5,
    );

ScanItem item(double x1, double y1, double x2, double y2, {bool produce = true}) =>
    ScanItem(Detection(x1, y1, x2, y2, 0.9), (x1.floor(), y1.floor(), x2.ceil(), y2.ceil()), cls(produce: produce));

// A 600 x 400 photo shown 300 px wide: scale 0.5, 200 px tall.
final photo = File('test/fixtures/parity/00_apple_fresh.png');

Future<Rect> pumpScan(WidgetTester tester, List<ScanItem> items,
    {bool selecting = false, ValueChanged<int>? onTap, ValueChanged<Detection>? onSelected}) async {
  await tester.pumpWidget(MaterialApp(
    home: Center(
      child: SizedBox(
        width: 300,
        child: ScanImage(
          image: photo,
          imageWidth: 600,
          imageHeight: 400,
          items: items,
          selecting: selecting,
          onTapItem: onTap,
          onSelected: onSelected,
        ),
      ),
    ),
  ));
  return tester.getRect(find.descendant(of: find.byType(ScanImage), matching: find.byType(GestureDetector)));
}

void main() {
  test('only recognised items are numbered', () {
    expect(itemNumbers([item(0, 0, 1, 1), item(0, 0, 1, 1, produce: false), item(0, 0, 1, 1)]), [1, null, 2]);
  });

  testWidgets('photo keeps its aspect ratio', (tester) async {
    final r = await pumpScan(tester, const []);
    expect(r.width, 300);
    expect(r.height, 200);
  });

  testWidgets('tapping a box opens it; overlapping boxes resolve to the smaller one', (tester) async {
    final tapped = <int>[];
    final r = await pumpScan(
      tester,
      [item(0, 0, 400, 300), item(100, 100, 200, 200)],
      onTap: tapped.add,
    );
    await tester.tapAt(r.topLeft + const Offset(75, 75)); // photo (150, 150): inside both
    await tester.tapAt(r.topLeft + const Offset(20, 20)); // photo (40, 40): only the big box
    await tester.tapAt(r.topLeft + const Offset(250, 180)); // outside every box
    expect(tapped, [1, 0]);
  });

  testWidgets('a drawn box is reported in photo pixels', (tester) async {
    Detection? got;
    final r = await pumpScan(tester, const [], selecting: true, onSelected: (d) => got = d);
    await tester.dragFrom(r.topLeft + const Offset(40, 30), const Offset(100, 80));
    await tester.pumpAndSettle();
    expect(got, isNotNull);
    expect(got!.x1, closeTo(80, 1e-6));
    expect(got!.y1, closeTo(60, 1e-6));
    expect(got!.x2, closeTo(280, 1e-6));
    expect(got!.y2, closeTo(220, 1e-6));
  });

  testWidgets('tiny drags are ignored and taps do nothing while selecting', (tester) async {
    Detection? got;
    var taps = 0;
    final r = await pumpScan(tester, [item(0, 0, 400, 300)],
        selecting: true, onSelected: (d) => got = d, onTap: (_) => taps++);
    await tester.dragFrom(r.topLeft + const Offset(40, 30), const Offset(10, 60));
    await tester.tapAt(r.topLeft + const Offset(20, 20));
    await tester.pumpAndSettle();
    expect(got, isNull);
    expect(taps, 0);
  });

  testWidgets('a drag past the edge is clipped to the photo', (tester) async {
    Detection? got;
    final r = await pumpScan(tester, const [], selecting: true, onSelected: (d) => got = d);
    await tester.dragFrom(r.topLeft + const Offset(250, 150), const Offset(200, 200));
    await tester.pumpAndSettle();
    expect(got!.x2, closeTo(600, 1e-6));
    expect(got!.y2, closeTo(400, 1e-6));
  });
}
