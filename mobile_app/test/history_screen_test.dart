import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:freshtrack_mobile/models/prediction_result.dart';
import 'package:freshtrack_mobile/screens/history_screen.dart';
import 'package:freshtrack_mobile/services/database_service.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

// "Fake DB": the real DatabaseService on an in-memory sqflite_common_ffi
// database (no isolate, so it runs inside the widget-test zone).
void main() {
  setUpAll(() {
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfiNoIsolate;
    DatabaseService.dbPath = inMemoryDatabasePath;
    final docs = Directory.systemTemp.createTempSync('freshtrack_hist');
    DatabaseService.docsDir = () async => docs;
  });

  tearDown(() async {
    await DatabaseService.close();
  });

  PredictionResult row(String id, String produce, String freshness) => PredictionResult(
        id: id,
        freshness: freshness,
        freshnessConfidence: 0.88,
        produceType: produce,
        quality: 'B',
        shelfLifeDays: 3.5,
        timestamp: DateTime(2026, 9, 25, 9),
      );

  Future<void> settle(WidgetTester tester) async {
    for (var i = 0; i < 5; i++) {
      await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 10)));
      await tester.pump();
    }
  }

  testWidgets('shows empty state, then refreshes on insert without pull-to-refresh',
      (tester) async {
    await tester.runAsync(() => DatabaseService.db); // open outside the fake-async zone
    await tester.pumpWidget(const MaterialApp(home: HistoryScreen()));
    await settle(tester);
    expect(find.text('No scans yet'), findsOneWidget);

    // Insert while the screen stays mounted (as it does in the IndexedStack).
    await tester.runAsync(() => DatabaseService.insert(row('1', 'bitter_gourd', 'Stale')));
    await settle(tester);

    expect(find.text('No scans yet'), findsNothing);
    expect(find.text('Bitter gourd'), findsOneWidget);
    expect(find.text('Stale'), findsOneWidget);
    expect(find.text('Quality B · 3.5 days (est.)'), findsOneWidget);
    expect(find.text('88%'), findsOneWidget);
  });

  testWidgets('clear-all confirms, empties the list and refreshes', (tester) async {
    await tester.runAsync(() async {
      await DatabaseService.insert(row('1', 'apple', 'Fresh'));
      await DatabaseService.insert(row('2', 'tomato', 'Stale'));
    });
    await tester.pumpWidget(const MaterialApp(home: HistoryScreen()));
    await settle(tester);
    expect(find.text('Apple'), findsOneWidget);
    expect(find.text('Tomato'), findsOneWidget);

    await tester.tap(find.byIcon(Icons.delete_outline));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Clear'));
    await settle(tester);

    expect(find.text('No scans yet'), findsOneWidget);
  });
}
