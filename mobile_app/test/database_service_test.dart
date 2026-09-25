import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:freshtrack_mobile/models/prediction_result.dart';
import 'package:freshtrack_mobile/services/database_service.dart';
import 'package:path/path.dart' as p;
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  late Directory tmp;

  setUpAll(() {
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfiNoIsolate;
  });

  setUp(() async {
    tmp = await Directory.systemTemp.createTemp('freshtrack_test');
    DatabaseService.docsDir = () async => tmp;
    DatabaseService.dbPath = p.join(tmp.path, 'test.db');
  });

  tearDown(() async {
    await DatabaseService.close();
    await tmp.delete(recursive: true);
  });

  PredictionResult result({String? id, String? imagePath}) => PredictionResult(
        id: id,
        freshness: 'Fresh',
        freshnessConfidence: 0.93,
        produceType: 'banana',
        quality: 'A',
        shelfLifeDays: 4.2,
        timestamp: DateTime(2026, 9, 25, 10),
        imagePath: imagePath,
      );

  test('insert copies the image to docs/scans/<id>.jpg and round-trips produce_type', () async {
    final picked = File(p.join(tmp.path, 'picker_cache.png'))..writeAsBytesSync([1, 2, 3]);
    final before = DatabaseService.changes.value;

    await DatabaseService.insert(result(id: 'p1', imagePath: picked.path));
    picked.deleteSync(); // temp file goes away; history must not care

    expect(DatabaseService.changes.value, before + 1);
    final rows = await DatabaseService.getRecent();
    expect(rows, hasLength(1));
    expect(rows.single.id, 'p1');
    expect(rows.single.produceType, 'banana');
    expect(rows.single.imagePath, p.join(tmp.path, 'scans', 'p1.jpg'));
    expect(File(rows.single.imagePath!).readAsBytesSync(), [1, 2, 3]);

    // Stored relative, so it survives a documents-directory move.
    final raw = await (await DatabaseService.db).query('predictions');
    expect(raw.single['image_path'], p.join('scans', 'p1.jpg'));
  });

  test('insert without a server id still gets a key and image', () async {
    final picked = File(p.join(tmp.path, 'x.jpg'))..writeAsBytesSync([9]);
    await DatabaseService.insert(result(imagePath: picked.path));
    final row = (await DatabaseService.getRecent()).single;
    expect(row.id, isNotNull);
    expect(File(row.imagePath!).existsSync(), isTrue);
  });

  test('clear deletes rows and scan files and notifies', () async {
    final picked = File(p.join(tmp.path, 'x.jpg'))..writeAsBytesSync([9]);
    await DatabaseService.insert(result(id: 'a', imagePath: picked.path));
    final before = DatabaseService.changes.value;

    await DatabaseService.clear();

    expect(await DatabaseService.getRecent(), isEmpty);
    expect(Directory(p.join(tmp.path, 'scans')).existsSync(), isFalse);
    expect(DatabaseService.changes.value, before + 1);
  });

  test('v1 -> v2 migration adds produce_type and keeps old rows', () async {
    final v1 = await databaseFactory.openDatabase(DatabaseService.dbPath!,
        options: OpenDatabaseOptions(
          version: 1,
          onCreate: (db, _) => db.execute('''
            CREATE TABLE predictions (
              id TEXT PRIMARY KEY, freshness TEXT, freshness_conf REAL, quality TEXT,
              shelf_life_days REAL, timestamp TEXT, image_path TEXT)'''),
        ));
    await v1.insert('predictions', {
      'id': 'old',
      'freshness': 'Stale',
      'freshness_conf': 0.8,
      'quality': 'C',
      'shelf_life_days': 1.0,
      'timestamp': '2026-01-01T00:00:00.000',
      'image_path': null,
    });
    await v1.close();

    final rows = await DatabaseService.getRecent();
    expect(rows.single.id, 'old');
    expect(rows.single.produceType, isNull);

    await DatabaseService.insert(result(id: 'new'));
    final all = await DatabaseService.getRecent();
    expect(all.map((r) => r.produceType), containsAll(['banana', null]));
    expect(await (await DatabaseService.db).getVersion(), 2);
  });
}
