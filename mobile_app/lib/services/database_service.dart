import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import '../models/prediction_result.dart';

class DatabaseService {
  static Future<Database>? _db;

  /// Bumped after every insert/clear. Screens kept alive in the IndexedStack
  /// listen to it to re-query.
  static final ValueNotifier<int> changes = ValueNotifier(0);

  /// Test seams: DB path (null = app default) and the documents directory.
  @visibleForTesting
  static String? dbPath;
  @visibleForTesting
  static Future<Directory> Function() docsDir = getApplicationDocumentsDirectory;

  static Future<Database> get db => _db ??= _init();

  static Future<Database> _init() async {
    return openDatabase(
      dbPath ?? p.join(await getDatabasesPath(), 'freshtrack.db'),
      version: 2,
      onCreate: (db, version) => db.execute('''
        CREATE TABLE predictions (
          id            TEXT PRIMARY KEY,
          freshness     TEXT,
          freshness_conf REAL,
          quality       TEXT,
          shelf_life_days REAL,
          timestamp     TEXT,
          image_path    TEXT,
          produce_type  TEXT
        )
      '''),
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          await db.execute('ALTER TABLE predictions ADD COLUMN produce_type TEXT');
        }
      },
    );
  }

  @visibleForTesting
  static Future<void> close() async {
    final d = _db;
    _db = null;
    if (d != null) await (await d).close();
  }

  static Future<Directory> _scansDir() async =>
      Directory(p.join((await docsDir()).path, 'scans'));

  /// Saves the row. If [result.imagePath] points at a (temporary) picker file,
  /// it is copied to `<documents>/scans/<id>.jpg` and stored as a path relative
  /// to the documents directory, which survives cache cleanup and iOS
  /// container moves.
  static Future<void> insert(PredictionResult result) async {
    final id = result.id ?? DateTime.now().microsecondsSinceEpoch.toString();
    String? stored;
    final src = result.imagePath;
    if (src != null && await File(src).exists()) {
      final dir = await _scansDir();
      await dir.create(recursive: true);
      await File(src).copy(p.join(dir.path, '$id.jpg'));
      stored = p.join('scans', '$id.jpg');
    }
    await (await db).insert(
      'predictions',
      {...result.toDb(), 'id': id, 'image_path': stored},
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
    changes.value++;
  }

  /// Rows come back with absolute image paths.
  static Future<List<PredictionResult>> getRecent({int limit = 30}) async {
    final rows = await (await db).query(
      'predictions',
      orderBy: 'timestamp DESC',
      limit: limit,
    );
    final docs = (await docsDir()).path;
    return rows.map((r) {
      final rel = r['image_path'] as String?;
      // p.join keeps v1 rows' absolute paths unchanged.
      return PredictionResult.fromDb({...r, 'image_path': rel == null ? null : p.join(docs, rel)});
    }).toList();
  }

  static Future<void> clear() async {
    await (await db).delete('predictions');
    final dir = await _scansDir();
    if (await dir.exists()) await dir.delete(recursive: true);
    changes.value++;
  }
}
