import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../models/prediction_result.dart';

class DatabaseService {
  static Database? _db;

  static Future<Database> get db async {
    _db ??= await _init();
    return _db!;
  }

  static Future<Database> _init() async {
    final dbPath = await getDatabasesPath();
    return openDatabase(
      join(dbPath, 'freshtrack.db'),
      version: 1,
      onCreate: (db, version) => db.execute('''
        CREATE TABLE predictions (
          id            TEXT PRIMARY KEY,
          freshness     TEXT,
          freshness_conf REAL,
          quality       TEXT,
          shelf_life_days REAL,
          timestamp     TEXT,
          image_path    TEXT
        )
      '''),
      onUpgrade: (db, oldVersion, newVersion) async {
        // Add migration steps here as schema evolves
      },
    );
  }

  static Future<void> insert(PredictionResult result) async {
    final database = await db;
    await database.insert(
      'predictions',
      result.toDb(),
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  static Future<List<PredictionResult>> getRecent({int limit = 30}) async {
    final database = await db;
    final rows = await database.query(
      'predictions',
      orderBy: 'timestamp DESC',
      limit: limit,
    );
    return rows.map(PredictionResult.fromDb).toList();
  }

  static Future<void> clear() async {
    final database = await db;
    await database.delete('predictions');
  }
}
