import 'package:intl/intl.dart';

class PredictionResult {
  final String? id;
  final String freshness;
  final double freshnessConfidence;
  final String? produceType; // apple | banana | bitter_gourd | ...
  final String quality;
  final double shelfLifeDays;
  final DateTime timestamp;
  final String? imagePath;

  const PredictionResult({
    this.id,
    required this.freshness,
    required this.freshnessConfidence,
    this.produceType,
    required this.quality,
    required this.shelfLifeDays,
    required this.timestamp,
    this.imagePath,
  });

  factory PredictionResult.fromDb(Map<String, dynamic> row) {
    return PredictionResult(
      id: row['id'] as String?,
      freshness: row['freshness'] as String? ?? 'Unknown',
      freshnessConfidence: (row['freshness_conf'] as num?)?.toDouble() ?? 0.0,
      produceType: row['produce_type'] as String?,
      quality: row['quality'] as String? ?? 'Unknown',
      shelfLifeDays: (row['shelf_life_days'] as num?)?.toDouble() ?? 0.0,
      timestamp: DateTime.tryParse(row['timestamp'] as String? ?? '') ?? DateTime.now(),
      imagePath: row['image_path'] as String?,
    );
  }

  Map<String, dynamic> toDb() => {
        'id': id,
        'freshness': freshness,
        'freshness_conf': freshnessConfidence,
        'produce_type': produceType,
        'quality': quality,
        'shelf_life_days': shelfLifeDays,
        'timestamp': timestamp.toIso8601String(),
        'image_path': imagePath,
      };

  String get formattedDate => DateFormat('MMM d, h:mm a').format(timestamp);

  /// 'bitter_gourd' -> 'Bitter gourd'; null for v1 history rows.
  String? get produceLabel {
    final t = produceType;
    if (t == null || t.isEmpty) return null;
    return t[0].toUpperCase() + t.substring(1).replaceAll('_', ' ');
  }

  // Visual freshness only: the model makes no food-safety judgement.
  bool get isConfident => freshnessConfidence >= 0.70;
  bool get looksFresh => freshness == 'Fresh' && isConfident;

  String get statusLabel {
    if (!isConfident) return 'Unsure: check manually';
    return freshness == 'Fresh' ? 'Looks fresh' : 'Looks stale';
  }

  /// Quality and shelf life are derived from P(fresh) on the device, not learned.
  static const heuristicDisclaimer =
      'Quality and shelf life are estimates from the freshness score, '
      'not a food-safety test. Check smell, texture and mould.';
}
