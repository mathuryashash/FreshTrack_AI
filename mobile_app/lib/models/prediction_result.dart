import 'package:intl/intl.dart';

class PredictionResult {
  final String? id;
  final String freshness;
  final double freshnessConfidence;
  final String quality;
  final double shelfLifeDays;
  final DateTime timestamp;
  final String? imagePath;

  const PredictionResult({
    this.id,
    required this.freshness,
    required this.freshnessConfidence,
    required this.quality,
    required this.shelfLifeDays,
    required this.timestamp,
    this.imagePath,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json, {String? imagePath}) {
    return PredictionResult(
      id: json['prediction_id'] as String?,
      freshness: json['freshness'] as String? ?? 'Unknown',
      freshnessConfidence: (json['freshness_confidence'] as num?)?.toDouble() ?? 0.0,
      quality: json['quality'] as String? ?? 'Unknown',
      shelfLifeDays: (json['shelf_life_days'] as num?)?.toDouble() ?? 0.0,
      timestamp: DateTime.now(),
      imagePath: imagePath,
    );
  }

  factory PredictionResult.fromDb(Map<String, dynamic> row) {
    return PredictionResult(
      id: row['id'] as String?,
      freshness: row['freshness'] as String? ?? 'Unknown',
      freshnessConfidence: (row['freshness_conf'] as num?)?.toDouble() ?? 0.0,
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
        'quality': quality,
        'shelf_life_days': shelfLifeDays,
        'timestamp': timestamp.toIso8601String(),
        'image_path': imagePath,
      };

  String get formattedDate => DateFormat('MMM d, h:mm a').format(timestamp);

  bool get isSafe =>
      (freshness == 'Fresh' || freshness == 'Semi-ripe') &&
      freshnessConfidence >= 0.70;

  String get safetyLabel {
    if (freshness == 'Fresh' && freshnessConfidence >= 0.70) return 'Safe to Eat';
    if (freshness == 'Semi-ripe' && freshnessConfidence >= 0.70) return 'Safe (Consume Soon)';
    if (freshness == 'Overripe') return 'Use Immediately';
    if (freshnessConfidence < 0.70) return 'Low Confidence — Check Manually';
    return 'Not Recommended';
  }
}
