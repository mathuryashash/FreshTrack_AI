class PredictionResult {
  final String freshness;
  final String qualityGrade;
  final double shelfLifeDays;
  final Map<String, dynamic> confidence;

  PredictionResult({
    required this.freshness,
    required this.qualityGrade,
    required this.shelfLifeDays,
    required this.confidence,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      freshness: json['freshness'] ?? 'Unknown',
      qualityGrade: json['quality_grade'] ?? 'Unknown',
      shelfLifeDays: (json['shelf_life_days'] ?? 0).toDouble(),
      confidence: json['confidence'] ?? {},
    );
  }
}
