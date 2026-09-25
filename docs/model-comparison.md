# FreshTrack Model Comparison Framework

## Objective
Compare different ML models to find the optimal balance of:
- Accuracy (freshness classification)
- Inference speed (mobile responsiveness)
- Model size (app bundle size)

---

## Models to Compare

| Model | Params | ImageNet Acc | Inference (CPU) | TFLite Size |
|-------|--------|------------|---------------|------------|
| MobileNetV3-Small | 2.5M | 67.7% | ~15ms | ~4MB |
| MobileNetV3-Large | 5.5M | 75.2% | ~25ms | ~8MB |
| EfficientNet-B0 | 5.3M | 77.1% | ~45ms | ~7MB |
| EfficientNet-B1 | 7.8M | 79.2% | ~65ms | ~10MB |
| EfficientNet-B2 | 9.2M | 80.3% | ~85ms | ~12MB |

---

## Metrics to Track

### 1. Classification Accuracy
- **Freshness Accuracy** - % correct freshness classification
- **Quality Accuracy** - % correct quality grading
- **F1-Score** - Harmonic mean of precision/recall

### 2. Performance
- **Inference Time** - ms per image (GPU vs CPU)
- **Throughput** - images/second
- **Cold Start Time** - time to load model

### 3. Model Characteristics
- **Model Size** - MB on disk
- **Parameters** - trainable weights
- **Memory Usage** - RAM during inference

### 4. Business Metrics
- **User Satisfaction** - feedback correction rate
- **Prediction Confidence** - avg confidence scores
- **OOD Rejection Rate** - % of low-confidence predictions

---

## Testing Protocol

### A. Accuracy Testing
```python
# scripts/evaluate_model.py
python evaluate_model.py \
  --model models/checkpoints/freshtrack.ckpt \
  --test_data data/test \
  --output reports/accuracy_metrics.json
```

### B. Speed Testing
```python
# scripts/benchmark_inference.py
python benchmark_inference.py \
  --model models/checkpoints/freshtrack.ckpt \
  --iterations 1000 \
  --output reports/speed_metrics.json
```

### C. Mobile Benchmarking
```dart
// In Flutter app
final stopwatch = Stopwatch()..start();
final result = await model.predict(image);
stopwatch.stop();
final inferenceMs = stopwatch.elapsedMilliseconds;
```

---

## Comparison Dashboard

### Metrics to Log
```python
# API logs this automatically
{
  "model_version": "1.0.0",
  "inference_ms": 45.2,
  "freshness": "Fresh",
  "freshness_confidence": 0.92,
  "quality": "A",
  "shelf_life_days": 7.0,
  "entropy_score": 0.5,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Analytics Queries
```sql
-- Average inference time
SELECT AVG(inference_ms) FROM predictions;

-- Confidence distribution
SELECT 
  CASE 
    WHEN freshness_conf > 0.9 THEN 'high'
    WHEN freshness_conf > 0.7 THEN 'medium'
    ELSE 'low'
  END as conf_bucket,
  COUNT(*) as count
FROM predictions
GROUP BY 1;

-- User correction rate
SELECT 
  CAST(SUM(CASE WHEN f.correct_freshness != p.freshness THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)
FROM predictions p
JOIN feedback f ON p.id = f.prediction_id;
```

---

## Results Template

| Metric | MobileNetV3-S | EfficientNet-B0 | EfficientNet-B1 |
|--------|--------------|----------------|-----------------|
| Accuracy (Freshness) | 72.3% | 78.5% | 80.1% |
| Accuracy (Quality) | 68.1% | 74.2% | 76.8% |
| F1-Score | 0.71 | 0.77 | 0.79 |
| Inference (CPU) | 18ms | 48ms | 68ms |
| Model Size | 4.2MB | 6.8MB | 9.5MB |
| User Correction Rate | 15.2% | 8.1% | 6.3% |

---

## Decision Matrix

| Priority | Model Choice |
|----------|------------|
| Speed Critical | MobileNetV3-Small |
| Offline + Balanced | EfficientNet-B0 |
| Maximum Accuracy | EfficientNet-B2 |
| Best Value | MobileNetV3-Large |

---

## Recommended Next Steps

1. **Export models to TFLite** and benchmark actual mobile performance
2. **A/B test** with real users in production
3. **Track user corrections** to measure real-world accuracy
4. **Active learning loop** to improve from corrections