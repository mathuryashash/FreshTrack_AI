# FreshTrack AI - App Status & Implementation Plan

## 1. Why Flutter - Benefits Summary

### Advantages of Flutter for FreshTrack

| Benefit | Description | Impact |
|----------|-------------|--------|
| **Single Codebase** | Write once, deploy to iOS + Android | 50% dev time saved |
| **Native Performance** | Skia rendering engine | 60fps smooth UI |
| **Rich Widgets** | Material Design + Cupertino | Beautiful UI |
| **tflite_flutter** | Native TensorFlow Lite support | **Offline inference** |
| **Hot Reload** | See changes instantly | Fast iteration |
| **Dart Language** | Easy to learn, type-safe | Quick onboarding |
| **Strong Typing** | Catch errors at compile time | Fewer bugs |
| **State Management** | Provider, Riverpod, Bloc | Scalable architecture |

### Comparison: Flutter vs React Native (Expo)

| Factor | Flutter | React Native (Expo) |
|--------|---------|-------------------|
| **Offline ML** | ✅ tflite_flutter | ⚠️ tfjs (limited) |
| **Camera API** | ✅ Mature | ✅ Good |
| **Build Speed** | Slower | ✅ Faster |
| **Learning Curve** | Medium | ✅ Easier (if you know React) |
| **Current Setup** | ✅ Working | Would need migration |

**Recommendation: Stay with Flutter** - Already working, excellent offline ML support.

---

## 2. App Status - How Far from Play Store?

### Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Backend API** | ✅ Working | FastAPI with model |
| **Database** | ✅ Working | SQLite + feedback |
| **Flutter App** | ⚠️ Partial | Needs OOD UI |
| **Model** | ✅ Working | EfficientNet-B0 |
| **TFLite Export** | ❌ Pending | For offline |
| **Security** | ✅ Good | API key auth, rate limiting |
| **Play Store** | ❌ Not Submitted | Needs work |

### What's Working Now

```
✅ REST API endpoints (/predict, /feedback, /health, /history, /stats)
✅ Model loading with verification
✅ SQLite database with prediction logging
✅ User feedback collection
✅ Rate limiting (slowapi)
✅ Security headers (CORS, CSP, HSTS)
✅ Health check endpoint
✅ Prediction confidence + entropy tracking
```

### What's Missing for Play Store

```
❌ Fun OOD pop-up (in progress)
❌ TFLite model for offline inference
❌ Privacy policy page
❌ App icon + screenshots
❌ Play Store listing
❌ A/B testing infrastructure
❌ Analytics dashboard
❌ Push notifications
```

---

## 3. Training Split Ratio

### Current: 70/15/15

The codebase uses **70% train, 15% validation, 15% test** split:

```python
# scripts/create_splits.py:11
def create_splits(metadata_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
```

### Distribution Logic

1. **Stratified**: By freshness class (Fresh/Rotten/Semi-ripe/Overripe)
2. **Deterministic**: Based on hash of file path (same every run)
3. **Random shuffle**: Within each class before split

### Class Distribution (Example)

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Fresh | 700 | 150 | 150 | 1,000 |
| Rotten | 700 | 150 | 150 | 1,000 |
| Semi-ripe | 350 | 75 | 75 | 500 |
| Overripe | 350 | 75 | 75 | 500 |
| **Total** | **2,100** | **450** | **450** | **3,000** |

---

## 4. API Flow - How It Works

### End-to-End Flow

```
USER                    FLUTTER APP                    BACKEND                  MODEL
  │                         │                           │                        │
  │  📸 Takes photo         │                           │                        │
  │─────────────────────────>│                           │                        │
  │                         │                           │                        │
  │                         │  ┌─ Convert to base64   │                        │
  │                         │  │  │                    │                        │
  │                         │  ▼                      │                        │
  │                         │ ────────────────────────>│                        │
  │                         │      POST /predict        │                        │
  │                         │      (multipart/form)     │                        │
  │                         │                           │                        │
  │                         │                           │  ┌─ Load model        │
  │                         │                           │  │ (once at startup)  │
  │                         │                           │  ▼                   │
  │                         │                           │ ─────────────────────>│
  │                         │                           │                       │
  │                         │                           │  ┌─ Preprocess        │
  │                         │                           │  │  (resize, normalize)│
  │                         │                           │  ▼                    │
  │                         │                           │ ─────────────────────>│
  │                         │                           │                       │
  │                         │                           │  ┌─ Forward pass      │
  │                         │                           │  │                   │
  │                         │                           │  ▼                   │
  │                         │                           │ ─────────────────────>│
  │                         │                           │                       │
  │                         │                           │ ◄────────────────────│
  │                         │       JSON response       │    │
  │                         │◄─────────────────────────│    │
  │                         │                           │    │
  │  ┌──────────────────────┤                           │    │
  │  │                      │                           │    │
  │  ▼                      │                           │    │
  │ Display results + heatmap│                           │    │
  │ (Fresh: 92%, Grade: A)   │                           │    │
  │                         │                           │    │
```

### API Request/Response

```http
POST /predict
Content-Type: multipart/form-data
X-API-Key: your_api_key

Response:
{
  "freshness": "Fresh",
  "freshness_confidence": 0.9234,
  "quality": "A",
  "shelf_life_days": 7.0,
  "prediction_id": "uuid-here",
  "heatmap_b64": "base64_encoded_heatmap"
}
```

### When OOD Detected

```
USER                    FLUTTER APP                    BACKEND
  │                         │                           │
  │  📸 Takes photo         │                           │
  │─────────────────────────>│                           │
  │                         │                           │
  │                         │ ────────────────────────>│
  │                         │      POST /predict        │
  │                         │                           │
  │                         │                           │  ┌─ OOD Check
  │                         │                           │  │  confidence < 0.6
  │                         │                           │  │  OR entropy > 2.5
  │                         │                           │  ▼
  │                         │◄───────────────────────────│
  │                         │    error response         │
  │                         │    "OBJECT_NOT_RECOGNIZED" │
  │                         │                           │
  │  ┌──────────────────────┤                           │
  │  │                      │                           │
  │  ▼                      │                           │
  │ 🎉 FUN POP-UP!          │                           │
  │ "Oops! That's not       │                           │
  │  a fruit! 🍌🍎🍊       │                           │
  │  Try with a fruit      │                           │
  │  or vegetable!"        │                           │
```

---

## 5. Play Store Security Requirements

### Google Play Policy Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| **Data Safety Form** | ✅ Need to complete | Declare data collection |
| **Privacy Policy** | ✅ Need to create | Publish on website |
| **HTTPS** | ✅ Enabled | All API calls secure |
| **API Key Auth** | ✅ Enabled | Rate limited |
| **No Sensitive Data** | ✅ No PII collected | Predictions only |
| **ML Model Disclosure** | ✅ Will complete | Explain AI usage |
| **Children's Privacy** | ✅ Not targeted | No under 13 |

### Data Safety Form Answers

| Question | Answer |
|----------|--------|
| Data collected? | No personal data |
| Data shared? | No third parties |
| Encryption? | HTTPS in transit |
| Delete data? | Local only, no account |
| Sensitive info? | No |

### Privacy Policy Template

```markdown
# FreshTrack AI Privacy Policy

Last Updated: [DATE]

FreshTrack AI respects your privacy.

## Data Collection
We do not collect personally identifiable information.
We only process images locally on our servers for fruit freshness detection.

## Data Use
Images are used solely to provide the freshness prediction service.
We do not use your images for any other purpose.

## Data Storage
Predictions are stored locally on your device.
No image data is transmitted to third parties.

## Contact
For questions, contact [EMAIL]
```

---

## 6. OOD - Fun Pop-Up Implementation

### Current (Boring Error)

```dart
// mobile_app/lib/screens/home_screen.dart:318
class _ErrorCard extends StatelessWidget {
  // ... red error box with "error" text
}
```

### New (Fun Pop-Up)

The OOD check returns: `"OBJECT_NOT_RECOGNIZED"` when confidence < 60%

**Implementation needed:**
1. Check for OOD error in response
2. Show fun animated pop-up
3. Add retry button

---

## 7. Implementation Roadmap

### Phase 1: Core Features (Current)
- [x] Backend API
- [x] Model training
- [x] Database
- [x] Feedback

### Phase 2: Polish (This Week)
- [ ] Fun OOD pop-up ← IN PROGRESS
- [ ] Privacy policy
- [ ] TFLite export

### Phase 3: Play Store (Next Week)
- [ ] App icon + screenshots
- [ ] Play Store listing
- [ ] Privacy policy URL

### Phase 4: Research Paper
- [ ] Document architecture
- [ ] Experimental results
- [ ] Submit to conference

---

## 8. Model Comparison Summary

| Model | Type | Accuracy | Speed | Size | Use Case |
|-------|------|----------|-------|------|---------|
| **EfficientNet-B2** | Accuracy+ | 80.3% | 85ms | 12MB | Server (this guide) |
| **EfficientNet-B0** | Balanced | 77.1% | 45ms | 7MB | Current |
| **MobileNetV3-Large** | Speed+ | 75.2% | 25ms | 8MB | Mobile offline (this guide) |

---

## Files Created

1. **`docs/implementation-guide-model-b2.md`** - EfficientNet-B2 guide
2. **`docs/implementation-guide-mobilenet.md`** - MobileNetV3-Large guide
3. **`docs/model-analytics.md`** - Analytics framework

---

## Questions for Clarification

1. **Training data**: How many images per class do you have?
2. **Target accuracy**: What's the minimum accuracy you need?
3. **Timeline**: When do you want to submit to Play Store?
4. **Research paper**: Which conference/journal?

