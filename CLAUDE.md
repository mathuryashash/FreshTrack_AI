# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FreshTrack AI — produce freshness assessment using multi-task deep learning (EfficientNet-B0 backbone). Learned tasks: freshness (Fresh/Stale) and produce type (6 classes). Quality grade and shelf life are heuristics derived from P(fresh) and are flagged as such in the API. See DECISIONS.md §0.

## Commands

### Backend (Python/FastAPI)

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Run Streamlit frontend
streamlit run src/app.py

# Build Docker image
docker build -t freshtrack-api .
```

### Mobile App (Flutter)

```bash
cd mobile_app

# Install dependencies
flutter pub get

# Run on device/emulator
flutter run

# Build APK
flutter build apk

# Run tests
flutter test
```

### Dataset & Training

```bash
# Build leakage-free grouped splits (needs data/downloads/ Kaggle folders)
python -m src.data.build_splits          # -> data/metadata_v2.json, data/metadata_external.json

# Train one run
python -m src.training.train --name b0_mtl_s0 --seed 0

# Full experiment matrix + baseline + aggregation (resumable)
python -m src.training.run_experiment    # -> models/runs/*, results/summary.{json,md}

# Evaluate specific runs (writes metrics.json + model_meta.json)
python -m src.training.evaluate models/runs/b0_mtl_s0

# Promote a run for serving
cp models/runs/mnv3_mtl_s1/best.ckpt models/checkpoints/freshtrack_v2.ckpt
cp models/runs/mnv3_mtl_s1/model_meta.json models/checkpoints/model_meta.json
```

## Architecture

### Backend Stack
- **Framework**: FastAPI with rate limiting (slowapi), API key auth, CORS
- **Model**: PyTorch Lightning module (`src/models/freshtrack_model.py`), timm backbone (deployed: MobileNetV3-Large; EfficientNet-B0 also supported)
- **Database**: SQLite (`src/api/database.py`) for prediction logging and feedback collection
- **Config**: Centralized in `src/config.py` with environment variable overrides

### API Endpoints
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | Status check |
| GET | `/health` | No | Model + DB health |
| POST | `/predict` | API Key | Upload image → freshness, produce_type, heuristic quality/shelf_life, OOD score |
| POST | `/feedback` | API Key | Submit corrected predictions |
| GET | `/history` | API Key | Recent predictions |
| GET | `/stats` | API Key | Aggregate statistics |

### Mobile App Structure
```
mobile_app/lib/
├── main.dart              # App entry, theme, bottom nav shell
├── screens/
│   ├── home_screen.dart   # Camera capture + prediction display
│   ├── history_screen.dart # Local SQLite prediction history
│   └── settings_screen.dart # API URL/key configuration
├── services/
│   ├── api_service.dart   # HTTP client with retry logic, image compression
│   └── database_service.dart # Local SQLite for offline history
├── models/
│   └── prediction_result.dart
└── widgets/
    ├── freshness_badge.dart
    └── result_card.dart
```

### Data Flow
1. **Training**: Metadata JSON → `FruitDataset` (Albumentations transforms) → `FreshTrackModel` (multi-task heads) → PyTorch Lightning trainer
2. **Inference**: Image upload → API validates → model predicts → SQLite log → JSON response
3. **Mobile**: Camera/gallery → compress → POST /predict → display results → cache locally

### Key Configuration
- `src/config.py`: Freshness/quality label mappings, loss weights, image settings
- `.env`: `API_KEY`, `MODEL_CHECKPOINT`, `DATABASE_URL`, `CORS_ORIGINS`
- `mobile_app/pubspec.yaml`: Flutter dependencies (http, sqflite, image_picker)

## Environment Variables

```bash
# API (required for production)
API_KEY=<your_secret_key>
MODEL_CHECKPOINT=models/checkpoints/freshtrack_v2.ckpt
MODEL_META=models/checkpoints/model_meta.json
DATABASE_URL=data/freshtrack.db
CORS_ORIGINS=http://localhost:8501,http://localhost:3000
```

## Testing Notes

- API tests use `TestClient` with mocked DB calls (`monkeypatch`)
- Tests validate: content-type, file extension, size limits, image magic bytes
- Run: `pytest tests/test_api.py -v` or `pytest tests/test_model.py -v`
