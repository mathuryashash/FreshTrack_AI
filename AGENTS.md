# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

FreshTrack AI — An intelligent fruit quality assessment system using multi-task deep learning (EfficientNet-B0 backbone) to classify freshness (4 classes), grade quality (3 classes), and predict shelf life (regression) from a single image.

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
# Generate metadata for a dataset
python scripts/generate_metadata.py --data_dir <path> --output data/metadata.json

# Train model
python src/training/train.py --metadata data/metadata.json --epochs 10 --name stage1

# Sequential multi-stage training
python scripts/train_sequential.py
```

## Architecture

### Backend Stack
- **Framework**: FastAPI with rate limiting (slowapi), API key auth, CORS
- **Model**: PyTorch Lightning module (`src/models/freshtrack_model.py`) with EfficientNet-B0 backbone
- **Database**: SQLite (`src/api/database.py`) for prediction logging and feedback collection
- **Config**: Centralized in `src/config.py` with environment variable overrides

### API Endpoints
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | Status check |
| GET | `/health` | No | Model + DB health |
| POST | `/predict` | API Key | Upload image → freshness, quality, shelf_life |
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
MODEL_CHECKPOINT=models/checkpoints/freshtrack_epoch=04_val_loss=0.01-v1.ckpt
DATABASE_URL=data/freshtrack.db
CORS_ORIGINS=http://localhost:8501,http://localhost:3000
```

## Testing Notes

- API tests use `TestClient` with mocked DB calls (`monkeypatch`)
- Tests validate: content-type, file extension, size limits, image magic bytes
- Run: `pytest tests/test_api.py -v` or `pytest tests/test_model.py -v`
