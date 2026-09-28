# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FreshTrack AI — produce freshness assessment using multi-task deep learning (timm backbone: EfficientNet-B0 by default, MobileNetV3-Large for the paper's chosen model and the app). Learned tasks: freshness (Fresh/Stale) and produce type (6 classes: apple, banana, bitter gourd, capsicum, orange, tomato). Quality grade and shelf life are heuristics derived from P(fresh) and are flagged as such in the API. See DECISIONS.md §0.

The Android app (v2.1) runs offline in two stages: an SSDlite detector finds each item, then the classifier checks a square crop around it (DECISIONS.md §0.2). The served classifier is `deploy_mnv3_v210`, not the research model in the paper (§0.1–0.2).

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
# Export both models into the app (writes content-named .onnx files + metadata + parity fixtures)
python -m src.training.export_onnx
python -m src.detection.export

cd mobile_app
flutter pub get
flutter analyze && flutter test                  # unit + widget tests
flutter test integration_test/detector_parity_test.dart -d emulator-5554   # on-device parity
flutter build apk --release --split-per-abi
```

### Research results (paper)

```bash
# Build leakage-free grouped splits (needs data/downloads/ Kaggle folders)
python -m src.data.build_splits          # -> data/metadata_v2.json, data/metadata_external.json

# Train one run
python -m src.training.train --name b0_mtl_s0 --seed 0

# Full experiment matrix + baseline + aggregation (resumable)
python -m src.training.run_experiment    # -> models/runs/*, results/summary.{json,md}

# Evaluate specific runs (writes metrics.json + model_meta.json)
python -m src.training.evaluate models/runs/b0_mtl_s0

# Paper numbers, tables and figures (never type results by hand)
python paper/make_tables.py
```

### Deployment models (app)

The full sequence (deploy classifier, detector data stages, detector training, gate calibration, export) is in README.md "Serve a model". `src.training.evaluate` cannot read the deploy metadata; use `src.training.evaluate_deploy` and `src.training.gate_sweep`.

## Architecture

### Backend Stack
- **Framework**: FastAPI with rate limiting (slowapi), API key auth, CORS
- **Model**: PyTorch Lightning module (`src/models/freshtrack_model.py`), timm backbone
- **Detector**: `src/detection/` — torchvision SSDlite320-MobileNetV3, one "produce" class; `detector.py` is the reference pipeline the Dart code mirrors
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

The API classifies the whole photo; only the app uses the detector.

### Mobile App Structure
```
mobile_app/lib/
├── main.dart                # App entry, theme, bottom nav shell
├── screens/
│   ├── home_screen.dart     # Capture/pick photo, boxes + item list, Select area
│   ├── result_screen.dart   # Details for one item
│   ├── history_screen.dart  # Local SQLite scan history
│   └── settings_screen.dart # About + Clear history (nothing to configure)
├── services/
│   ├── classifier.dart      # ONNX Runtime sessions (detector + classifier), analyse()
│   ├── pipeline.dart        # Pure-Dart pre/post-processing, mirrors src/detection/detector.py
│   └── database_service.dart # Local SQLite history
├── models/
│   └── prediction_result.dart
└── widgets/
    ├── scan_view.dart       # Photo with tappable boxes, drag-to-select
    ├── freshness_badge.dart
    └── result_card.dart
```

### Data Flow
1. **Training**: Metadata JSON → `FruitDataset` (Albumentations transforms) → `FreshTrackModel` (multi-task heads) → PyTorch Lightning trainer
2. **Inference (API)**: Image upload → API validates → model predicts → SQLite log → JSON response
3. **Mobile**: Camera/gallery → detector (≤ 8 boxes) → square crop per box → classifier + crop OOD gate → results + history. No detection → whole photo with the whole-photo gate.

### Key Configuration
- `src/config.py`: label mappings, loss weights, image settings, paths
- `.env`: `API_KEY`, `MODEL_CHECKPOINT`, `MODEL_META`, `DATABASE_URL`, `CORS_ORIGINS`
- `mobile_app/assets/model/`: `model_meta.json` (classifier, whole-photo gate), `detector_meta.json` (score threshold, crop gate)
- `mobile_app/pubspec.yaml`: flutter_onnxruntime, image, image_picker, sqflite

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
- Run: `pytest tests/test_api.py -v`, `pytest tests/test_model.py -v`, `pytest tests/test_detection.py -v`
- ONNX files in the app must keep content-hashed names (the export scripts do this): flutter_onnxruntime reuses any cached file with the same name, so an update would keep the old model.
- Windows: run one GPU training at a time with `--num_workers 2` or less, or commit memory runs out.
