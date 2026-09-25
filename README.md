# FreshTrack AI

Produce freshness assessment from a single photo, using multi-task deep learning.

- **Learned tasks:** freshness (Fresh / Stale) and produce type (apple, banana, bitter gourd, capsicum, orange, tomato).
- **Heuristic fields:** quality grade and shelf-life days are derived from P(fresh). They are not learned or validated, and the API marks them `*_is_heuristic: true`.
- **Evaluation:** a leakage-free split grouped by source photo. The Kaggle dataset ships many augmented copies of each photo, and a per-file split leaks them into test. See `DECISIONS.md` §0 and `paper/`.

## Architecture

- Backbone: EfficientNet-B0 (ImageNet-pretrained, via timm), with MobileNetV3-Large as an alternative
- Heads: freshness (2 classes) and produce type (6 classes), trained with an equal-weight cross-entropy loss
- OOD gate: energy score on the produce-type head, thresholded at 95% validation TPR (stored in `model_meta.json`)
- Training: PyTorch Lightning, mixed precision, warmup + cosine LR, early stopping, CSV logs per run
- API: FastAPI with rate limiting, API key auth, input validation and SQLite logging
- Frontends: Streamlit (with Grad-CAM) and a Flutter mobile app (`mobile_app/`)

## Project Structure

```
src/
├── data/build_splits.py   # grouped leakage-free split -> data/metadata_v2.json
├── data/dataset.py        # FruitDataset + Albumentations transforms
├── models/                # FreshTrackModel (PyTorch Lightning)
├── training/train.py      # one run -> models/runs/<name>/
├── training/evaluate.py   # metrics.json + model_meta.json per run
├── training/run_experiment.py  # full matrix + baseline -> results/
├── api/                   # FastAPI inference server
├── app.py                 # Streamlit frontend
└── config.py              # labels, heuristics, paths
paper/                     # IEEE paper; numbers are generated from results/
tests/                     # pytest suite
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # set API_KEY, CORS_ORIGINS, ...
```

## Reproduce the results

1. Download the Kaggle datasets "Fresh and Stale Images of Fruits and Vegetables" and "Fruit and Vegetable Image Recognition" into `data/downloads/`.
2. Build the splits, run the experiments and generate the paper's numbers:

```bash
python -m src.data.build_splits          # grouped split + external evaluation set
python -m src.training.run_experiment    # 5 configs x 3 seeds + HSV baseline (resumable)
python paper/make_tables.py              # LaTeX macros/tables/figure from results/
```

`results/summary.md` holds the headline table. Every run records the metadata SHA-256, git SHA and seed in `models/runs/<run>/run_config.json`.

## Serve a model

```bash
cp models/runs/mnv3_mtl_s1/best.ckpt      models/checkpoints/freshtrack_v2.ckpt
cp models/runs/mnv3_mtl_s1/model_meta.json models/checkpoints/model_meta.json
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
streamlit run src/app.py
```

Docker: the image does not contain the model, so mount it at runtime.

```bash
docker build -t freshtrack-api .
docker run -p 8000:8000 --env-file .env \
  -v "$PWD/models/checkpoints:/app/models/checkpoints:ro" freshtrack-api
```

## Mobile App (Flutter, `mobile_app/`)

Android package `in.rvitm.freshtrack`. It runs the model on the phone with ONNX Runtime, so it needs no server and no internet (the release build does not request the INTERNET permission). Scan history, with copies of the images, is kept in a local SQLite database.

```bash
python -m src.training.export_onnx      # model + metadata + parity fixture -> mobile_app/
cd mobile_app
flutter pub get
flutter analyze && flutter test         # 34 tests
flutter build apk --release --split-per-abi   # arm64 APK ~52 MB
```

- **Measurements** (APK size, RAM, cold start, latency, parity with PyTorch): see `docs/mobile_on_device_report.md` and `results/mobile_metrics.json`.
- **Release signing**: put `storeFile`, `storePassword`, `keyAlias` and `keyPassword` in `android/key.properties`. Without that file, release builds fall back to the debug key. `key.properties` and `*.jks` are gitignored.
- **Screenshots**: `mobile_app/screenshots/ondevice_*.png`, taken with networking switched off.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Status check |
| GET | `/health` | Model + device health |
| POST | `/predict` | Image → freshness, produce type, heuristic quality/shelf-life, OOD score |
| POST | `/feedback` | Submit a corrected freshness label (`Fresh`/`Stale`) |
| GET | `/history`, `/stats` | Logged predictions |

`/predict` returns `{"error": "OBJECT_NOT_RECOGNIZED", ...}` with status 200 when the OOD gate rejects the image.

Authentication: pass the `X-API-Key` header when `API_KEY` is set. Rate limits: `/predict` 30/min, `/feedback` 10/min.

## Tests

```bash
pytest tests/ -v
```

## Known Limitations

- Freshness is binary (Fresh/Stale). The data has no intermediate ripeness stages.
- Quality grade and shelf life are heuristics, not predictions: no dataset used here has graded or time-to-spoilage labels.
- Training images are mostly single items on plain backgrounds. Accuracy on cluttered market photos is untested.
- Only 6 produce types are supported. The OOD gate is meant to reject others, but its near-OOD rejection rate is imperfect (see `results/`).
