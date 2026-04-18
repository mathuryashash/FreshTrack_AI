# FreshTrack AI

An intelligent fruit quality assessment system using multi-task deep learning to classify freshness, grade quality, and predict shelf life from a single image.

## Architecture

- Backbone: EfficientNet-B0 (pretrained ImageNet)
- Task heads: Freshness (4 classes), Quality (3 classes), Shelf-life (regression), Rotation (auxiliary)
- Training: PyTorch Lightning with mixed precision, W&B logging, early stopping
- API: FastAPI with rate limiting, API key auth, and input validation
- Frontend: Streamlit with Grad-CAM visualisation

## Project Structure

```
freshtrack-ai/
├── src/
│   ├── models/         # FreshTrackModel (PyTorch Lightning)
│   ├── data/           # FruitDataset + Albumentations transforms
│   ├── training/       # train.py, train_sequential.py
│   ├── api/            # FastAPI inference server
│   ├── utils/          # GradCAM, quantization, data_setup
│   ├── app.py          # Streamlit frontend
│   └── config.py       # Centralised configuration
├── scripts/            # Data prep, splits, validation utilities
├── tests/              # Unit and integration tests
├── data/               # Metadata JSON files (generated)
├── models/checkpoints/ # Saved model checkpoints
├── Dockerfile
├── requirements.txt
└── requirements-api.txt
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set API_KEY, MODEL_CHECKPOINT, Kaggle credentials, etc.
```

### 3. Prepare dataset metadata

Place your datasets under the project root (`FruitNet_Indian/`, `Fruit_Quality_Classification/`, `Fruits_360/`), then run:

```bash
python scripts/prepare_dataset.py
```

This creates `data/metadata_fruitnet.json`, `data/metadata_fruitquality.json`, and `data/metadata_fruits360.json`.

### 4. Multi-stage training

```bash
python scripts/train_sequential.py
```

Or train a single stage directly:

```bash
python src/training/train.py --metadata data/metadata_fruitnet.json --epochs 10 --name stage1
```

### 5. Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 6. Run the Streamlit frontend

```bash
streamlit run src/app.py
```

### 7. Docker

```bash
docker build -t freshtrack-api .
docker run -p 8000:8000 --env-file .env freshtrack-api
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Status check |
| GET | `/health` | Model + device health |
| POST | `/predict` | Upload image → freshness, quality, shelf-life |
| POST | `/feedback` | Submit corrected prediction |
| GET | `/docs` | Swagger UI |

Authentication: pass `X-API-Key: <your_key>` header when `API_KEY` is set in `.env`.

Rate limits: `/predict` — 30 req/min; `/feedback` — 10 req/min.

## Running Tests

```bash
pytest tests/ -v
```

## Known Limitations

- Training data only contains `Fresh` and `Rotten` labels. `Semi-ripe` and `Overripe` classes are reserved for when granular annotations are available.
- Quality grades (`A`/`B`/`C`) are currently derived from freshness labels, not independently annotated.
- Shelf-life predictions use per-fruit heuristics; replace with real annotations for production accuracy.
