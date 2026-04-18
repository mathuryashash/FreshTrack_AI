import sys
import os
import io
import hmac
import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional
from uuid import UUID

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from PIL import Image
Image.MAX_IMAGE_PIXELS = 50_000_000
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import uvicorn

sys.path.append(os.getcwd())

from src.models.freshtrack_model import FreshTrackModel
from src.api.database import init_db, log_prediction, log_feedback, get_recent_predictions, get_stats
from src.config import (
    MODEL_CHECKPOINT,
    FRESHNESS_LABELS,
    QUALITY_LABELS,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    IMAGE_SIZE,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_EXTENSIONS,
    API_KEY,
    TRUSTED_HOSTS,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("freshtrack")

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ── API Key auth (optional — set API_KEY env var to enable) ──────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Depends(api_key_header)):
    if not API_KEY:
        return  # auth disabled when no key configured
    if not hmac.compare_digest(key or "", API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Model state ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global model
    init_db()
    logger.info(f"Loading model from {MODEL_CHECKPOINT} on {device}...")
    try:
        import pathlib as _pathlib
        ckpt_path = _pathlib.Path(MODEL_CHECKPOINT).resolve()
        allowed_dir = _pathlib.Path("models/checkpoints").resolve()
        if not str(ckpt_path).startswith(str(allowed_dir)):
            raise RuntimeError(f"MODEL_CHECKPOINT outside allowed directory: {ckpt_path}")
        model = FreshTrackModel.load_from_checkpoint(MODEL_CHECKPOINT)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    model = None
    logger.info("Model unloaded.")


# ── App ───────────────────────────────────────────────────────────────────────
_debug = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
app = FastAPI(
    title="FreshTrack API",
    description="Fruit Freshness Detection, Quality Grading & Shelf-Life Prediction",
    version="1.0.0",
    docs_url="/docs" if _debug else None,
    redoc_url="/redoc" if _debug else None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Trusted hosts (prevents Host header injection)
if TRUSTED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)

# CORS — restrict to explicit origins only
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:8501,http://localhost:3000"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,          # credentials=True only if cookies are needed
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.perf_counter() - start) * 1000:.1f}ms"
    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ── Transforms (built once) ───────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_transforms():
    return A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ToTensorV2(),
        ]
    )


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class FeedbackPayload(BaseModel):
    image_id: str
    predicted_freshness: str
    correct_freshness: str
    notes: str = ""

    @field_validator("correct_freshness")
    @classmethod
    def validate_freshness(cls, v):
        valid = set(FRESHNESS_LABELS.values())
        if v not in valid:
            raise ValueError(f"correct_freshness must be one of {valid}")
        return v

    @field_validator("predicted_freshness")
    @classmethod
    def validate_predicted_freshness(cls, v):
        valid = set(FRESHNESS_LABELS.values())
        if v not in valid:
            raise ValueError(f"predicted_freshness must be one of {valid}")
        return v

    @field_validator("image_id")
    @classmethod
    def validate_uuid_format(cls, v):
        try:
            UUID(v)
        except ValueError:
            raise ValueError("image_id must be a valid UUID")
        return v

    @field_validator("image_id", "predicted_freshness", "correct_freshness")
    @classmethod
    def no_empty_strings(cls, v):
        if not v.strip():
            raise ValueError("Field must not be empty")
        if len(v) > 500:
            raise ValueError("Field exceeds maximum length of 500 characters")
        return v


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "FreshTrack API is running"}


@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request):
    """Health check for container orchestration."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
    }


@app.post("/predict", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    # ── Input validation first (before model check so tests get correct status codes) ──

    # Validate content-type header
    content_type = (file.content_type or "").lower()
    allowed_mime = {"image/jpeg", "image/png", "image/webp"}
    if content_type not in allowed_mime:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {content_type}",
        )

    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read with size cap
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    contents = await file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB} MB.",
        )

    # Validate it's actually an image (magic bytes via Pillow)
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()                          # raises on corrupt/invalid files
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file")

    # ── Model availability check (after validation) ───────────────────────────
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Inference
    transforms = get_transforms()
    image_np = np.array(image)
    aug = transforms(image=image_np)
    tensor = aug["image"].unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        fresh_logits, qual_logits, shelf_pred, _ = model(tensor)
    inference_ms = (time.perf_counter() - t0) * 1000

    fresh_probs = torch.softmax(fresh_logits, dim=1)
    fresh_idx = int(torch.argmax(fresh_probs, dim=1).item())

    qual_probs = torch.softmax(qual_logits, dim=1)
    qual_idx = int(torch.argmax(qual_probs, dim=1).item())

    logger.info(
        "Prediction: freshness=%s (%.2f), quality=%s, shelf_life=%.1f",
        FRESHNESS_LABELS.get(fresh_idx, "Unknown"),
        float(fresh_probs[0][fresh_idx]),
        QUALITY_LABELS.get(qual_idx, "Unknown"),
        shelf_pred.item(),
    )

    result = {
        "freshness": FRESHNESS_LABELS.get(fresh_idx, "Unknown"),
        "freshness_confidence": round(float(fresh_probs[0][fresh_idx]), 4),
        "quality": QUALITY_LABELS.get(qual_idx, "Unknown"),
        "shelf_life_days": round(float(shelf_pred.item()), 1),
    }

    # Persist to database
    try:
        pred_id = log_prediction(
            freshness=result["freshness"],
            freshness_conf=result["freshness_confidence"],
            quality=result["quality"],
            shelf_life_days=result["shelf_life_days"],
            inference_ms=inference_ms,
        )
        result["prediction_id"] = pred_id
    except Exception as db_err:
        logger.warning("DB log failed: %s", db_err)

    return result


@app.post("/feedback", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def submit_feedback(request: Request, payload: FeedbackPayload):
    """Collect corrected predictions for future retraining."""
    try:
        log_feedback(
            prediction_id=payload.image_id,
            predicted_freshness=payload.predicted_freshness,
            correct_freshness=payload.correct_freshness,
            notes=payload.notes,
        )
    except Exception as db_err:
        logger.error("Feedback DB log failed: %s", db_err)
        raise HTTPException(status_code=500, detail="Failed to record feedback")
    logger.info("Feedback received for prediction_id=%s", payload.image_id)
    return {"status": "success", "message": "Feedback recorded. Thank you!"}


@app.get("/history", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def prediction_history(request: Request, limit: int = Query(default=20, ge=1, le=100)):
    """Return the most recent predictions from the database."""
    return {"predictions": get_recent_predictions(limit)}


@app.get("/stats", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def prediction_stats(request: Request):
    """Aggregate stats: total predictions, confidence avg, freshness distribution."""
    return get_stats()


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", 8000)),
        reload=False,
    )
