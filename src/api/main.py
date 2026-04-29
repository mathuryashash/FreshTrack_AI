"""
FreshTrack AI - FastAPI Backend
Enhanced with authentication, rate limiting, comprehensive error handling,
optimized inference, and production-ready features.
"""

import sys
import os
import io
import hmac
import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional, Dict, Any, List
from uuid import UUID
from threading import Lock

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Request,
    Depends,
    Query,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from PIL import Image

try:
    import magic
except ImportError:
    magic = None

Image.MAX_IMAGE_PIXELS = 50_000_000
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import uvicorn

sys.path.append(os.getcwd())

from src.models.freshtrack_model import FreshTrackModel
from src.api.database import (
    init_db,
    log_prediction,
    log_feedback,
    get_recent_predictions,
    get_stats,
    get_uncertain_predictions,
)
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
    CORS_ORIGINS,
)

# OOD Detection Configuration
OOD_ENTROPY_THRESHOLD = 1.5  # Bits - above this indicates uncertain/OOD
OOD_CONFIDENCE_THRESHOLD = 0.3  # Below this indicates uncertain/OOD

# Magic bytes for image validation (if python-magic not available, use Pillow)
IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # Note: also matches WAV, need additional check
}

# ── Logging Configuration ───────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s [%(request_id)s]: %(message)s",
)
logger = logging.getLogger("freshtrack")


# Add request_id filter to log records
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "no-request-id"
        return True


logger.addFilter(RequestIdFilter())


# ── Rate limiter ───────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ── API Key auth (optional — set API_KEY env var to enable) ──────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Depends(api_key_header)):
    """Verify API key with constant-time comparison."""
    if not API_KEY:
        return  # auth disabled when no key configured
    if not hmac.compare_digest(key or "", API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ── Model state ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
model_lock = Lock()  # Thread lock for concurrent inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup with optimizations, clean up on shutdown."""
    global model
    import asyncio

    init_db()

    logger.info(f"Loading model from {MODEL_CHECKPOINT} on {device}...")
    try:
        import pathlib as _pathlib

        ckpt_path = _pathlib.Path(MODEL_CHECKPOINT).resolve()
        # Allow models from either 'models/checkpoints' or root 'checkpoints'
        allowed_dirs = [
            _pathlib.Path("models/checkpoints").resolve(),
            _pathlib.Path("checkpoints").resolve(),
        ]
        if not any(str(ckpt_path).startswith(str(d)) for d in allowed_dirs):
            raise RuntimeError(
                f"MODEL_CHECKPOINT outside allowed directories: {ckpt_path}"
            )

        model = FreshTrackModel.load_from_checkpoint(MODEL_CHECKPOINT)
        model.to(device)
        model.eval()

        # Optimize model for inference
        if device.type == "cuda":
            # Enable cuDNN benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA benchmark mode enabled")

        # Try to use torch.compile if available (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile not available: {e}")

        logger.info("Model loaded successfully and optimized for inference.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    model = None
    logger.info("Model unloaded.")


# ── Custom Exception Handlers ─────────────────────────────────────────────
class FreshTrackException(Exception):
    """Base exception for FreshTrack API."""

    def __init__(self, status_code: int, detail: str, error_code: str = None):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code


async def freshtrack_exception_handler(request: Request, exc: FreshTrackException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code or "INTERNAL_ERROR",
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def validation_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# ── App ────────────────────────────────────────────────────────────────────
_debug = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
_app_description = """
## FreshTrack AI API

Intelligent fruit quality assessment system using multi-task deep learning.

### Features
- **Freshness Detection**: Classify fruit as Fresh, Semi-ripe, Overripe, or Rotten
- **Quality Grading**: Rate quality as High (A), Medium (B), or Low (C)
- **Shelf-life Prediction**: Estimate remaining shelf life in days
- **Out-of-Distribution Detection**: Identifies non-fruit images

### Authentication
Set `API_KEY` environment variable to enable API key authentication.
Include the key in `X-API-Key` header.

### Rate Limiting
- General: 60 requests/minute
- Predict: 30 requests/minute
- Health/Stats/History: 10 requests/minute
"""

app = FastAPI(
    title="FreshTrack AI API",
    description=_app_description,
    version="1.1.0",
    docs_url="/docs" if _debug else None,
    redoc_url="/redoc" if _debug else None,
    openapi_tags=[
        {"name": "predictions", "description": "Fruit quality prediction endpoints"},
        {"name": "feedback", "description": "User feedback collection"},
        {"name": "monitoring", "description": "Health and metrics endpoints"},
    ],
    lifespan=lifespan,
)

# Exception handlers
app.add_exception_handler(FreshTrackException, freshtrack_exception_handler)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Trusted hosts (prevents Host header injection)
if TRUSTED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)

# CORS — configurable via environment for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)


# ── Request middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request for tracing."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    # Add request_id to logger context
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        return record

    logging.setLogRecordFactory(record_factory)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.1f}ms"
    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    if not _debug:  # Only in production
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains"
        )
        response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ── Transforms (built once) ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_transforms():
    return A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ToTensorV2(),
        ]
    )


# ── Pydantic schemas ──────────────────────────────────────────────────────
class FeedbackPayload(BaseModel):
    """Feedback payload for corrected predictions."""

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


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    freshness: str
    freshness_confidence: float
    quality: str
    shelf_life_days: float
    entropy_score: float
    prediction_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    timestamp: str


class PaginationParams:
    """Common pagination parameters."""

    def __init__(
        self,
        page: int = Query(default=1, ge=1, description="Page number"),
        limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    ):
        self.page = page
        self.limit = limit
        self.offset = (page - 1) * limit


# ── Helper Functions ──────────────────────────────────────────────────────
def _compute_entropy(probs):
    """Compute Shannon entropy in bits for OOD detection."""
    eps = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    return (entropy / torch.log(torch.tensor(2.0, device=probs.device))).item()


def _is_ood(entropy, max_confidence):
    """Determine if input is out-of-distribution (not a fruit)."""
    return entropy > OOD_ENTROPY_THRESHOLD or max_confidence < OOD_CONFIDENCE_THRESHOLD


def _validate_image_bytes(contents: bytes) -> Image.Image:
    """
    Validate image using multiple methods:
    1. Magic bytes detection (if python-magic available)
    2. Pillow verification and loading
    """
    # Check magic bytes if python-magic is available
    if magic:
        try:
            mime_type = magic.from_buffer(contents, mime=True)
            if not mime_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file type detected: {mime_type}",
                )
        except ImportError:
            pass  # Fall back to Pillow check

    # Pillow verification
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # raises on corrupt/invalid files
        # Re-open for actual use after verify()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        logger.warning(f"Image validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupt image file",
        )


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/", tags=["monitoring"])
def read_root():
    """API root endpoint."""
    return {
        "message": "FreshTrack AI API is running",
        "version": "1.1.0",
        "docs": "/docs" if _debug else "disabled",
    }


@app.get("/health", tags=["monitoring"])
@limiter.limit("10/minute")
async def health_check(request: Request):
    """Health check for container orchestration and load balancers."""
    import datetime

    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        device=str(device),
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )


@app.get("/metrics", tags=["monitoring"])
@limiter.limit("10/minute")
async def metrics(request: Request):
    """
    Prometheus-style metrics endpoint for monitoring.
    Returns API performance and model statistics.
    """
    import psutil
    import datetime

    # Model metrics
    model_loaded = model is not None
    model_device = str(device) if model_loaded else "N/A"

    # System metrics
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
    except ImportError:
        cpu_percent = 0.0
        memory = None
        disk = None

    # Database stats
    db_stats = get_stats()

    metrics_text = f"""# HELP freshtrack_model_loaded Model loaded status (0 or 1)
# TYPE freshtrack_model_loaded gauge
freshtrack_model_loaded{{{model_device}}} {1 if model_loaded else 0}

# HELP freshtrack_total_predictions Total number of predictions made
# TYPE freshtrack_total_predictions counter
freshtrack_total_predictions {db_stats.get("total_predictions", 0)}

# HELP freshtrack_avg_confidence Average prediction confidence
# TYPE freshtrack_avg_confidence gauge
freshtrack_avg_confidence {db_stats.get("avg_confidence", 0.0)}

# HELP freshtrack_cpu_percent CPU usage percentage
# TYPE freshtrack_cpu_percent gauge
freshtrack_cpu_percent {cpu_percent}

# HELP freshtrack_memory_percent Memory usage percentage
# TYPE freshtrack_memory_percent gauge
freshtrack_memory_percent {memory.percent if memory else 0}

# HELP freshtrack_system_timestamp Current server timestamp
# TYPE freshtrack_system_timestamp gauge
freshtrack_system_timestamp {int(time.time())}
"""

    return JSONResponse(
        content={
            "metrics": metrics_text,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "model": {
                "loaded": model_loaded,
                "device": model_device,
            },
            "system": {
                "cpu_percent": cpu_percent if "cpu_percent" in locals() else None,
                "memory_percent": memory.percent if memory else None,
            },
            "database": db_stats,
        },
        media_type="application/json",
    )


@app.post("/predict", tags=["predictions"], response_model=PredictionResponse)
@limiter.limit("30/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Predict fruit freshness, quality, and shelf-life from an image.

    - **file**: Image file (JPEG, PNG, or WebP format, max 10MB)

    Returns freshness classification, quality grade, shelf-life estimate, and OOD detection.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # ── Input validation first (before model check) ──────────────────────
    # Validate content-type header
    content_type = (file.content_type or "").lower()
    allowed_mime = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}

    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext == ".jpeg":
        ext = ".jpg"

    if content_type not in allowed_mime and ext not in ALLOWED_EXTENSIONS:
        logger.warning(
            f"[{request_id}] Unsupported content type: {content_type} or extension: {ext}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type: {content_type} or extension: {ext}",
        )

    # Read with size cap
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    contents = await file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB} MB.",
        )

    # Validate it's actually an image (magic bytes via Pillow/magic)
    image = _validate_image_bytes(contents)

    # ── Model availability check (after validation) ──────────────────────
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    # Inference with thread lock for concurrent requests
    transforms = get_transforms()
    image_np = np.array(image)
    aug = transforms(image=image_np)
    tensor = aug["image"].unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with model_lock:
        with torch.no_grad():
            fresh_logits, qual_logits, shelf_pred, _ = model(tensor)
    inference_ms = (time.perf_counter() - t0) * 1000

    fresh_probs = torch.softmax(fresh_logits, dim=1)
    fresh_idx = int(torch.argmax(fresh_probs, dim=1).item())
    fresh_conf = float(fresh_probs[0][fresh_idx])

    qual_probs = torch.softmax(qual_logits, dim=1)
    qual_idx = int(torch.argmax(qual_probs, dim=1).item())

    # Compute entropy for OOD detection
    entropy = _compute_entropy(fresh_probs)
    max_confidence = fresh_conf

    logger.info(
        f"[{request_id}] Prediction: freshness=%s (%.2f), quality=%s, shelf_life=%.1f, entropy=%.2f",
        FRESHNESS_LABELS.get(fresh_idx, "Unknown"),
        fresh_conf,
        QUALITY_LABELS.get(qual_idx, "Unknown"),
        shelf_pred.item(),
        entropy,
    )

    # Check for OOD (Object Not Recognized)
    is_ood = _is_ood(entropy, max_confidence)

    if is_ood:
        logger.info(
            f"[{request_id}] OOD detected: entropy=%.2f, max_conf=%.2f",
            entropy,
            max_confidence,
        )
        result = {
            "error": "OBJECT_NOT_RECOGNIZED",
            "message": "The input image does not appear to contain a recognizable fruit.",
            "details": {
                "entropy_score": round(entropy, 4),
                "max_confidence": round(max_confidence, 4),
            },
        }
        # Still log to database for active learning
        try:
            pred_id = log_prediction(
                freshness="Unknown",
                freshness_conf=max_confidence,
                quality="Unknown",
                shelf_life_days=0.0,
                inference_ms=inference_ms,
                entropy_score=entropy,
            )
            result["prediction_id"] = pred_id
        except Exception as db_err:
            logger.warning(f"[{request_id}] DB log failed: %s", db_err)
        return result

    result = {
        "freshness": FRESHNESS_LABELS.get(fresh_idx, "Unknown"),
        "freshness_confidence": round(fresh_conf, 4),
        "quality": QUALITY_LABELS.get(qual_idx, "Unknown"),
        "shelf_life_days": round(float(shelf_pred.item()), 1),
        "entropy_score": round(entropy, 4),
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
        logger.warning(f"[{request_id}] DB log failed: %s", db_err)

    return result


@app.post("/feedback", tags=["feedback"])
@limiter.limit("10/minute")
async def submit_feedback(request: Request, payload: FeedbackPayload):
    """
    Submit corrected predictions for model improvement.

    Provide the prediction ID and the correct freshness label to help improve the model.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    try:
        log_feedback(
            prediction_id=payload.image_id,
            predicted_freshness=payload.predicted_freshness,
            correct_freshness=payload.correct_freshness,
            notes=payload.notes,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as db_err:
        logger.error(f"[{request_id}] Feedback DB log failed: %s", db_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record feedback",
        )
    logger.info(
        f"[{request_id}] Feedback received for prediction_id=%s", payload.image_id
    )
    return {"status": "success", "message": "Feedback recorded. Thank you!"}


@app.get("/history", tags=["predictions"])
@limiter.limit("30/minute")
async def prediction_history(
    request: Request,
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    freshness: Optional[str] = Query(
        default=None, description="Filter by freshness label"
    ),
):
    """
    Return paginated prediction history from the database.

    - **page**: Page number (starts at 1)
    - **limit**: Number of items per page (max 100)
    - **freshness**: Optional filter by freshness label
    """
    offset = (page - 1) * limit
    predictions = get_recent_predictions(
        limit, offset, freshness_filter=freshness or ""
    )
    total = get_stats()["total_predictions"]

    return {
        "predictions": predictions,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit,
        },
    }


@app.get("/stats", tags=["monitoring"])
@limiter.limit("30/minute")
async def prediction_stats(request: Request):
    """Get aggregated prediction statistics."""
    return get_stats()


@app.get("/uncertain-predictions", tags=["predictions"])
@limiter.limit("10/minute")
async def uncertain_predictions(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    threshold: float = Query(default=1.5, ge=0.0, le=5.0),
):
    """
    Get predictions with high entropy (uncertain predictions).
    Useful for active learning and identifying edge cases.
    """
    return {"predictions": get_uncertain_predictions(limit, threshold)}


# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", 8000)),
        reload=_debug,
        workers=int(os.environ.get("API_WORKERS", "1")),
    )
