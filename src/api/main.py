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
    Response,
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
from PIL import Image, ImageOps

try:
    import magic
except ImportError:
    magic = None

Image.MAX_IMAGE_PIXELS = 50_000_000
import torch
import numpy as np
import uvicorn

sys.path.append(os.getcwd())

from src.models.freshtrack_model import FreshTrackModel, energy_score, entropy_bits
from src.data.dataset import get_val_transforms
from src.api.database import (
    init_db,
    log_prediction,
    log_feedback,
    get_recent_predictions,
    get_stats,
    count_predictions,
    get_uncertain_predictions,
)
from src.config import (
    MODEL_CHECKPOINT,
    MODEL_META,
    FRESHNESS_LABELS,
    PRODUCE_TYPES,
    derive_quality,
    derive_shelf_life,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_EXTENSIONS,
    API_KEY,
    TRUSTED_HOSTS,
    CORS_ORIGINS,
)

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
model_version = "unknown"
checkpoint_hash = "unknown"
model_meta: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global model, model_version, checkpoint_hash, model_meta
    import json as _json

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
        if not any(ckpt_path.is_relative_to(d) for d in allowed_dirs):
            raise RuntimeError(
                f"MODEL_CHECKPOINT outside allowed directories: {ckpt_path}"
            )

        # Compute checkpoint hash for versioning
        with open(MODEL_CHECKPOINT, "rb") as f:
            checkpoint_hash = hashlib.sha256(f.read()).hexdigest()[:12]

        # Labels, preprocessing and the OOD threshold were written next to the
        # checkpoint by src/training/evaluate.py; refuse to serve without them.
        meta = _json.loads(_pathlib.Path(MODEL_META).read_text())
        expected = {
            "produce_types": PRODUCE_TYPES,
            "freshness_labels": [FRESHNESS_LABELS[i] for i in sorted(FRESHNESS_LABELS)],
            "image_size": IMAGE_SIZE,
            "normalize_mean": list(NORMALIZE_MEAN),
            "normalize_std": list(NORMALIZE_STD),
        }
        mismatched = [k for k, v in expected.items() if meta.get(k) != v]
        if mismatched or meta.get("ood_score") != "energy_produce_type":
            raise RuntimeError(f"model_meta.json does not match src/config.py: {mismatched}")

        # weights_only: never unpickle arbitrary objects from a mounted checkpoint
        loaded = FreshTrackModel.load_from_checkpoint(
            MODEL_CHECKPOINT, pretrained=False, weights_only=True, map_location=device
        )
        loaded.to(device).eval()
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Startup gate: a checkpoint that cannot run inference must not be served.
        with torch.no_grad():
            out = loaded(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
        if set(out) != {"freshness", "produce_type"} or not all(
            torch.isfinite(v).all() for v in out.values()
        ):
            raise RuntimeError(f"Dummy inference failed: heads={sorted(out)}")

        model_meta = meta
        model_version = f"v2.0.0-{meta['backbone']}-{checkpoint_hash[:8]}"
        model = loaded
        logger.info(f"Model loaded: version={model_version}, checkpoint_hash={checkpoint_hash}")
    except Exception as e:
        model = None
        logger.error(f"Failed to load model: {e}")
    yield
    model = None
    model_version = "unknown"
    checkpoint_hash = "unknown"
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
- **Freshness Detection** (learned): Fresh or Stale
- **Produce Type** (learned): apple, banana, bitter gourd, capsicum, orange, tomato
- **Quality Grade** (heuristic, derived from P(fresh); not a learned grade)
- **Shelf-life Estimate** (heuristic reference days x P(fresh); not validated)
- **Out-of-Distribution Detection**: energy score on the produce-type head

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
    allow_origins=CORS_ORIGINS or [],  # fail closed; set CORS_ORIGINS for browser clients
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
# Same pipeline as evaluation, imported so train/serve preprocessing cannot drift.
get_transforms = lru_cache(maxsize=1)(get_val_transforms)


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
    """Response model for predictions.

    quality and shelf_life_days are heuristics derived from P(fresh) and the
    produce type; the *_is_heuristic flags make that explicit to clients.
    """

    freshness: str
    freshness_confidence: float
    produce_type: str
    produce_type_confidence: float
    quality: str
    quality_is_heuristic: bool = True
    shelf_life_days: float
    shelf_life_is_heuristic: bool = True
    entropy_score: float
    ood_score: float
    prediction_id: Optional[str] = None
    model_version: Optional[str] = None
    checkpoint_hash: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    timestamp: str
    model_version: Optional[str] = None
    checkpoint_hash: Optional[str] = None


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
def _ood_score(logits: Dict[str, torch.Tensor]) -> float:
    """Score named in model_meta.json; higher = more in-distribution."""
    name = model_meta["ood_score"]
    if name == "energy_produce_type":
        return float(energy_score(logits["produce_type"]).item())
    raise RuntimeError(f"Unsupported OOD score {name!r}")


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
        # Re-open for actual use after verify(); apply EXIF orientation so
        # phone photos match training, where cv2.imread applies it.
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
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
        "model_version": model_version if model is not None else None,
        "checkpoint_hash": checkpoint_hash if model is not None else None,
        "docs": "/docs" if _debug else "disabled",
    }


@app.get("/health", tags=["monitoring"])
@limiter.limit("10/minute")
async def health_check(request: Request, response: Response):
    """Health check for container orchestration and load balancers.

    Returns 503 when the model is not loaded so `curl -f` probes fail.
    """
    import datetime

    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        device=str(device),
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        model_version=model_version if model is not None else None,
        checkpoint_hash=checkpoint_hash if model is not None else None,
    )


@app.get("/metrics", tags=["monitoring"], dependencies=[Depends(verify_api_key)])
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
                "version": model_version if model_loaded else None,
                "checkpoint_hash": checkpoint_hash if model_loaded else None,
            },
            "system": {
                "cpu_percent": cpu_percent if "cpu_percent" in locals() else None,
                "memory_percent": memory.percent if memory else None,
            },
            "database": db_stats,
        },
        media_type="application/json",
    )


@app.post("/predict", tags=["predictions"], response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
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
            logits = {k: v.float().cpu() for k, v in model(tensor).items()}
    inference_ms = (time.perf_counter() - t0) * 1000

    fresh_probs = torch.softmax(logits["freshness"], dim=1)[0]
    fresh_idx = int(fresh_probs.argmax())
    fresh_conf = float(fresh_probs[fresh_idx])
    p_fresh = float(fresh_probs[0])  # index 0 == "Fresh"

    type_probs = torch.softmax(logits["produce_type"], dim=1)[0]
    type_idx = int(type_probs.argmax())
    produce_type = PRODUCE_TYPES[type_idx]

    entropy = float(entropy_bits(logits["freshness"]).item())
    ood_score = _ood_score(logits)

    logger.info(
        f"[{request_id}] Prediction: freshness=%s (%.2f), type=%s (%.2f), ood_score=%.2f",
        FRESHNESS_LABELS[fresh_idx],
        fresh_conf,
        produce_type,
        float(type_probs[type_idx]),
        ood_score,
    )

    # Object Not Recognized: returned as a 200 JSONResponse so it bypasses
    # response_model validation (the client checks the "error" field).
    if ood_score < model_meta["ood_threshold"]:
        logger.info(f"[{request_id}] OOD detected: score=%.2f", ood_score)
        result = {
            "error": "OBJECT_NOT_RECOGNIZED",
            "message": "The image does not appear to show a supported fruit or vegetable.",
            "details": {
                "ood_score": round(ood_score, 4),
                "ood_threshold": round(model_meta["ood_threshold"], 4),
            },
            "model_version": model_version,
            "checkpoint_hash": checkpoint_hash,
        }
        # Still log to database for active learning
        try:
            result["prediction_id"] = log_prediction(
                freshness="Unknown",
                freshness_conf=fresh_conf,
                quality="Unknown",
                shelf_life_days=0.0,
                inference_ms=inference_ms,
                entropy_score=entropy,
                ood_score=ood_score,
                model_version=model_version,
            )
        except Exception as db_err:
            logger.warning(f"[{request_id}] DB log failed: %s", db_err)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    result = {
        "freshness": FRESHNESS_LABELS[fresh_idx],
        "freshness_confidence": round(fresh_conf, 4),
        "produce_type": produce_type,
        "produce_type_confidence": round(float(type_probs[type_idx]), 4),
        "quality": derive_quality(p_fresh),
        "quality_is_heuristic": True,
        "shelf_life_days": derive_shelf_life(produce_type, p_fresh),
        "shelf_life_is_heuristic": True,
        "entropy_score": round(entropy, 4),
        "ood_score": round(ood_score, 4),
        "model_version": model_version,
        "checkpoint_hash": checkpoint_hash,
    }

    # Persist to database
    try:
        pred_id = log_prediction(
            freshness=result["freshness"],
            freshness_conf=result["freshness_confidence"],
            quality=result["quality"],
            shelf_life_days=result["shelf_life_days"],
            inference_ms=inference_ms,
            entropy_score=result["entropy_score"],
            produce_type=result["produce_type"],
            produce_type_conf=result["produce_type_confidence"],
            ood_score=result["ood_score"],
            model_version=model_version,
        )
        result["prediction_id"] = pred_id
    except Exception as db_err:
        logger.warning(f"[{request_id}] DB log failed: %s", db_err)

    return result


@app.post("/feedback", tags=["feedback"], dependencies=[Depends(verify_api_key)])
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


@app.get("/history", tags=["predictions"], dependencies=[Depends(verify_api_key)])
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
    total = count_predictions(freshness or "")

    return {
        "predictions": predictions,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit,
        },
    }


@app.get("/stats", tags=["monitoring"], dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def prediction_stats(request: Request):
    """Get aggregated prediction statistics."""
    return get_stats()


@app.get("/uncertain-predictions", tags=["predictions"], dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def uncertain_predictions(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    threshold: float = Query(default=0.9, ge=0.0, le=1.0),  # bits; binary max = 1
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
