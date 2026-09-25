"""Shared configuration for FreshTrack AI project."""

import os

# ── Label mappings (single source of truth) ──────────────────────────────────
# Learned tasks: binary freshness and produce type, both annotated in the
# source dataset (folder names). See src/data/build_splits.py.
FRESHNESS_LABELS = {0: "Fresh", 1: "Stale"}
FRESHNESS_TO_IDX = {v: k for k, v in FRESHNESS_LABELS.items()}

PRODUCE_TYPES = ["apple", "banana", "bitter_gourd", "capsicum", "orange", "tomato"]
PRODUCE_TO_IDX = {t: i for i, t in enumerate(PRODUCE_TYPES)}

# Quality grade is NOT learned: no dataset provides graded labels. The API
# derives it from P(fresh) and marks it as heuristic.
QUALITY_LABELS = {0: "High (A)", 1: "Medium (B)", 2: "Low (C)"}
QUALITY_FROM_P_FRESH = [(0.85, 0), (0.50, 1), (0.0, 2)]  # (min P(fresh), grade idx)

# ── Image settings ──────────────────────────────────────────────────────
IMAGE_SIZE = 224
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# ── Model settings ──────────────────────────────────────────────────────
# Production model: produced by src/training/run_experiment.py and promoted by
# copying the chosen run's checkpoint + model_meta.json here.
MODEL_CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT", "models/checkpoints/freshtrack_v2.ckpt"
)
# Labels, preprocessing and OOD threshold travel with the checkpoint.
MODEL_META = os.environ.get("MODEL_META", "models/checkpoints/model_meta.json")
NUM_FRESHNESS_CLASSES = len(FRESHNESS_LABELS)
NUM_PRODUCE_TYPES = len(PRODUCE_TYPES)

# ── Loss weights ──────────────────────────────────────────────────────
LOSS_WEIGHTS = {"freshness": 0.5, "produce_type": 0.5}

# ── Training defaults ─────────────────────────────────────────────────────
DEFAULT_BACKBONE = "efficientnet_b0"
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_WARMUP_EPOCHS = 1
DEFAULT_NUM_WORKERS = min(4, os.cpu_count() or 1)

# ── API settings ──────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Optional API key auth — set API_KEY env var to enable, leave blank to disable
API_KEY = os.environ.get("API_KEY", "")

# Trusted hosts for TrustedHostMiddleware — comma-separated, blank = disabled
_trusted_hosts_raw = os.environ.get("TRUSTED_HOSTS", "")
TRUSTED_HOSTS = [h.strip() for h in _trusted_hosts_raw.split(",") if h.strip()]

# CORS origins for production — comma-separated, blank = allow all (not recommended for prod)
_cors_origins_raw = os.environ.get("CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()] or None

# ── External API keys ─────────────────────────────────────────────────────
OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", "")
KAGGLE_API_KEY = os.environ.get("KAGGLE_API_KEY", "")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")

# ── Shelf-life heuristic (days) ─────────────────────────────────────────────
# NOT learned and NOT validated: no dataset here has measured shelf-life.
# Rough room-temperature reference days for a fully fresh item; the API
# reports reference_days * P(fresh) and flags the value as heuristic.
SHELF_LIFE_REFERENCE_DAYS = {
    "apple": 10.0,
    "banana": 4.0,
    "bitter_gourd": 3.0,
    "capsicum": 4.0,
    "orange": 10.0,
    "tomato": 5.0,
}


def derive_quality(p_fresh: float) -> str:
    """Heuristic grade from P(fresh); not a learned quality grade."""
    for min_p, idx in QUALITY_FROM_P_FRESH:
        if p_fresh >= min_p:
            return QUALITY_LABELS[idx]
    return QUALITY_LABELS[QUALITY_FROM_P_FRESH[-1][1]]


def derive_shelf_life(produce_type: str, p_fresh: float) -> float:
    """Heuristic days: reference room-temperature days x P(fresh). Not validated."""
    return round(SHELF_LIFE_REFERENCE_DAYS[produce_type] * p_fresh, 1)
