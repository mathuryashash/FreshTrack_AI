"""Shared configuration for FreshTrack AI project."""

import os

# ── Label mappings (single source of truth) ───────────────────────────────────
# NOTE: The dataset currently only provides Fresh/Rotten labels.
# Semi-ripe and Overripe are reserved for when granular annotations are available.
FRESHNESS_LABELS = {0: "Fresh", 1: "Semi-ripe", 2: "Overripe", 3: "Rotten"}
FRESHNESS_TO_IDX = {v: k for k, v in FRESHNESS_LABELS.items()}

QUALITY_LABELS = {0: "High (A)", 1: "Medium (B)", 2: "Low (C)"}
QUALITY_TO_IDX = {"A": 0, "B": 1, "C": 2}

# ── Image settings ────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# ── Model settings ────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT", "models/checkpoints/freshtrack_epoch=04_val_loss=0.01-v1.ckpt"
)
NUM_FRESHNESS_CLASSES = 4
NUM_QUALITY_CLASSES = 3

# ── Loss weights ──────────────────────────────────────────────────────────────
LOSS_WEIGHTS = {
    "freshness": 0.4,
    "quality": 0.3,
    "shelf_life": 0.25,
    "rotation": 0.05,
}

# ── Training defaults ─────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_WORKERS = min(4, os.cpu_count() or 1)

# ── API settings ──────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Optional API key auth — set API_KEY env var to enable, leave blank to disable
API_KEY = os.environ.get("API_KEY", "")

# Trusted hosts for TrustedHostMiddleware — comma-separated, blank = disabled
_trusted_hosts_raw = os.environ.get("TRUSTED_HOSTS", "")
TRUSTED_HOSTS = [h.strip() for h in _trusted_hosts_raw.split(",") if h.strip()]

# ── External API keys ─────────────────────────────────────────────────────────
OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", "")
KAGGLE_API_KEY = os.environ.get("KAGGLE_API_KEY", "")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")

# ── Shelf-life heuristics (days) per freshness + fruit type ──────────────────
# Used by prepare_dataset.py when no explicit annotation is available.
# Values are conservative estimates; replace with real annotations when possible.
SHELF_LIFE_HEURISTICS = {
    ("Fresh", "apple"):   7.0,
    ("Fresh", "banana"):  5.0,
    ("Fresh", "orange"):  10.0,
    ("Fresh", "default"): 7.0,
    ("Semi-ripe", "default"): 4.0,
    ("Overripe", "default"):  1.5,
    ("Rotten", "default"):    0.0,
}
