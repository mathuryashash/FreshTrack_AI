"""Shared configuration for FreshTrack AI project."""

import os

# Label mappings (single source of truth)
FRESHNESS_LABELS = {0: "Fresh", 1: "Semi-ripe", 2: "Overripe", 3: "Rotten"}
FRESHNESS_TO_IDX = {v: k for k, v in FRESHNESS_LABELS.items()}

QUALITY_LABELS = {0: "High (A)", 1: "Medium (B)", 2: "Low (C)"}
QUALITY_TO_IDX = {"A": 0, "B": 1, "C": 2}

# Image settings
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Model settings
MODEL_CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT", "models/checkpoints/freshtrack_epoch=04_val_loss=0.01-v1.ckpt"
)
NUM_FRESHNESS_CLASSES = 4
NUM_QUALITY_CLASSES = 3

# Loss weights
LOSS_WEIGHTS = {
    "freshness": 0.4,
    "quality": 0.3,
    "shelf_life": 0.25,
    "rotation": 0.05,
}

# Training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_WORKERS = min(4, os.cpu_count() or 1)

# API settings
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
