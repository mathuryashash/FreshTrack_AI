"""
prepare_dataset.py
Scans dataset directories and produces per-source metadata JSON files.

Freshness labels: Fresh | Rotten  (Semi-ripe / Overripe reserved for future data)
Quality labels  : A (Fresh) | C (Rotten) | B (default fallback)
Shelf-life      : Per-fruit heuristics from config — NOT flat 7/0 days.
"""

import os
import json
import hashlib
from pathlib import Path

import sys
sys.path.append(os.getcwd())

from src.config import SHELF_LIFE_HEURISTICS

# ── Folder-name → freshness label ─────────────────────────────────────────────
FRESHNESS_MAPPING = {
    "freshapples":          "Fresh",
    "freshbanana":          "Fresh",
    "freshoranges":         "Fresh",
    "rottenapples":         "Rotten",
    "rottenbanana":         "Rotten",
    "rottenoranges":        "Rotten",
    "fresh":                "Fresh",
    "rotten":               "Rotten",
    "good quality_fruits":  "Fresh",
    "bad quality_fruits":   "Rotten",
}

# Folder-name → fruit type (for shelf-life lookup)
FRUIT_TYPE_MAPPING = {
    "apple":   "apple",
    "banana":  "banana",
    "orange":  "orange",
}

QUALITY_MAPPING = {"Fresh": "A", "Rotten": "C"}


def _infer_fruit_type(path_parts: tuple) -> str:
    """Return fruit type string from path components, or 'default'."""
    joined = " ".join(p.lower() for p in path_parts)
    for key, fruit in FRUIT_TYPE_MAPPING.items():
        if key in joined:
            return fruit
    return "default"


def _shelf_life(freshness: str, fruit_type: str) -> float:
    return SHELF_LIFE_HEURISTICS.get(
        (freshness, fruit_type),
        SHELF_LIFE_HEURISTICS.get((freshness, "default"), 3.0),
    )


def _deterministic_split(filename: str, val_frac: float = 0.2, test_frac: float = 0.2) -> str:
    """Assign a deterministic split based on filename hash (stable across runs)."""
    h = int(hashlib.md5(filename.encode()).hexdigest(), 16) % 10
    test_cutoff = int(test_frac * 10)
    val_cutoff = test_cutoff + int(val_frac * 10)
    if h < test_cutoff:
        return "test"
    if h < val_cutoff:
        return "val"
    return "train"


def log(msg: str, log_path: str = "scripts/prepare_log.txt"):
    safe_msg = msg.replace('\n', '\\n').replace('\r', '\\r')
    with open(log_path, "a") as f:
        f.write(safe_msg + "\n")
    print(safe_msg)


def create_metadata():
    """
    Produces:
      data/metadata_fruitnet.json
      data/metadata_fruitquality.json
      data/metadata_fruits360.json
    """
    log_path = "scripts/prepare_log.txt"
    with open(log_path, "w") as f:
        f.write("Starting prepare_dataset.py\n")

    datasets_config = [
        (Path("FruitNet_Indian"),              "quality_folders", "metadata_fruitnet.json"),
        (Path("Fruit_Quality_Classification"), "quality_folders", "metadata_fruitquality.json"),
        (Path("Fruits_360"),                   "fruits_360",      "metadata_fruits360.json"),
    ]

    os.makedirs("data", exist_ok=True)

    for base_path, dataset_type, output_filename in datasets_config:
        if not base_path.exists():
            log(f"Skipping {base_path} (not found)", log_path)
            continue

        log(f"Processing {base_path} → {output_filename}...", log_path)
        metadata = []
        count = 0

        for root, _dirs, files in os.walk(base_path):
            for file in files:
                if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                full_path = Path(root) / file
                parts = full_path.parts
                freshness = None
                split = "train"

                if dataset_type == "quality_folders":
                    if any("Good Quality" in p for p in parts):
                        freshness = "Fresh"
                    elif any("Bad Quality" in p for p in parts):
                        freshness = "Rotten"
                    split = _deterministic_split(file)

                elif dataset_type == "fruits_360":
                    split = "test" if ("Test" in parts or "test" in parts) else _deterministic_split(file)
                    class_name = full_path.parent.name.lower()
                    freshness = "Rotten" if "rotten" in class_name else "Fresh"

                if freshness is None:
                    continue

                fruit_type = _infer_fruit_type(parts)
                quality = QUALITY_MAPPING.get(freshness, "B")
                shelf_life = _shelf_life(freshness, fruit_type)

                abs_path = full_path.absolute()
                try:
                    rel_path = abs_path.relative_to(Path.cwd())
                except ValueError:
                    rel_path = abs_path

                metadata.append({
                    "image_path":     str(rel_path),
                    "freshness":      freshness,
                    "quality":        quality,
                    "shelf_life_days": shelf_life,
                    "split":          split,
                    "source":         str(base_path),
                    "fruit_type":     fruit_type,
                })
                count += 1
                if count % 1000 == 0:
                    log(f"  Processed {count} images...", log_path)

        output_file = Path("data") / output_filename
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)

        log(f"Created {output_filename} with {len(metadata)} images", log_path)


if __name__ == "__main__":
    create_metadata()
