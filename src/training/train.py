"""Train one FreshTrack model and record everything needed to reproduce it.

Example:
    python -m src.training.train --metadata data/metadata_v2.json --name b0_mtl_s0
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from src.config import (
    DEFAULT_BACKBONE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_WORKERS,
)
from src.data.dataset import FruitDataset, get_train_transforms, get_val_transforms
from src.models.freshtrack_model import FreshTrackModel

RUNS_DIR = Path("models/runs")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def train(
    metadata_path,
    run_name,
    backbone=DEFAULT_BACKBONE,
    tasks=("freshness", "produce_type"),
    split_field="split",
    epochs=10,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    seed=0,
    num_workers=DEFAULT_NUM_WORKERS,
):
    """Train one run into models/runs/<run_name>/ and return the best checkpoint path."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(seed, workers=True)

    config = {
        "run_name": run_name,
        "backbone": backbone,
        "tasks": list(tasks),
        "split_field": split_field,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "metadata": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "git_sha": _git_sha(),
        "torch": torch.__version__,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    train_ds = FruitDataset(metadata_path, get_train_transforms(), "train", split_field)
    val_ds = FruitDataset(metadata_path, get_val_transforms(), "val", split_field)
    print(f"[{run_name}] train={len(train_ds)} val={len(val_ds)} tasks={tasks}")

    model = FreshTrackModel(
        backbone=backbone,
        tasks=tasks,
        learning_rate=learning_rate,
        max_epochs=epochs,
    )

    checkpoint = ModelCheckpoint(
        dirpath=run_dir, filename="best", monitor="val_loss", mode="min", save_top_k=1
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ],
        logger=CSVLogger(save_dir=str(run_dir), name="", version=""),
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        enable_progress_bar=False,
    )
    trainer.fit(
        model,
        make_loader(train_ds, batch_size, True, num_workers),
        make_loader(val_ds, batch_size, False, num_workers),
    )
    config["best_checkpoint"] = checkpoint.best_model_path
    config["best_val_loss"] = float(checkpoint.best_model_score)
    config["epochs_trained"] = trainer.current_epoch
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    return checkpoint.best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata_v2.json")
    parser.add_argument("--name", required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--tasks", default="freshness,produce_type")
    parser.add_argument("--split_field", default="split", choices=["split", "split_naive"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    train(
        args.metadata,
        args.name,
        backbone=args.backbone,
        tasks=tuple(args.tasks.split(",")),
        split_field=args.split_field,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )
