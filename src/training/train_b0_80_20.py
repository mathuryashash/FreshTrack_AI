import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torch
import os
import sys
import json
import random
from pathlib import Path

sys.path.append(os.getcwd())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_SILENT"] = "true"

from src.models.freshtrack_model import FreshTrackModel
from src.data.dataset import FruitDataset, get_train_transforms, get_val_transforms
from src.config import DEFAULT_LEARNING_RATE


def create_80_20_split(base_metadata_path, output_path):
    with open(base_metadata_path) as f:
        data = json.load(f)

    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.8)

    for item in data[:split_idx]:
        item["split"] = "train"
    for item in data[split_idx:]:
        item["split"] = "test"

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    train_count = sum(1 for x in data if x["split"] == "train")
    test_count = sum(1 for x in data if x["split"] == "test")
    print(f"Created: {train_count} train, {test_count} test")
    return train_count, test_count


def train(metadata_path, epochs, run_name, batch_size, checkpoint_dir):
    print(f"\nStarting training with metadata: {metadata_path}")
    print(f"Epochs: {epochs}")
    print(f"Run Name: {run_name}")
    print(f"Batch Size: {batch_size}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        project="freshtrack-ai",
        name=run_name,
        config={
            "architecture": "EfficientNet-B0",
            "batch_size": batch_size,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "epochs": epochs,
            "metadata": metadata_path,
        },
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{run_name}_{{epoch:02d}}_{{val_loss:.2f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    train_dataset = FruitDataset(
        metadata_path, transform=get_train_transforms(), split="train"
    )

    test_dataset = FruitDataset(
        metadata_path, transform=get_val_transforms(), split="test"
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    pin_memory = accelerator == "gpu"

    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    model = FreshTrackModel(learning_rate=DEFAULT_LEARNING_RATE)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "gpu" else "32",
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=2,
    )

    trainer.fit(model, train_loader, val_loader)

    results = {
        "model_name": "EfficientNet-B0",
        "split": "80/20",
        "epochs_trained": epochs,
        "epoch_metrics": [],
        "final_test_acc": None,
        "final_val_acc": None,
        "checkpoint_path": None,
    }

    if trainer.checkpoint_callback.best_model_path:
        results["checkpoint_path"] = trainer.checkpoint_callback.best_model_path

        final_metrics = trainer.callback_metrics
        results["final_val_acc"] = float(final_metrics.get("val_acc_fresh", 0))
        results["final_test_acc"] = float(final_metrics.get("test_acc_fresh", 0))

    for epoch in range(epochs):
        epoch_key = f"epoch_{epoch}"
        if epoch_key in trainer.callback_metrics:
            results["epoch_metrics"].append(
                {
                    "epoch": epoch,
                    "train_acc": float(
                        trainer.callback_metrics.get(f"train_acc_fresh", 0)
                    ),
                    "val_acc": float(trainer.callback_metrics.get("val_acc_fresh", 0)),
                }
            )

    return results, checkpoint_callback.best_model_path


if __name__ == "__main__":
    METADATA_OUT = "data/metadata_b0_80_20.json"
    CHECKPOINT_DIR = "models/checkpoints/b0_80_20"
    LOG_FILE = "models/logs/b0_80_20_results.json"

    os.makedirs("models/checkpoints/b0_80_20", exist_ok=True)
    os.makedirs("models/logs", exist_ok=True)

    create_80_20_split("data/metadata_train.json", METADATA_OUT)

    results, ckpt_path = train(
        metadata_path=METADATA_OUT,
        epochs=5,
        run_name="b0_80_20",
        batch_size=32,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    results["checkpoint_path"] = ckpt_path

    with open(LOG_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {LOG_FILE}")
    print(f"Best checkpoint: {ckpt_path}")
