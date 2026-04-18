import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torch
import os
import sys

sys.path.append(os.getcwd())

from src.models.freshtrack_model import FreshTrackModel
from src.data.dataset import FruitDataset, get_train_transforms, get_val_transforms
from src.config import DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_NUM_WORKERS


import argparse


def train(metadata_path, resume_from, epochs, run_name, batch_size=DEFAULT_BATCH_SIZE):
    print(f"Starting training with metadata: {metadata_path}")
    print(f"Resume from: {resume_from}")
    print(f"Epochs: {epochs}")
    print(f"Run Name: {run_name}")
    print(f"Batch Size: {batch_size}")

    os.makedirs("models/checkpoints", exist_ok=True)

    wandb_logger = WandbLogger(
        project="freshtrack-ai",
        name=run_name,
        config={
            "architecture": "EfficientNet-B0",
            "batch_size": batch_size,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "epochs": epochs,
            "metadata": metadata_path,
            "resume_from": resume_from,
        },
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename=f"{run_name}_{{epoch:02d}}_{{val_loss:.2f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True,
        check_on_train_epoch_end=False,
    )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    train_dataset = FruitDataset(
        metadata_path, transform=get_train_transforms(), split="train"
    )

    val_dataset = FruitDataset(
        metadata_path, transform=get_val_transforms(), split="val"
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
        val_dataset,
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
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=2,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)

    test_dataset = FruitDataset(
        metadata_path, transform=get_val_transforms(), split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", type=str, required=True, help="Path to metadata.json"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--name", type=str, default="run", help="Name of the run")
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size"
    )

    args = parser.parse_args()

    train(args.metadata, args.resume_from, args.epochs, args.name, args.batch_size)
