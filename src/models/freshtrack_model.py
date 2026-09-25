import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl

from src.config import (
    NUM_FRESHNESS_CLASSES,
    NUM_PRODUCE_TYPES,
    LOSS_WEIGHTS,
    DEFAULT_BACKBONE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WARMUP_EPOCHS,
)

TASK_CLASSES = {"freshness": NUM_FRESHNESS_CLASSES, "produce_type": NUM_PRODUCE_TYPES}


def entropy_bits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of softmax(logits) in bits, per row."""
    log_p = torch.log_softmax(logits, dim=1)
    return -(log_p.exp() * log_p).sum(dim=1) / torch.log(torch.tensor(2.0))


def energy_score(logits: torch.Tensor) -> torch.Tensor:
    """Negative free energy (Liu et al., 2020); higher = more in-distribution."""
    return torch.logsumexp(logits, dim=1)


class FreshTrackModel(pl.LightningModule):
    """Shared timm backbone with one linear-MLP head per task.

    tasks=("freshness", "produce_type") is the multi-task model; a single task
    gives the single-task baseline with an identical backbone and head.
    """

    def __init__(
        self,
        backbone=DEFAULT_BACKBONE,
        tasks=("freshness", "produce_type"),
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        warmup_epochs=DEFAULT_WARMUP_EPOCHS,
        max_epochs=10,
        pretrained=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tasks = tuple(tasks)

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        # MobileNetV3 has a conv head after the last stage: its pooled output
        # (head_hidden_size=1280) is wider than num_features (960).
        in_features = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features

        self.heads = nn.ModuleDict(
            {
                t: nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(in_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, TASK_CLASSES[t]),
                )
                for t in self.tasks
            }
        )
        # Renormalise so single- and multi-task losses share one scale.
        total = sum(LOSS_WEIGHTS[t] for t in self.tasks)
        self.loss_weights = {t: LOSS_WEIGHTS[t] / total for t in self.tasks}
        self._cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.backbone(x)
        return {t: head(features) for t, head in self.heads.items()}

    def _shared_step(self, batch, stage):
        images, labels = batch
        logits = self(images)
        total = 0.0
        for t in self.tasks:
            loss = self._cross_entropy(logits[t], labels[t])
            acc = (logits[t].argmax(dim=1) == labels[t]).float().mean()
            total = total + self.loss_weights[t] * loss
            self.log(f"{stage}_loss_{t}", loss)
            self.log(f"{stage}_acc_{t}", acc, prog_bar=True)
        self.log(f"{stage}_loss", total, prog_bar=True)
        return total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        warmup = self.hparams.warmup_epochs
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, self.hparams.max_epochs - warmup),
                    eta_min=1e-6,
                ),
            ],
            milestones=[warmup],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
