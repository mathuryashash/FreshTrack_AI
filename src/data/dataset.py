import torch
from torch.utils.data import Dataset
import cv2
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from src.config import (
    FRESHNESS_TO_IDX,
    PRODUCE_TO_IDX,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    IMAGE_SIZE,
)

IGNORE_LABEL = -100  # nn.CrossEntropyLoss default ignore_index


class FruitDataset(Dataset):
    def __init__(self, metadata_file, transform=None, split="train", split_field="split"):
        """
        Args:
            metadata_file: Path to metadata JSON (see src/data/build_splits.py)
            transform: Albumentations transform
            split: 'train', 'val', or 'test'
            split_field: 'split' (grouped, leakage-free) or 'split_naive'
                (per-file, leaky; used only for the leakage ablation)
        """
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Handle both formats: dict with "images" key or direct list
        if isinstance(metadata, dict) and "images" in metadata:
            images_data = metadata["images"]
        elif isinstance(metadata, list):
            images_data = metadata
        else:
            raise ValueError(f"Unsupported metadata format in {metadata_file}")

        # Filter by split
        self.data = [item for item in images_data if item.get(split_field) == split]
        self.transform = transform
        for item in self.data:
            # None = freshness not annotated (type-only sources); the loss skips it
            if item["freshness"] is not None and item["freshness"] not in FRESHNESS_TO_IDX:
                raise ValueError(f"Unknown freshness label {item['freshness']!r}")
            if item["produce_type"] not in PRODUCE_TO_IDX:
                raise ValueError(f"Unknown produce type {item['produce_type']!r}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_path = item["image_path"]
        image = cv2.imread(image_path)

        if image is None:
            # Handle missing images gracefully, or raise error
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        fresh = IGNORE_LABEL if item["freshness"] is None else FRESHNESS_TO_IDX[item["freshness"]]
        labels = {
            "freshness": torch.tensor(fresh, dtype=torch.long),
            "produce_type": torch.tensor(
                PRODUCE_TO_IDX[item["produce_type"]], dtype=torch.long
            ),
        }

        return image, labels


# Define transforms
def get_train_transforms(strong=False):
    if strong:
        # For the deployment model: phone photos are off-centre, cluttered,
        # blurred and recompressed, unlike the plain-background training data.
        return A.Compose(
            [
                A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.3, 1.0), ratio=(0.6, 1.67)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.05, p=0.8),
                A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 9))], p=0.25),
                A.ImageCompression(quality_range=(35, 95), p=0.4),
                A.GaussNoise(p=0.2),
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(16, 48),
                    hole_width_range=(16, 48),
                    fill=0,
                    p=0.3,
                ),
                A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(
                num_holes_range=(4, 4),
                hole_height_range=(32, 32),
                hole_width_range=(32, 32),
                fill=0,
                p=0.3,
            ),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ToTensorV2(),
        ]
    )
