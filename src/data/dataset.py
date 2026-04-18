import torch
from torch.utils.data import Dataset
import cv2
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from src.config import (
    FRESHNESS_TO_IDX,
    QUALITY_TO_IDX,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    IMAGE_SIZE,
)


class FruitDataset(Dataset):
    def __init__(self, metadata_file, transform=None, split="train"):
        """
        Args:
            metadata_file: Path to metadata.json
            transform: Albumentations transform
            split: 'train', 'val', or 'test'
        """
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        # Filter by split
        self.data = [item for item in self.metadata if item.get("split") == split]
        self.transform = transform

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

        # Prepare labels - use config mappings
        freshness_label = FRESHNESS_TO_IDX.get(item["freshness"], 0)
        quality_label = QUALITY_TO_IDX.get(item["quality"], 1)

        # Rotation: 0=0°, 1=90°, 2=180°, 3=270° (derived from image filename if available)
        rotation_label = int(item.get("rotation", 0)) % 4

        labels = {
            "freshness": torch.tensor(freshness_label, dtype=torch.long),
            "quality": torch.tensor(quality_label, dtype=torch.long),
            "shelf_life": torch.tensor([item["shelf_life_days"]], dtype=torch.float32),
            "rotation": torch.tensor(rotation_label, dtype=torch.long),
        }

        return image, labels


# Define transforms
def get_train_transforms():
    return A.Compose(
        [
            A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(
                max_holes=4,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
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
