
import os
import sys
sys.path.append(os.getcwd())

from src.data.dataset import FruitDataset, get_train_transforms

try:
    dataset = FruitDataset(
        'data/metadata.json', 
        transform=get_train_transforms(),
        split='train'
    )
    print(f"Dataset Loaded Successfully. Size: {len(dataset)}")
    img, labels = dataset[0]
    print(f"First item shape: {img.shape}")
    print(f"Labels: {labels}")
except Exception as e:
    print(f"Error loading dataset: {e}")
