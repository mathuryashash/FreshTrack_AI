import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FruitDataset(Dataset):
    """
    Custom Dataset for FreshTrack AI Multi-task learning.
    Expects a CSV file with: image_path, freshness, quality_grade, shelf_life_days, storage_temp, storage_humidity
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Mapping labels to integers
        self.freshness_map = {'Fresh': 0, 'Semi-ripe': 1, 'Overripe': 2, 'Rotten': 3}
        self.quality_map = {'A': 0, 'B': 1, 'C': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['image_path'])
        image = Image.open(img_name).convert('RGB')
        
        # Get labels
        freshness = self.freshness_map[self.data.iloc[idx]['freshness']]
        quality = self.quality_map[self.data.iloc[idx]['quality_grade']]
        shelf_life = float(self.data.iloc[idx]['shelf_life_days'])
        
        # Get metadata
        temp = float(self.data.iloc[idx]['storage_temp'])
        humidity = float(self.data.iloc[idx]['storage_humidity'])
        meta = torch.tensor([temp, humidity], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            image = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])(image)

        sample = {
            'image': image,
            'meta': meta,
            'freshness': freshness,
            'quality': quality,
            'shelf_life': torch.tensor([shelf_life], dtype=torch.float32)
        }

        return sample

if __name__ == "__main__":
    # Test script - will fail if no data exists, but shows the structure
    print("FruitDataset class defined. Ready for data integration.")
