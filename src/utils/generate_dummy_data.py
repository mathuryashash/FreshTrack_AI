import os
import pandas as pd
import numpy as np
from PIL import Image
import torch

def generate_dummy_dataset(output_dir='data', num_samples=20):
    """
    Generates a dummy dataset for testing the FreshTrack AI pipeline.
    """
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    data = []
    freshness_options = ['Fresh', 'Semi-ripe', 'Overripe', 'Rotten']
    quality_options = ['A', 'B', 'C']
    
    for i in range(num_samples):
        img_name = f"fruit_{i}.jpg"
        img_path = os.path.join(img_dir, img_name)
        
        # Create a random image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(img_path)
        
        # Random labels
        freshness = np.random.choice(freshness_options)
        quality = np.random.choice(quality_options)
        shelf_life = np.random.uniform(0, 10)
        temp = np.random.uniform(15, 35)
        humidity = np.random.uniform(30, 90)
        
        data.append({
            'image_path': img_name,
            'freshness': freshness,
            'quality_grade': quality,
            'shelf_life_days': shelf_life,
            'storage_temp': temp,
            'storage_humidity': humidity
        })
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, 'train.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated {num_samples} dummy samples in {output_dir}")

if __name__ == "__main__":
    generate_dummy_dataset()
