import os
import json
import argparse
import random

def generate_metadata(data_dir, output_file):
    metadata = []
    
    # Expected structure: data_dir/FruitName/FreshnessClass/image.jpg
    # FreshnessClass mapper
    freshness_map = {
        'fresh': 'Fresh',
        'rotten': 'Rotten',
        'semi': 'Semi-ripe',
        'over': 'Overripe'
    }
    
    # Walk through the directory
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start=os.getcwd()) # Store relative path
                
                parts = full_path.replace('\\', '/').split('/')
                
                # Heuristic to find freshness from folder name
                freshness = "Unknown"
                for part in parts:
                    lower_part = part.lower()
                    if 'fresh' in lower_part: freshness = 'Fresh'
                    elif 'rotten' in lower_part: freshness = 'Rotten'
                    elif 'semi' in lower_part: freshness = 'Semi-ripe'
                    elif 'over' in lower_part: freshness = 'Overripe'
                
                # Heuristic for Quality
                quality = "A" if freshness == "Fresh" else "C" if freshness == "Rotten" else "B"
                
                # Heuristic for Shelf Life
                shelf_life = 7.0 if freshness == "Fresh" else 0.0 if freshness == "Rotten" else 3.0
                
                # Random split
                r = random.random()
                if r < 0.8: split = 'train'
                elif r < 0.9: split = 'val'
                else: split = 'test'
                
                entry = {
                    "image_path": rel_path,
                    "freshness": freshness,
                    "quality": quality,
                    "shelf_life_days": shelf_life,
                    "split": split
                }
                metadata.append(entry)
    
    print(f"Found {len(metadata)} images.")
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Saved metadata to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to your new data folder")
    parser.add_argument("--output", default="data/metadata_new.json", help="Output JSON file")
    args = parser.parse_args()
    
    generate_metadata(args.data_dir, args.output)
