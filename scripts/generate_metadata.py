import os
import json
import argparse
import hashlib


def deterministic_split(file_path: str, split_ratios=(0.8, 0.1, 0.1)):
    """Deterministic split using SHA-256 hash of file path.
    
    The same file path always maps to the same split, making
    experiments fully reproducible across runs.
    """
    hash_bytes = hashlib.sha256(file_path.encode()).digest()
    # Use first 4 bytes of hash as a seed value in [0, 1)
    rand = int.from_bytes(hash_bytes[:4], 'big') / (2**32)

    cumulative = 0.0
    splits = ['train', 'val', 'test']
    for i, ratio in enumerate(split_ratios):
        cumulative += ratio
        if rand < cumulative:
            return splits[i]
    return splits[-1]  # fallback (rounding edge case)


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
                
                # Deterministic split based on file path hash
                split = deterministic_split(rel_path)
                
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
