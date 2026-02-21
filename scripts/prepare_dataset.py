
import os
import json
from pathlib import Path

# Map folder names to standardized labels
FRESHNESS_MAPPING = {
    'freshapples': 'Fresh',
    'freshbanana': 'Fresh',
    'freshoranges': 'Fresh',
    'rottenapples': 'Rotten',
    'rottenbanana': 'Rotten',
    'rottenoranges': 'Rotten',
    'fresh': 'Fresh',
    'rotten': 'Rotten',
    'good quality_fruits': 'Fresh',
    'bad quality_fruits': 'Rotten'
}

# Heuristic Shelf Life Mapping (Days)
SHELF_LIFE_MAPPING = {
    'Fresh': 7.0,
    'Rotten': 0.0
}

# Quality Mapping based on Freshness
QUALITY_MAPPING = {
    'Fresh': 'A',
    'Rotten': 'C'
}



def log(msg):
    with open("scripts/prepare_log.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)

def create_metadata():
    """
    Create separate metadata json files for each dataset:
    - data/metadata_fruitnet.json
    - data/metadata_fruits360.json
    - data/metadata_fruitquality.json
    """
    
    # Clear log file
    with open("scripts/prepare_log.txt", "w") as f:
        f.write("Starting prepare_dataset.py\n")

    # Define dataset configurations
    # (path, type, output_filename)
    datasets_config = [
        (Path("FruitNet_Indian"), "quality_folders", "metadata_fruitnet.json"),
        (Path("Fruit_Quality_Classification"), "quality_folders", "metadata_fruitquality.json"),
        (Path("Fruits_360"), "fruits_360", "metadata_fruits360.json")
    ]
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    log(f"Scanning for datasets...")

    for base_path, dataset_type, output_filename in datasets_config:
        if not base_path.exists():
            log(f"Skipping {base_path} (not found)")
            continue
            
        log(f"Processing {base_path} into {output_filename}...")
        metadata = []
        
        count = 0
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    full_path = Path(root) / file
                    parts = full_path.parts
                    
                    freshness = None
                    split = 'train' # Default split
                    
                    # Logic based on dataset type
                    if dataset_type == "quality_folders":
                        # Look for "Good Quality_Fruits" or "Bad Quality_Fruits" in path
                        if any("Good Quality" in part for part in parts):
                            freshness = "Fresh"
                        elif any("Bad Quality" in part for part in parts):
                            freshness = "Rotten"
                        
                        # Deterministic split based on filename hash
                        h = hash(file) % 10
                        if h < 2:
                            split = 'test'
                        elif h < 4:
                            split = 'val'
                        else:
                            split = 'train'

                    elif dataset_type == "fruits_360":
                        # Determine split from folder name
                        if 'Test' in parts or 'test' in parts:
                            split = 'test'
                        else:
                            # Split Training folder into Train/Val
                            h = hash(file) % 10
                            if h < 2:
                                split = 'val'
                            else:
                                split = 'train'
                            
                        # Determine freshness from class name (parent folder)
                        class_name = full_path.parent.name.lower()
                        if 'rotten' in class_name:
                            freshness = 'Rotten'
                        else:
                            freshness = 'Fresh'

                    # Add to metadata if we found a valid label
                    if freshness:
                        quality = QUALITY_MAPPING.get(freshness, 'B')
                        shelf_life = SHELF_LIFE_MAPPING.get(freshness, 3.0)
                        
                        # Use absolute path
                        abs_path = full_path.absolute()
                        try:
                            rel_path = abs_path.relative_to(Path.cwd())
                        except ValueError:
                            rel_path = abs_path

                        metadata.append({
                            'image_path': str(rel_path),
                            'freshness': freshness,
                            'quality': quality,
                            'shelf_life_days': shelf_life,
                            'split': split,
                            'source': str(base_path)
                        })
                        count += 1
                        if count % 1000 == 0:
                            log(f"  Processed {count} images...")

        # Save individual metadata file
        output_file = Path('data') / output_filename
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log(f"Created {output_filename} with {len(metadata)} images")

if __name__ == '__main__':
    create_metadata()

