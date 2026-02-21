
import os
import json
from pathlib import Path

# Mapping for quality and shelf life based on freshness folder name
FRESHNESS_MAP = {
    'Fresh': {'quality': 'A', 'shelf_life': 7.0},
    'Rotten': {'quality': 'C', 'shelf_life': 0.0},
    'Semi-ripe': {'quality': 'B', 'shelf_life': 3.0},
    'Overripe': {'quality': 'C', 'shelf_life': 1.0}
}

def create_stage4_metadata():
    base_path = Path("data/New_Fruits")
    output_filename = "data/metadata_stage4_fixed.json"
    
    print(f"Scanning {base_path}...")
    metadata = []
    
    if not base_path.exists():
        print("Base path does not exist.")
        return

    for fruit_dir in base_path.iterdir():
        if not fruit_dir.is_dir():
            continue
            
        fruit_name = fruit_dir.name
        print(f"Processing {fruit_name}...")
        
        has_subdirs = any(x.is_dir() for x in fruit_dir.iterdir())
        
        if not has_subdirs:
            print(f"  {fruit_name} appears to be flat. Assuming 'Fresh'.")
            freshness_label = 'Fresh'
            props = FRESHNESS_MAP['Fresh']
            
            for img_file in fruit_dir.glob("*"):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue
                    
                # Deterministic split
                h = hash(img_file.name) % 10
                if h < 2:
                    split = 'test'
                elif h < 4:
                    split = 'val'
                else:
                    split = 'train'
                
                metadata.append({
                    'image_path': str(img_file),
                    'freshness': freshness_label,
                    'quality': props['quality'],
                    'shelf_life_days': props['shelf_life'],
                    'split': split,
                    'label': fruit_name,
                    'source': str(base_path)
                })
        else:
            for freshness_dir in fruit_dir.iterdir():
                if not freshness_dir.is_dir():
                    continue
                    
                freshness_label = freshness_dir.name
                props = FRESHNESS_MAP.get(freshness_label, {'quality': 'B', 'shelf_life': 3.0})
                
                for img_file in freshness_dir.glob("*"):
                    if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                        continue
                        
                    # Deterministic split
                    h = hash(img_file.name) % 10
                    if h < 2:
                        split = 'test'
                    elif h < 4:
                        split = 'val'
                    else:
                        split = 'train'
                    
                    metadata.append({
                        'image_path': str(img_file),
                        'freshness': freshness_label,
                        'quality': props['quality'],
                        'shelf_life_days': props['shelf_life'],
                        'split': split,
                        'label': fruit_name,
                        'source': str(base_path)
                    })

    print(f"Found {len(metadata)} images.")
    
    if len(metadata) > 0:
        with open(output_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {output_filename}")
    else:
        print("No images found. Check directory structure.")

if __name__ == "__main__":
    create_stage4_metadata()
