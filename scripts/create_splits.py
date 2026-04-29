
import json
import random
import hashlib
import os
import sys
# Add project root to path
sys.path.append(os.getcwd())

from pathlib import Path

def create_splits(metadata_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=None):
    """
    Split dataset into train/val/test with stratification.
    
    Args:
        metadata_file: Path to metadata JSON file
        train_ratio: Training split ratio (default 0.7)
        val_ratio: Validation split ratio (default 0.15)  
        test_ratio: Test split ratio (default 0.15)
        seed: Random seed for deterministic splits. If None, auto-derived from metadata.
    """
    if not os.path.exists(metadata_file):
        print(f"Error: {metadata_file} not found")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Set random seed for reproducibility
    if seed is None:
        # Derive deterministic seed from metadata content
        content_str = json.dumps(metadata, sort_keys=True)
        seed = int(hashlib.md5(content_str.encode()).hexdigest()[:8], 16)
    
    random.seed(seed)
    print(f"Using random seed: {seed} for deterministic splits")
    
    # Group by freshness for stratification
    freshness_groups = {}
    for item in metadata:
        freshness = item['freshness']
        if freshness not in freshness_groups:
            freshness_groups[freshness] = []
        freshness_groups[freshness].append(item)
    
    train_data, val_data, test_data = [], [], []
    
    for freshness, items in freshness_groups.items():
        random.shuffle(items)
        
        # Split
        n = len(items)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_items = items[:train_size]
        val_items = items[train_size:train_size + val_size]
        test_items = items[train_size + val_size:]
        
        # Add split label
        for item in train_items:
            item['split'] = 'train'
        for item in val_items:
            item['split'] = 'val'
        for item in test_items:
            item['split'] = 'test'
        
        train_data.extend(train_items)
        val_data.extend(val_items)
        test_data.extend(test_items)
    
    # Combine and save
    all_data = train_data + val_data + test_data
    
    output_path = Path('data/metadata_with_splits.json')
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    # Print distribution
    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        dist = {}
        for item in split_data:
            freshness = item['freshness']
            dist[freshness] = dist.get(freshness, 0) + 1
        print(f"{split_name} distribution: {dist}")

if __name__ == '__main__':
    create_splits('data/metadata.json')
