import os
import json
import glob
import torch
import cv2
import random
import numpy as np
import pandas as pd
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())
from src.models.freshtrack_model import FreshTrackModel

def get_transforms():
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def extract_label(item):
    """Extract label from item, inferring from path if necessary."""
    if 'label' in item:
        return item['label']
    
    path = item['image_path'].lower()
    if 'banana' in path: return 'Banana'
    if 'apple' in path: return 'Apple'
    if 'orange' in path: return 'Orange'
    if 'cucumber' in path: return 'Cucumber'
    if 'capsicum' in path: return 'Capsicum'
    if 'bitter_gourd' in path or 'bittergourd' in path: return 'Bitter_Gourd'
    if 'strawberry' in path: return 'Strawberry'
    if 'tomato' in path: return 'Tomato'
    if 'potato' in path: return 'Potato'
    
    return 'Unknown'

def fix_path(path):
    # Mapping for old metadata paths to current location
    if os.path.exists(path): return path
    
    # Try replacing backslashes
    path = path.replace('\\', '/')
    if os.path.exists(path): return path
    
    # Check for "archive/dataset/dataset..." pattern
    if 'archive/dataset/dataset' in path:
        # Extract filename and parent folder (e.g., freshbanana)
        parts = path.split('/')
        filename = parts[-1]
        parent = parts[-2]
        
        # Map parent folders
        mapping = {
            'freshbanana': 'fresh_banana',
            'freshapples': 'fresh_apples',
            'freshoranges': 'fresh_oranges',
            'rottenbanana': 'rotten_banana',
            'rottenapples': 'rotten_apples',
            'rottenoranges': 'rotten_oranges',
            'freshcucumber': 'fresh_cucumber', # guess
            'rottencucumber': 'rotten_cucumber' # guess
            # Add others if needed or rely on heuristic
        }
        
        new_parent = mapping.get(parent, parent)
        
        # Try constructing new path in downloads
        base = "data/downloads/fresh-and-stale-images-of-fruits-and-vegetables"
        candidate = os.path.join(base, new_parent, filename)
        if os.path.exists(candidate): return candidate
        
        # Try with original parent name just in case
        candidate = os.path.join(base, parent, filename)
        if os.path.exists(candidate): return candidate

    return path

def load_data(sample_per_class=20):
    files = [
        'data/metadata_stage4_fixed.json',
        'data/metadata_with_splits.json'
    ]
    
    all_items = []
    
    print("Loading metadata...")
    for f_path in files:
        if os.path.exists(f_path):
            with open(f_path, 'r') as f:
                data = json.load(f)
                print(f"Loaded {len(data)} items from {f_path}")
                all_items.extend(data)
    
    # Group by label
    grouped = {}
    print("Grouping items...")
    for item in all_items:
        label = extract_label(item)
        if label == 'Unknown': continue
        
        # Optimization: Don't fix path here. Just group.
        if label not in grouped:
            grouped[label] = []
        grouped[label].append(item)
        
    # Sample and Fix Paths
    sampled_data = []
    print(f"Sampling {sample_per_class} images per class...")
    for label, items in grouped.items():
        # Prefer 'test' split if available, else 'val', else 'train'
        test_items = [x for x in items if x.get('split') == 'test']
        if len(test_items) < sample_per_class:
            test_items.extend([x for x in items if x.get('split') == 'val'])
        if len(test_items) < sample_per_class:
            test_items.extend([x for x in items if x.get('split') == 'train'])
            
        # Shuffle
        random.shuffle(test_items)
        
        # Pick valid ones until we have enough
        selected = []
        for candidate in test_items:
             if len(selected) >= sample_per_class: break
             
             # Fix path NOW
             candidate['image_path'] = fix_path(candidate['image_path'])
             if os.path.exists(candidate['image_path']):
                 candidate['param_label'] = label
                 selected.append(candidate)
        
        sampled_data.extend(selected)
        print(f"  - {label}: Found {len(items)} total, selected {len(selected)}")
        
    return sampled_data

def validate():
    # Load Model
    checkpoint_dir = os.path.join("models", "checkpoints")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoints:
        print("No checkpoints found!")
        return

    # Prefer robust checkpoints
    robust_checkpoints = [ckpt for ckpt in checkpoints if "stage3" not in ckpt]
    latest = max(robust_checkpoints, key=os.path.getmtime) if robust_checkpoints else max(checkpoints, key=os.path.getmtime)
    
    print(f"Loading Model: {os.path.basename(latest)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FreshTrackModel.load_from_checkpoint(latest)
    model.to(device)
    model.eval()
    
    # Load Data
    data = load_data(sample_per_class=20)
    transforms = get_transforms()
    
    freshness_map = {'Fresh': 0, 'Semi-ripe': 1, 'Overripe': 2, 'Rotten': 3}
    
    results = []
    
    print(f"\nRunning Validation on {len(data)} images...")
    for item in tqdm(data):
        img_path = item['image_path']
        
        try:
            image = cv2.imread(img_path)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            aug = transforms(image=image)
            tensor = aug['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                fresh_logits, _, shelf_pred, _ = model(tensor)
                
            # Predictions
            fresh_probs = torch.softmax(fresh_logits, dim=1)
            pred_fresh_idx = torch.argmax(fresh_probs, dim=1).item()
            pred_shelf = shelf_pred.item()
            
            # Ground Truth
            gt_fresh = item['freshness']
            gt_fresh_idx = freshness_map.get(gt_fresh, -1)
            
            # Heuristic Logic (Ported from UI for consistency)
            # Strict logic check
            probs = {i: fresh_probs[0][i].item() for i in range(4)}
            final_idx = pred_fresh_idx
            
            if final_idx == 0:
                score = probs[2] * 1.5 + probs[3] * 2.5
                if score > 0.15:
                     final_idx = 3 if probs[3] > probs[2] else 2
                elif probs[1] > 0.30:
                     final_idx = 1
                     
            is_correct = (final_idx == gt_fresh_idx)
            
            results.append({
                'Label': item['param_label'],
                'Freshness Spec': gt_fresh,
                'Predicted idx': final_idx,
                'Is Correct': is_correct,
                'Shelf Life Pred': pred_shelf
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    # Summary
    df = pd.DataFrame(results)
    if df.empty:
        print("No results generated.")
        return

    print("\n--- Validation Report ---\n")
    # Group by Label
    summary = df.groupby('Label').agg(
        Samples=('Label', 'count'),
        Accuracy=('Is Correct', lambda x: f"{x.mean():.1%}"),
        Avg_Shelf_Life=('Shelf Life Pred', lambda x: f"{x.mean():.1f} d")
    )
    
    print(summary)
    
    total_acc = df['Is Correct'].mean()
    print(f"\nOverall Accuracy: {total_acc:.1%}")

    # Write to file
    with open("validation_report.txt", "w") as f:
        f.write("--- Validation Report ---\n\n")
        f.write(summary.to_string())
        f.write(f"\n\nOverall Accuracy: {total_acc:.1%}\n")
    print("Report saved to validation_report.txt")


if __name__ == "__main__":
    validate()
