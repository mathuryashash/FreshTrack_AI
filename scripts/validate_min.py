import os
import json
import torch
import cv2
import pandas as pd
import sys
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.getcwd())
try:
    from src.models.freshtrack_model import FreshTrackModel
except ImportError:
    print("Could not import model. Run from root.")
    sys.exit(1)

def main():
    print("Starting validation on New Fruits...")
    
    # 1. Load Model
    checkpoint_dir = os.path.join("models", "checkpoints")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoints:
        print("No checkpoints!")
        return
        
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Model: {os.path.basename(latest)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FreshTrackModel.load_from_checkpoint(latest)
    model.to(device)
    model.eval()
    
    # 2. Load Data (New Fruits Only)
    with open('data/metadata_stage4_fixed.json', 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")
    
    # 3. Transform
    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    freshness_map = {'Fresh': 0, 'Semi-ripe': 1, 'Overripe': 2, 'Rotten': 3}
    results = []
    
    # Limit to 50 random samples
    import random
    random.shuffle(data)
    subset = data[:50]
    
    print(f"Processing {len(subset)} samples...")
    for item in subset:
        path = item['image_path']
        if not os.path.exists(path): continue
        
        img = cv2.imread(path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        input_tensor = val_transform(image=img)['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            f_logits, _, s_answ, _ = model(input_tensor)
            
        probs = torch.softmax(f_logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
        gt = item['freshness']
        gt_idx = freshness_map.get(gt, -1)
        
        results.append({
            'Fruit': item.get('label', 'Unknown'),
            'Available Truth': gt,
            'Prediction': pred_idx, 
            'Match': (pred_idx == gt_idx),
            'Shelf Life': s_answ.item()
        })
        
    df = pd.DataFrame(results)
    if not df.empty:
        print("\n--- NEW FRUITS REPORT ---")
        summary = df.groupby('Fruit').agg(
            Count=('Fruit', 'count'),
            Accuracy=('Match', 'mean'),
            Avg_Shelf=('Shelf Life', 'mean')
        )
        print(summary)
        print(f"\nOverall Acc: {df['Match'].mean():.2%}")
        
        with open("validation_min_report.txt", "w") as f:
            f.write(summary.to_string())
            f.write(f"\n\nTotal Acc: {df['Match'].mean():.2%}")

if __name__ == "__main__":
    main()
