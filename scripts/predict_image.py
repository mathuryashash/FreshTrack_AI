import torch
import torchvision.transforms as T
from PIL import Image
import os
import sys
import glob
import argparse

# Add project root to path
sys.path.append(os.getcwd())

from src.models.freshtrack_model import FreshTrackModel

def load_best_model(device):
    checkpoint_dir = os.path.join("models", "checkpoints")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    if not checkpoints:
        print("Error: No checkpoints found!")
        sys.exit(1)
        
    # Logic from API: Prefer robust checkpoints (Stage 1/2) over Stage 3 if available
    # because Stage 3 might be biased towards Fresh Only.
    # However, if user wants to test Stage 3 specifically, they can modify this or use --checkpoint arg
    robust_checkpoints = [ckpt for ckpt in checkpoints if "stage3" not in ckpt]
    
    if robust_checkpoints:
        latest = max(robust_checkpoints, key=os.path.getmtime)
        print(f"Selected robust checkpoint: {os.path.basename(latest)}")
    else:
        latest = max(checkpoints, key=os.path.getmtime)
        print(f"Selected latest checkpoint: {os.path.basename(latest)}")
        
    model = FreshTrackModel.load_from_checkpoint(latest)
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        model = FreshTrackModel.load_from_checkpoint(checkpoint_path)
        model.to(device)
        model.eval()
    else:
        model = load_best_model(device)
        
    # Transforms
    transforms = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = transforms(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    with torch.no_grad():
        fresh_logits, qual_logits, shelf_pred, _ = model(tensor)
        
        # Post-processing
        fresh_probs = torch.softmax(fresh_logits, dim=1)
        fresh_class_idx = torch.argmax(fresh_probs, dim=1).item()
        
        qual_probs = torch.softmax(qual_logits, dim=1)
        qual_class_idx = torch.argmax(qual_probs, dim=1).item()
        
        shelf_life = shelf_pred.item()
        
        freshness_map = {0: 'Fresh', 1: 'Semi-ripe', 2: 'Overripe', 3: 'Rotten'}
        quality_map = {0: 'A', 1: 'B', 2: 'C'}
        
        print("\n" + "="*30)
        print(f"Image: {os.path.basename(image_path)}")
        print("="*30)
        print(f"Freshness:   {freshness_map.get(fresh_class_idx, 'Unknown')} ({fresh_probs[0][fresh_class_idx]:.2%})")
        print(f"Quality:     {quality_map.get(qual_class_idx, 'Unknown')} ({qual_probs[0][qual_class_idx]:.2%})")
        print(f"Shelf Life:  {shelf_life:.1f} days")
        print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_image.py <path_to_image> [checkpoint_path]")
    else:
        img_path = sys.argv[1]
        ckpt = sys.argv[2] if len(sys.argv) > 2 else None
        predict_image(img_path, ckpt)
