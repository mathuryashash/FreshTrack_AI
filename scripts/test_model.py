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

def load_best_model():
    """Load the best checkpoint from models/checkpoints"""
    checkpoint_dir = "models/checkpoints"
    # Find all .ckpt files
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    if not checkpoints:
        print("Error: No checkpoints found. Please complete training first.")
        sys.exit(1)
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    latest_checkpoint = checkpoints[0]
    print(f"Loading model from: {latest_checkpoint}")
    
    model = FreshTrackModel.load_from_checkpoint(latest_checkpoint)
    model.eval()
    return model

def predict_message(image_path, model):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Preprocess
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            fresh_logits, qual_logits, shelf_pred, _ = model(tensor)
            
            # Post-processing
            fresh_probs = torch.softmax(fresh_logits, dim=1)
            fresh_class_idx = torch.argmax(fresh_probs, dim=1).item()
            
            qual_probs = torch.softmax(qual_logits, dim=1)
            qual_class_idx = torch.argmax(qual_probs, dim=1).item()
            
            shelf_life = shelf_pred.item()
            
            # Mappings
            freshness_map = {0: 'Fresh', 1: 'Semi-ripe', 2: 'Overripe', 3: 'Rotten'}
            quality_map = {0: 'A', 1: 'B', 2: 'C'}
            
            print("\n" + "="*30)
            print("PREDICTION RESULTS")
            print("="*30)
            print(f"Image: {image_path}")
            print(f"Freshness: {freshness_map.get(fresh_class_idx, 'Unknown')} ({fresh_probs[0][fresh_class_idx].item()*100:.1f}%)")
            print(f"Quality:   {quality_map.get(qual_class_idx, 'Unknown')} ({qual_probs[0][qual_class_idx].item()*100:.1f}%)")
            print(f"Shelf Life: {shelf_life:.1f} days")
            print("="*30 + "\n")
            
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FreshTrack Model on an image")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()
    
    model = load_best_model()
    predict_message(args.image_path, model)
