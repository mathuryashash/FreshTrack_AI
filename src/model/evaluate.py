import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, classification_report
import os

from .network import multi_task_fruit_model
from .dataset import FruitDataset

def evaluate_model(model_path, data_csv, img_dir, batch_size=16):
    """
    Evaluates the multi-task model on a given dataset.
    """
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Data Loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(data_csv):
        print(f"Data CSV {data_csv} not found. Skipping evaluation.")
        return

    dataset = FruitDataset(data_csv, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 3. Model Loading
    model = multi_task_fruit_model(backbone_name='resnet18', pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model path {model_path} not found. Using randomly initialized model.")
    
    model.to(device)
    model.eval()

    # 4. Inference
    all_freshness_preds = []
    all_freshness_labels = []
    all_quality_preds = []
    all_quality_labels = []
    all_shelf_life_preds = []
    all_shelf_life_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            meta = batch['meta'].to(device)
            freshness_labels = batch['freshness']
            quality_labels = batch['quality']
            shelf_life_labels = batch['shelf_life']

            outputs = model(images, meta)

            # Get predictions
            _, freshness_preds = torch.max(outputs['freshness'], 1)
            _, quality_preds = torch.max(outputs['quality'], 1)
            shelf_life_preds = outputs['shelf_life']

            all_freshness_preds.extend(freshness_preds.cpu().numpy())
            all_freshness_labels.extend(freshness_labels.numpy())
            
            all_quality_preds.extend(quality_preds.cpu().numpy())
            all_quality_labels.extend(quality_labels.numpy())
            
            all_shelf_life_preds.extend(shelf_life_preds.cpu().numpy())
            all_shelf_life_labels.extend(shelf_life_labels.numpy())

    # 5. Calculate Metrics
    f1_freshness = f1_score(all_freshness_labels, all_freshness_preds, average='weighted')
    f1_quality = f1_score(all_quality_labels, all_quality_preds, average='weighted')
    mae_shelf_life = mean_absolute_error(all_shelf_life_labels, all_shelf_life_preds)

    print("\nEvaluation Metrics:")
    print(f"Freshness F1-score: {f1_freshness:.4f}")
    print(f"Quality F1-score: {f1_quality:.4f}")
    print(f"Shelf-life MAE: {mae_shelf_life:.4f}")
    
    print("\nClassification Report (Freshness):")
    print(classification_report(all_freshness_labels, all_freshness_preds, target_names=['Fresh', 'Semi-ripe', 'Overripe', 'Rotten']))

    # 6. Log to MLflow
    if mlflow.active_run():
        mlflow.log_metrics({
            "val_f1_freshness": f1_freshness,
            "val_f1_quality": f1_quality,
            "val_mae_shelf_life": mae_shelf_life
        })

    return {
        "f1_freshness": f1_freshness,
        "f1_quality": f1_quality,
        "mae_shelf_life": mae_shelf_life
    }

if __name__ == "__main__":
    # Example usage
    # evaluate_model('models/final_model.pth', 'data/train.csv', 'data/images/')
    print("Evaluation script initialized.")
