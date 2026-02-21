import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import mlflow
import os

from .network import multi_task_fruit_model
from .dataset import FruitDataset
try:
    from .evaluate import evaluate_model
except ImportError:
    # Handle relative import when running as script
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from evaluate import evaluate_model

def train_model(data_csv, img_dir, epochs=10, batch_size=16, lr=0.001, train_split=0.8):
    # 1. Setup MLflow
    mlflow.set_experiment("FreshTrackAI_MultiTask")
    
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "backbone": "resnet18",
            "train_split": train_split
        })

        # 2. Data Loading
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Check if data exists
        if not os.path.exists(data_csv):
            print(f"Data CSV {data_csv} not found. Skipping training.")
            return

        full_dataset = FruitDataset(data_csv, img_dir, transform=transform)
        
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        if val_size > 0:
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            train_dataset = full_dataset
            val_loader = None
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 3. Model, Loss, Optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = multi_task_fruit_model(backbone_name='resnet18', pretrained=True).to(device)
        
        criterion_freshness = nn.CrossEntropyLoss()
        criterion_quality = nn.CrossEntropyLoss()
        criterion_shelf_life = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 4. Training Loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for i, batch in enumerate(train_loader):
                images = batch['image'].to(device)
                meta = batch['meta'].to(device)
                freshness_labels = batch['freshness'].to(device)
                quality_labels = batch['quality'].to(device)
                shelf_life_labels = batch['shelf_life'].to(device)
                
                optimizer.zero_grad()
                
                outputs = model(images, meta)
                
                loss_freshness = criterion_freshness(outputs['freshness'], freshness_labels)
                loss_quality = criterion_quality(outputs['quality'], quality_labels)
                loss_shelf_life = criterion_shelf_life(outputs['shelf_life'], shelf_life_labels)
                
                # Combined loss with weighting
                # Freshness and quality are classification, shelf-life is regression
                loss = 1.0 * loss_freshness + 1.0 * loss_quality + 0.5 * loss_shelf_life
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # 5. Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch['image'].to(device)
                        meta = batch['meta'].to(device)
                        freshness_labels = batch['freshness'].to(device)
                        quality_labels = batch['quality'].to(device)
                        shelf_life_labels = batch['shelf_life'].to(device)
                        
                        outputs = model(images, meta)
                        loss_f = criterion_freshness(outputs['freshness'], freshness_labels)
                        loss_q = criterion_quality(outputs['quality'], quality_labels)
                        loss_s = criterion_shelf_life(outputs['shelf_life'], shelf_life_labels)
                        
                        val_loss += (loss_f + loss_q + 0.5 * loss_s).item()
                
                avg_val_loss = val_loss / len(val_loader)
                print(f"Validation Loss: {avg_val_loss:.4f}")
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        # 6. Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/final_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Log model with registry
        mlflow.pytorch.log_model(
            model, 
            artifact_path="model",
            registered_model_name="FreshTrackFruitModel"
        )
        mlflow.log_artifact(model_path)
        
        # 7. Final Evaluation on full train set (as a proxy for now)
        print("\nFinal Evaluation on all data:")
        evaluate_model(model_path, data_csv, img_dir)

if __name__ == "__main__":
    # Check if data exists and run training if so
    DATA_CSV = 'data/train.csv'
    IMG_DIR = 'data/images/'
    
    if os.path.exists(DATA_CSV):
        print("Starting training with available data...")
        train_model(DATA_CSV, IMG_DIR, epochs=5)
    else:
        print("Training script initialized. Please provide data to start training.")
