"""
Model Setup for Demo Deployment.
Sets up directories without requiring Hugging Face downloads.
"""
import os

MODEL_DIR = "models/checkpoints"
MODEL_FILENAME = "freshtrack_epoch=04_val_loss=0.01-v1.ckpt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model found at {MODEL_PATH}.")
    else:
        print("⚠️  Running in Demo Mode: No Hugging Face download configured.")
        print("⚠️  The API will start in degraded mode without a model.")
    return None

if __name__ == "__main__":
    download_model()
