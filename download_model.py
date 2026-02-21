"""
Model Downloader for Render Deployment.
Run this before starting the API server.
Uploads the .ckpt file to Google Drive and pastes the file ID below.
"""
import os
import gdown

MODEL_DIR = "models/checkpoints"
MODEL_PATH = os.path.join(MODEL_DIR, "freshtrack_epoch=04_val_loss=0.01-v1.ckpt")

# TODO: Upload your .ckpt file to Google Drive, share it publicly (Anyone with link),
# then replace this with the file ID from the Drive share URL.
# Share URL example: https://drive.google.com/file/d/XXXXXXXXXX_FILE_ID/view
GOOGLE_DRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}. Skipping download.")
        return
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"Downloading model checkpoint...")
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded successfully!")

if __name__ == "__main__":
    download_model()
