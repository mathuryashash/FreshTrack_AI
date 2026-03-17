import sys
import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import uvicorn

# Add project root to path for imports
sys.path.append(os.getcwd())

from src.models.freshtrack_model import FreshTrackModel

app = FastAPI(title="FreshTrack API", description="API for Freshness Detection")

# CORS for mobile access
ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:8501,http://localhost:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = os.environ.get(
    "MODEL_CHECKPOINT", "models/checkpoints/freshtrack_epoch=04_val_loss=0.01-v1.ckpt"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


@app.on_event("startup")
async def load_model():
    global model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = FreshTrackModel.load_from_checkpoint(MODEL_PATH)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")


def get_transforms():
    return A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


@app.get("/")
def read_root():
    return {"message": "FreshTrack API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    import os as _os

    ext = _os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use JPG, PNG, or WebP.",
        )

    # Read and validate file size (10 MB limit)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail="File too large. Maximum size is 10 MB."
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Transform
    transforms = get_transforms()
    aug = transforms(image=image_np)
    tensor = aug["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        fresh_logits, qual_logits, shelf_pred, _ = model(tensor)

    # Process results
    fresh_probs = torch.softmax(fresh_logits, dim=1)
    fresh_idx = torch.argmax(fresh_probs, dim=1).item()

    qual_probs = torch.softmax(qual_logits, dim=1)
    qual_idx = torch.argmax(qual_probs, dim=1).item()

    shelf_life = shelf_pred.item()

    # Mappings
    freshness_map = {0: "Fresh", 1: "Semi-ripe", 2: "Overripe", 3: "Rotten"}
    # Dataset uses A=0, B=1, C=2. Assuming A is best.
    quality_map = {0: "High (A)", 1: "Medium (B)", 2: "Low (C)"}

    return {
        "freshness": freshness_map.get(fresh_idx, "Unknown"),
        "freshness_confidence": float(fresh_probs[0][fresh_idx]),
        "quality": quality_map.get(qual_idx, "Unknown"),
        "shelf_life_days": round(shelf_life, 1),
    }


@app.post("/feedback")
async def submit_feedback(feedback: dict):
    """Store user feedback for model improvement."""
    import logging

    logger = logging.getLogger("freshtrack")
    logger.info(f"Feedback received: {feedback}")
    # TODO: persist feedback to database/file for retraining
    return {"status": "success", "message": "Feedback recorded. Thank you!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
