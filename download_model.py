"""
Model Downloader for Production Deployment.
Downloads the FreshTrack AI model checkpoint from Hugging Face Hub on startup.
"""
import os
import hashlib
from huggingface_hub import hf_hub_download

MODEL_DIR = "models/checkpoints"
MODEL_FILENAME = "freshtrack_epoch=04_val_loss=0.01-v1.ckpt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
HF_REPO_ID = "YashashMathur/freshtrack-ai"

# Set EXPECTED_SHA256 to the known good checksum of the checkpoint file.
# Leave empty to skip verification (not recommended for production).
EXPECTED_SHA256 = os.environ.get("MODEL_SHA256", "")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists at {MODEL_PATH}. Skipping download.")
    else:
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"⬇️  Downloading model from Hugging Face: {HF_REPO_ID}...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=MODEL_DIR,
        )
        print(f"✅ Model downloaded to: {MODEL_PATH}")

    if EXPECTED_SHA256:
        actual = _sha256(MODEL_PATH)
        if actual != EXPECTED_SHA256:
            os.remove(MODEL_PATH)
            raise RuntimeError(
                f"Model checksum mismatch! Expected {EXPECTED_SHA256}, got {actual}. "
                "File removed. Set MODEL_SHA256 env var to the correct checksum."
            )
        print(f"✅ Checksum verified: {actual[:16]}...")

    return MODEL_PATH


if __name__ == "__main__":
    download_model()
