"""API unit tests — run with: pytest tests/test_api.py -v"""
import io
import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.api.main import app

client = TestClient(app, raise_server_exceptions=False)


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal valid PNG in memory."""
    img = Image.new("RGB", (width, height), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Basic endpoints ───────────────────────────────────────────────────────────

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "FreshTrack API is running"
    assert "version" not in data


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" not in data


# ── /predict validation ───────────────────────────────────────────────────────

def test_predict_no_file():
    """Missing file should return 422 Unprocessable Entity."""
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_wrong_extension():
    """Non-image extension should return 400."""
    response = client.post(
        "/predict",
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_predict_wrong_content_type():
    """Wrong MIME type should return 400."""
    response = client.post(
        "/predict",
        files={"file": ("image.jpg", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_corrupt_image():
    """Corrupt image bytes should return 400."""
    response = client.post(
        "/predict",
        files={"file": ("corrupt.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, "image/png")},
    )
    assert response.status_code == 400


def test_predict_oversized_file():
    """File exceeding size limit should return 413."""
    big_data = b"A" * (11 * 1024 * 1024)  # 11 MB
    response = client.post(
        "/predict",
        files={"file": ("big.jpg", big_data, "image/jpeg")},
    )
    assert response.status_code == 413


# ── /feedback validation ──────────────────────────────────────────────────────

def test_feedback_invalid_freshness():
    """Invalid freshness label should return 422."""
    response = client.post(
        "/feedback",
        json={
            "image_id": str(uuid.uuid4()),
            "predicted_freshness": "Fresh",
            "correct_freshness": "NotALabel",
        },
    )
    assert response.status_code == 422


def test_feedback_valid(monkeypatch):
    """Valid feedback payload should return 200 (DB call mocked)."""
    import src.api.main as main_module
    monkeypatch.setattr(main_module, "log_feedback", lambda **_: str(uuid.uuid4()))
    response = client.post(
        "/feedback",
        json={
            "image_id": str(uuid.uuid4()),
            "predicted_freshness": "Fresh",
            "correct_freshness": "Rotten",
            "notes": "Clearly rotten on the bottom",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
