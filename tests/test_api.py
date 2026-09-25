"""API unit tests — run with: pytest tests/test_api.py -v"""
import io
import os
import sys
import uuid

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.api.main as main_module
from src.api.main import app, derive_quality, derive_shelf_life
from src.config import PRODUCE_TYPES, SHELF_LIFE_REFERENCE_DAYS

client = TestClient(app, raise_server_exceptions=False)


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal valid PNG in memory."""
    img = Image.new("RGB", (width, height), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeModel:
    """Stands in for the checkpoint: fixed logits, apple, 90% fresh."""

    def __call__(self, x):
        fresh = torch.log(torch.tensor([[0.9, 0.1]]))
        types = torch.full((1, len(PRODUCE_TYPES)), -5.0)
        types[0, PRODUCE_TYPES.index("apple")] = 5.0
        return {"freshness": fresh, "produce_type": types}


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(main_module, "model", _FakeModel())
    monkeypatch.setattr(main_module, "log_prediction", lambda **_: str(uuid.uuid4()))

    def with_threshold(threshold):
        monkeypatch.setattr(
            main_module,
            "model_meta",
            {"ood_score": "energy_produce_type", "ood_threshold": threshold},
        )

    return with_threshold


# ── Basic endpoints ───────────────────────────────────────────────────────────

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "FreshTrack AI API is running"


def test_health_reports_degraded_without_model():
    """503 so Docker's `curl -f` HEALTHCHECK fails when no model is loaded."""
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["model_loaded"] is False
    assert data["status"] == "degraded"


# ── /predict validation ───────────────────────────────────────────────────────

def test_predict_no_file():
    """Missing file should return 422 Unprocessable Entity."""
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_wrong_extension():
    """Non-image extension should return 400."""
    response = client.post(
        "/predict",
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/x-msdownload")},
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


def test_predict_without_model_returns_503():
    response = client.post(
        "/predict", files={"file": ("ok.png", _make_png_bytes(), "image/png")}
    )
    assert response.status_code == 503


# ── /predict inference paths (fake model) ─────────────────────────────────────

def test_predict_success_marks_heuristics(fake_model):
    fake_model(-1e9)  # accept everything
    response = client.post(
        "/predict", files={"file": ("ok.png", _make_png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["freshness"] == "Fresh"
    assert data["produce_type"] == "apple"
    assert data["quality"] == "High (A)"
    assert data["quality_is_heuristic"] is True
    assert data["shelf_life_is_heuristic"] is True
    assert data["shelf_life_days"] == pytest.approx(SHELF_LIFE_REFERENCE_DAYS["apple"] * 0.9, abs=0.05)


def test_predict_ood_returns_200_with_error(fake_model):
    """Regression: OOD used to 500 because the error dict failed response_model."""
    fake_model(1e9)  # reject everything
    response = client.post(
        "/predict", files={"file": ("ok.png", _make_png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["error"] == "OBJECT_NOT_RECOGNIZED"


# ── Heuristic helpers ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("p,grade", [(0.95, "High (A)"), (0.6, "Medium (B)"), (0.1, "Low (C)")])
def test_derive_quality(p, grade):
    assert derive_quality(p) == grade


def test_derive_shelf_life_scales_with_p_fresh():
    assert derive_shelf_life("banana", 1.0) == SHELF_LIFE_REFERENCE_DAYS["banana"]
    assert derive_shelf_life("banana", 0.0) == 0.0


# ── /feedback validation ──────────────────────────────────────────────────────

def test_feedback_invalid_freshness():
    """Invalid freshness label should return 422."""
    response = client.post(
        "/feedback",
        json={
            "image_id": str(uuid.uuid4()),
            "predicted_freshness": "Fresh",
            "correct_freshness": "Rotten",  # not a label since v2
        },
    )
    assert response.status_code == 422


def test_feedback_valid(monkeypatch):
    """Valid feedback payload should return 200 (DB call mocked)."""
    monkeypatch.setattr(main_module, "log_feedback", lambda **_: str(uuid.uuid4()))
    response = client.post(
        "/feedback",
        json={
            "image_id": str(uuid.uuid4()),
            "predicted_freshness": "Fresh",
            "correct_freshness": "Stale",
            "notes": "Soft spots on the bottom",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


# ── API key enforcement ───────────────────────────────────────────────────────

def test_protected_routes_require_api_key_when_configured(monkeypatch):
    monkeypatch.setattr(main_module, "API_KEY", "test-key-123")
    for method, path in [("get", "/history"), ("get", "/stats"), ("get", "/metrics"),
                         ("get", "/uncertain-predictions")]:
        assert getattr(client, method)(path).status_code == 401, path
        assert getattr(client, method)(path, headers={"X-API-Key": "wrong"}).status_code == 401, path
    r = client.post("/predict", files={"file": ("ok.png", _make_png_bytes(), "image/png")})
    assert r.status_code == 401


def test_api_key_accepted_and_health_stays_public(monkeypatch):
    monkeypatch.setattr(main_module, "API_KEY", "test-key-123")
    monkeypatch.setattr(main_module, "get_stats", lambda: {"total_predictions": 0})
    assert client.get("/stats", headers={"X-API-Key": "test-key-123"}).status_code == 200
    assert client.get("/health").status_code in (200, 503)
    assert client.get("/").status_code == 200
