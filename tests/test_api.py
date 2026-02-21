from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to FreshTrack AI API"}

def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422 # Unprocessable Entity because file is missing
