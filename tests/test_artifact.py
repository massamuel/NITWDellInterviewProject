"""Integration check against the trained artifact, including serialization."""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.app import create_app

@pytest.mark.skipif(not Path("artifacts/model.keras").exists(), reason="Train model first")
def test_real_model():
    with TestClient(create_app()) as client:
        response = client.post("/api/predict", json={"text": "I enjoy learning new things and reflecting on creative ideas."})
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["confidence"] <= 1
        assert len(data["scores"]) == 16
        assert abs(sum(score["probability"] for score in data["scores"]) - 1) < 1e-5
