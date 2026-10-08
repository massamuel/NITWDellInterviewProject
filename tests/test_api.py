import numpy as np
from fastapi.testclient import TestClient
from backend.app import create_app
from models.preprocessing import LABELS, clean_text

class FakePredictor:
    metadata = {"model_version": "test"}
    def predict(self, text):
        return np.array([.7] + [.3 / 15] * 15)


def test_prediction_contract():
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/api/predict", json={"text": "I enjoy creative work and new ideas"})
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == LABELS[0]
        assert len(data["scores"]) == 16
        assert abs(sum(s["probability"] for s in data["scores"]) - 1) < 1e-6
        assert client.get("/health/ready").status_code == 200
        assert "prediction_duration_seconds" in client.get("/metrics").text


def test_bad_input():
    with TestClient(create_app(FakePredictor())) as client:
        for text in ["", "   ", "https://example.com INTJ INFJ", "x" * 10001]:
            assert client.post("/api/predict", json={"text": text}).status_code == 422
        assert client.post("/api/predict", json={}).status_code == 422


def test_inference_failure():
    class Broken(FakePredictor):
        def predict(self, text):
            raise RuntimeError("private error")
    with TestClient(create_app(Broken())) as client:
        response = client.post("/api/predict", json={"text": "This is valid writing"})
        assert response.status_code == 503
        assert "private error" not in response.text


def test_cleaning():
    assert clean_text("INFJ ||| I like ideas https://example.com/test") == "i like ideas"
