import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from backend.text import LABELS, clean_text

REQUESTS = Counter("prediction_requests_total", "Prediction requests", ["status"])
LATENCY = Histogram("prediction_duration_seconds", "Inference duration",
                    buckets=(.01, .025, .05, .1, .25, .5, 1, 2, 5))

class Predictor:
    def __init__(self, directory):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        self.tf = tf
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata["labels"] != LABELS:
            raise ValueError("Artifact labels differ from service labels")
        self.model = tf.keras.models.load_model(directory / "model.keras")
        self.lock = threading.Lock()
        self.predict("A warmup sentence about learning and building.")

    def predict(self, text):
        with self.lock:
            return self.model(self.tf.constant([text]), training=False).numpy()[0]

class PredictionInput(BaseModel):
    text: str = Field(min_length=1, max_length=10000)

class Score(BaseModel):
    label: str
    probability: float

class PredictionOutput(BaseModel):
    label: str
    confidence: float
    scores: list[Score]
    model_version: str
    latency_ms: float


def create_app(predictor=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = predictor or Predictor(Path(os.getenv("MODEL_DIR", "artifacts")))
        yield
    app = FastAPI(title="NITW Dell · Text classifier", lifespan=lifespan)
    origins = [s.strip() for s in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if s.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["POST", "GET"],
                       allow_headers=["Content-Type"])

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        if not getattr(app.state, "predictor", None):
            raise HTTPException(503, "Model not ready")
        return {"status": "ready", "model_version": app.state.predictor.metadata["model_version"]}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/api/predict", response_model=PredictionOutput)
    def predict(payload: PredictionInput):
        text = clean_text(payload.text)
        if len(text.split()) < 3:
            REQUESTS.labels("invalid").inc()
            raise HTTPException(422, "Enter at least three words after removing links and MBTI labels.")
        start = perf_counter()
        try:
            probabilities = app.state.predictor.predict(text)
        except Exception:
            REQUESTS.labels("error").inc()
            raise HTTPException(503, "Prediction temporarily unavailable") from None
        elapsed = perf_counter() - start
        LATENCY.observe(elapsed)
        REQUESTS.labels("success").inc()
        order = np.argsort(probabilities)[::-1]
        return {"label": LABELS[order[0]], "confidence": float(probabilities[order[0]]),
                "scores": [{"label": LABELS[i], "probability": float(probabilities[i])} for i in order],
                "model_version": app.state.predictor.metadata["model_version"], "latency_ms": elapsed * 1000}
    return app

app = create_app()
