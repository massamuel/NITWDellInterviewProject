"""Sequential, in-process latency check. Use k6 for concurrent network load."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from fastapi.testclient import TestClient
from backend.app import create_app

samples = []
inference = []
with TestClient(create_app()) as client:
    for _ in range(100):
        start = perf_counter()
        response = client.post("/api/predict", json={"text": "I enjoy exploring new ideas and understanding how systems work. I reflect before making decisions and enjoy independent creative projects."})
        response.raise_for_status()
        samples.append((perf_counter() - start) * 1000)
        inference.append(response.json()["latency_ms"])
result = {"mode": "sequential in-process TestClient; no network or concurrent load",
          "samples": len(samples), "request_p50_ms": float(np.percentile(samples, 50)),
          "request_p95_ms": float(np.percentile(samples, 95)),
          "inference_p95_ms": float(np.percentile(inference, 95))}
Path("docs/local-latency.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
