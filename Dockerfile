FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 KERAS_HOME=/tmp/keras
WORKDIR /app
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY backend ./backend
COPY models ./models

FROM base AS trainer
COPY data/raw/mbti_1.csv ./data/raw/mbti_1.csv
ENTRYPOINT ["python", "-m", "models.keras.train"]

FROM base AS api
COPY artifacts/model.keras artifacts/metadata.json ./artifacts/
RUN useradd --uid 10001 --create-home appuser
USER 10001
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "64"]
