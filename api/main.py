"""Heart Disease Risk API (FastAPI).

Endpoints:
  GET  /health   -> is the service up and the model loaded
  POST /predict  -> patient data (JSON) -> prediction + confidence
  GET  /metrics  -> Prometheus metrics (used for monitoring in Task 8)
"""
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "heart_model.joblib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-api")

model = joblib.load(MODEL_PATH)
logger.info("Model loaded from %s", MODEL_PATH)

app = FastAPI(title="Heart Disease Risk API", version="1.0",
              description="Predicts heart disease risk from 13 clinical features (UCI Cleveland data).")

REQUESTS = Counter("api_requests_total", "Number of API requests", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Time taken to answer a request", ["endpoint"])
PREDICTIONS = Counter("api_predictions_total", "Number of predictions by class", ["prediction"])


class Patient(BaseModel):
    """The 13 patient features. Impossible values are rejected with HTTP 422."""
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)")
    ca: int = Field(..., ge=0, le=3, description="Major vessels coloured by fluoroscopy (0-3)")
    thal: int = Field(..., ge=3, le=7, description="Thallium test: 3 normal, 6 fixed, 7 reversible defect")

    model_config = {"json_schema_extra": {"example": {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 289, "fbs": 0, "restecg": 2,
        "thalach": 122, "exang": 1, "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7}}}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request and record it in the monitoring metrics."""
    start = time.perf_counter()
    response = await call_next(request)
    seconds = time.perf_counter() - start
    path = request.url.path
    if path != "/metrics":
        REQUESTS.labels(path, str(response.status_code)).inc()
        LATENCY.labels(path).observe(seconds)
    logger.info("%s %s -> %s in %.1f ms", request.method, path, response.status_code, seconds * 1000)
    return response


@app.get("/")
def home():
    return {"message": "Heart Disease Risk API. Open /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: Patient):
    X = pd.DataFrame([patient.model_dump()])
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= 0.5)
    confidence = probability if prediction == 1 else 1 - probability
    PREDICTIONS.labels(str(prediction)).inc()
    logger.info("prediction=%d probability=%.3f", prediction, probability)
    return {
        "prediction": prediction,
        "label": "heart disease" if prediction == 1 else "no heart disease",
        "probability_of_disease": round(probability, 4),
        "confidence": round(confidence, 4),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)