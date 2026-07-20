import os
import joblib
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sklearn.metrics import classification_report, confusion_matrix

app = FastAPI(title = "HHAR Activity Recognition API", version = "2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

MODEL_PATH = "best_hhar.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

model = None
scaler = None
encoder = None


@app.on_event("startup")
def load_resources():
    global model, scaler, encoder

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

    if os.path.exists(ENCODER_PATH):
        encoder = joblib.load(ENCODER_PATH)

    print("Resources Loaded")


class WindowInput(BaseModel):
    data: List[List[float]]


class BatchEvalInput(BaseModel):
    x_data: List[List[List[float]]]
    y_true: List[str]


class GenerateInput(BaseModel):
    activity: Optional[str] = "walking"


def create_demo_signal(activity: str):

    t = np.linspace(0, 4 * np.pi, 128)
    signal = np.zeros((128, 12))

    if activity == "walking":
        for i in range(12):
            signal[:, i] = ( np.sin(t + i * 0.2) + np.random.normal(0, 0.15, 128))

    elif activity == "stairsup":
        for i in range(12):
            signal[:, i] = (1.5 * np.sin(2 * t + i * 0.3) + np.random.normal(0, 0.25, 128))

    elif activity == "stairsdown":
        for i in range(12):
            signal[:, i] = (1.5 * np.sin(2 * t + i * 0.3) + 0.4 * np.random.randn(128))

    elif activity == "sitting":
        signal = np.random.normal(0, 0.02, (128, 12))
        signal[:, 2] += 9.8

    elif activity == "standing":
        signal = np.random.normal(0, 0.01, (128, 12))
        signal[:, 2] += 9.8

    elif activity == "biking":
        for i in range(12):
            signal[:, i] = (2 * np.sin(3 * t + i * 0.2) + np.random.normal(0, 0.2, 128))

    else:
        signal = np.random.normal(0, 1, (128, 12))

    return signal.tolist()


@app.get("/")
def root():
    return { "message": "HHAR Activity Recognition API" }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": encoder is not None
    }


@app.get("/classes")
def classes():
    if encoder is None:
        raise HTTPException(status_code = 503, detail = "Encoder not loaded.")

    return {"activities": encoder.classes_.tolist() }


@app.post("/generate")
def generate_demo(payload: GenerateInput):
    return { "activity": payload.activity, "data": create_demo_signal(payload.activity) }


@app.post("/predict")
def predict_activity(payload: WindowInput):
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code = 503, detail = "Model assets not loaded.")

    window = np.array(payload.data, dtype = np.float32)

    if window.shape != (128, 12):
        raise HTTPException(status_code = 400, detail = "Input must be shape (128,12)")

    scaled = scaler.transform(window)

    pred = model.predict(np.expand_dims(scaled, axis = 0), verbose = 0)[0]
    idx = np.argmax(pred)

    return {
        "activity": encoder.classes_[idx],
        "confidence": float(pred[idx]),
        "all_probabilities": {encoder.classes_[i]: float(pred[i])for i in range(len(pred))}
    }


@app.post("/evaluate")
def evaluate(payload: BatchEvalInput):
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code = 503, detail = "Model assets not loaded.")

    X = np.array(payload.x_data)
    if len(X.shape) != 3 or X.shape[1:] != (128, 12):
        raise HTTPException(status_code = 400, detail = "Expected (N,128,12)")

    y_true = np.array(payload.y_true)

    n = X.shape[0]

    X = scaler.transform(X.reshape(-1, 12)).reshape(n, 128, 12)

    pred = model.predict( X, verbose = 0)
    y_pred = encoder.inverse_transform(np.argmax(pred, axis = 1))

    labels = list(encoder.classes_)

    return {
        "classes": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels = labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels = labels, output_dict = True, zero_division = 0)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8000)
