import os
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from sklearn.metrics import classification_report, confusion_matrix

app = FastAPI(title = "HHAR API", version = "1.0")

app.add_middleware(CORSMiddleware, allow_origins = ["*"], allow_credentials = True, allow_methods = ["*"], allow_headers = ["*"],)

MODEL_PATH = "best_hhar.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"
DEMO_DATA_PATH = "demo_windows.npy" 

model, scaler, encoder = None, None, None

@app.on_event("startup")
def load_resources():
    global model, scaler, encoder
    if os.path.exists(MODEL_PATH): model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(SCALER_PATH): scaler = joblib.load(SCALER_PATH)
    if os.path.exists(ENCODER_PATH): encoder = joblib.load(ENCODER_PATH)

class WindowInput(BaseModel):
    data: List[List[float]]

class BatchEvalInput(BaseModel):
    x_data: List[List[List[float]]]  
    y_true: List[str]              

@app.post("/predict")
def predict_activity(payload: WindowInput):
    if not all([model, scaler, encoder]):
        raise HTTPException(status_code = 503, detail = "Model assets not loaded.")
    
    input_data = np.array(payload.data)
    if input_data.shape != (128, 12):
        raise HTTPException(status_code = 400, detail = "Shape must be (128, 12)")
    
    scaled_data = scaler.transform(input_data)
    model_input = np.expand_dims(scaled_data, axis = 0)
    
    predictions = model.predict(model_input)
    class_idx = np.argmax(predictions, axis = 1)[0]
    
    return {
        "activity": encoder.classes_[class_idx],
        "confidence": float(predictions[0][class_idx]),
        "all_probabilities": {encoder.classes_[i]: float(p) for i, p in enumerate(predictions[0])}
    }

@app.post("/evaluate")
def evaluate_model(payload: BatchEvalInput):
    if not all([model, scaler, encoder]):
        raise HTTPException(status_code = 503, detail = "Model assets not loaded.")
    
    X_raw = np.array(payload.x_data) 
    y_true_labels = np.array(payload.y_true)
    
    if len(X_raw.shape) != 3 or X_raw.shape[1:] != (128, 12):
        raise HTTPException(status_code = 400, detail = "x_data shape must be (num_samples, 128, 12)")
    
    num_samples = X_raw.shape[0]

    X_scaled = np.zeros_like(X_raw)
    for i in range(num_samples):
        X_scaled[i] = scaler.transform(X_raw[i])

    predictions = model.predict(X_scaled)
    pred_indices = np.argmax(predictions, axis = 1)
    y_pred_labels = encoder.inverse_transform(pred_indices)

    labels_order = list(encoder.classes_)
    
    report_dict = classification_report(y_true_labels,  y_pred_labels, labels = labels_order, output_dict = True, zero_division = 0)
    cm_array = confusion_matrix(y_true_labels, y_pred_labels, labels = labels_order)
    
    return {
        "classes": labels_order,
        "confusion_matrix": cm_array.tolist(),
        "classification_report": report_dict
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
