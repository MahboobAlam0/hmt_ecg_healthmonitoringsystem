import os
import json
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from models.hmt_ecgnet import HMT_ECGNet
from config import N_LEADS
from transforms import preprocess_signal


# Setup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIAG_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

with open("artifacts/multilabel_thresholds.json") as f:
    MULTI_THRESHOLDS = json.load(f)

MI_BINARY_THRESHOLD = 0.05

app = FastAPI(title="ECG Diagnostic API")


# Find latest checkpoint automatically

def find_ckpt():
    for root, _, files in os.walk("artifacts"):
        if "multilabel_best.pth" in files:
            return os.path.join(root, "multilabel_best.pth")
    raise FileNotFoundError("multilabel_best.pth not found in artifacts/")


# Load model ONCE at startup

def load_model():
    ckpt_path = find_ckpt()

    model = HMT_ECGNet(num_classes=5, num_leads=N_LEADS).to(DEVICE)

    # IMPORTANT FIX for PyTorch 2.6
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


model = load_model()


# ECG Request Schema (real device style)

class ECGRequest(BaseModel):
    lead1: List[float]
    lead2: List[float]
    lead3: List[float]
    lead4: List[float]
    lead5: List[float]
    lead6: List[float]
    lead7: List[float]
    lead8: List[float]
    lead9: List[float]
    lead10: List[float]
    lead11: List[float]
    lead12: List[float]


# Prediction Endpoint

@app.post("/predict")
def predict_ecg(req: ECGRequest):
    try:
        # Combine leads into array
        raw = np.array([
            req.lead1, req.lead2, req.lead3, req.lead4,
            req.lead5, req.lead6, req.lead7, req.lead8,
            req.lead9, req.lead10, req.lead11, req.lead12
        ])

        # Auto-select correct 10s window (like real systems)
        fs = 500
        window = 10 * fs

        if raw.shape[1] > window:
            start = raw.shape[1] // 2 - window // 2
            raw = raw[:, start:start + window]

        # Preprocess exactly like training
        ecg = preprocess_signal(raw)

        x = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Model inference
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy()[0]

        result = {}
        predicted = []

        for cls, p in zip(DIAG_CLASSES, probs):
            thr = MULTI_THRESHOLDS[cls]
            result[cls] = float(p)
            if p >= thr:
                predicted.append(cls)

        mi_prob = float(probs[1])

        return {
            "probabilities": result,
            "predicted_classes": predicted,
            "mi_probability": mi_prob,
            "mi_risk": mi_prob >= MI_BINARY_THRESHOLD,
        }

    except Exception as e:
        return {"error": str(e)}
