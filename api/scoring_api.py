from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI()
MODEL_PATH = Path("models/rf.pkl")
FEATS_PATH = Path("models/features.json")
rf = joblib.load(MODEL_PATH)
feats = json.loads(FEATS_PATH.read_text())

class ScoreRequest(BaseModel):
    payment_ratio: float
    medicare_coverage_ratio: float
    financial_stress_index: float
    avg_charges_log: float
    state_avg_payment_ratio: float
    drg_diversity_index: float
    year: int
    hospital_size_category: str | None = None

@app.post("/score")
def score(req: ScoreRequest):
    d = pd.DataFrame([{**req.dict()}])
    for c in ["hospital_size_category_small","hospital_size_category_medium","hospital_size_category_large"]:
        d[c] = 0
    if req.hospital_size_category:
        d[f"hospital_size_category_{req.hospital_size_category}"] = 1
    X = d[feats]
    y_pred = rf.predict(X)[0]
    return {"pred_next_readmit_discharges": float(y_pred)}