"""
Belt RUL Prediction API - v2
=============================
Clean FastAPI backend built around the belt_rul_training_data.csv dataset.

Endpoints:
  GET  /metadata              - Model info and feature stats
  GET  /datasets              - List available CSV datasets in data/ folder
  POST /upload-dataset        - Upload a new CSV dataset
  POST /train                 - Train (or retrain) on the selected dataset
  POST /predict               - Predict RUL with confidence from inspection readings
  POST /generate-data         - Run training_data_5_oct.py to generate fresh data
"""

import os
import sys
import json
import shutil
import asyncio
import io
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
import traceback

# Add v2 backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Belt RUL Prediction API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # v2/
PARENT_DIR  = os.path.abspath(os.path.join(BASE_DIR, ".."))                    # project root
MODELS_DIR  = os.path.join(BASE_DIR, "models_v2") if os.path.exists(os.path.join(BASE_DIR, "models_v2")) else os.path.join(PARENT_DIR, "models_v2")
FRONTEND_DIR= os.path.join(BASE_DIR, "frontend")
DATA_DIR    = os.path.join(BASE_DIR, "data") if os.path.exists(os.path.join(BASE_DIR, "data")) else os.path.join(PARENT_DIR, "data")
TRAINING_SCRIPT = (
    os.path.join(BASE_DIR, "training", "training_data_5_oct.py")
    if os.path.exists(os.path.join(BASE_DIR, "training", "training_data_5_oct.py"))
    else os.path.join(PARENT_DIR, "training", "training_data_5_oct.py")
)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ── Serve frontend ─────────────────────────────────────────────────────────
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui")

# ── Global model state ────────────────────────────────────────────────────────
MODEL_PAYLOAD: Optional[Dict] = None

FEATURE_COLS = [
    "rubber_hardness_shore_a",
    "belt_thickness_mm",
    "tensile_strength_kn_m",
    "belt_width_mm",
    "belt_length_mm",
    "cover_thickness_top_mm",
    "pulley_alignment_mm",
    "load_pct",
    "belt_speed_mps",
    "temperature_c",
    "humidity_pct",
    "vibration_mms",
    "torque_nm",
    "belt_tension_n",
    "maintenance_count",
    "wear_pct",
    "elongation_pct",
    "crack_length_mm",
    "hour",
]

# Confidence grows with number of inspections performed
def compute_confidence(n_inspections: int, r2: float) -> float:
    """
    Confidence starts low (first reading = ~40%) and grows toward model R2
    as more inspection data points are submitted.
    Caps out at 97%.
    """
    base   = 40.0
    ceiling = min(r2 * 100, 97.0)
    growth  = 1.0 - float(np.exp(-0.35 * (n_inspections - 1)))
    conf    = base + (ceiling - base) * growth
    return float(round(min(conf, 97.0), 1))


def load_model():
    global MODEL_PAYLOAD
    path = os.path.join(MODELS_DIR, "rul_model.pkl")
    if os.path.exists(path):
        MODEL_PAYLOAD = joblib.load(path)
        print(f"[startup] Loaded model from {path}")
    else:
        print("[startup] No pre-trained model found. Use /train after selecting a dataset.")


@app.on_event("startup")
async def startup_event():
    load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/datasets")
async def list_datasets():
    """List all CSV datasets available in the data/ folder."""
    csvs = []
    if os.path.isdir(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.lower().endswith(".csv"):
                full = os.path.join(DATA_DIR, f)
                size_mb = round(os.path.getsize(full) / 1_048_576, 2)
                # Peek at row/column count
                try:
                    df_peek = pd.read_csv(full, nrows=3)
                    cols = list(df_peek.columns)
                except Exception:
                    cols = []
                csvs.append({
                    "filename": f,
                    "path": full,
                    "size_mb": size_mb,
                    "columns": cols,
                })
    return {"datasets": csvs, "data_dir": DATA_DIR}


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new CSV dataset to the data/ folder."""
    if not file.filename.lower().endswith(".csv"):
        return JSONResponse(status_code=400, content={"error": "Only CSV files accepted."})
    dest = os.path.join(DATA_DIR, file.filename)
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)
    # Quick validation
    try:
        df = pd.read_csv(io.BytesIO(contents))
        row_count = len(df)
        columns = list(df.columns)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not parse CSV: {e}"})
    return {
        "success": True,
        "filename": file.filename,
        "rows": row_count,
        "columns": columns,
        "path": dest,
    }


@app.post("/generate-data")
async def generate_data(request: Request):
    """
    Run training_data_5_oct.py to generate a fresh synthetic dataset.
    Body params (all optional): belts, max_hours, step, seed
    """
    body = await request.json()
    belts     = int(body.get("belts",     300))
    max_hours = int(body.get("max_hours", 20000))
    step      = int(body.get("step",      100))
    seed      = int(body.get("seed",      42))
    out_name  = body.get("output_name",   "belt_rul_training_data.csv")
    out_path  = os.path.join(DATA_DIR, out_name)

    if not os.path.exists(TRAINING_SCRIPT):
        return JSONResponse(status_code=404,
            content={"error": f"training_data_5_oct.py not found at {TRAINING_SCRIPT}"})

    cmd = (
        f'python "{TRAINING_SCRIPT}" '
        f'--belts {belts} --max-hours {max_hours} --step {step} '
        f'--seed {seed} --output "{out_path}"'
    )
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        return JSONResponse(status_code=500, content={
            "error": "Data generation failed.",
            "stderr": stderr.decode()
        })
    return {
        "success": True,
        "output_path": out_path,
        "log": stdout.decode(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/train")
async def train_model(request: Request):
    """Train on the specified CSV dataset."""
    global MODEL_PAYLOAD
    body    = await request.json()
    csv_path = body.get("csv_path", os.path.join(DATA_DIR, "belt_rul_training_data.csv"))

    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404,
            content={"error": f"Dataset not found: {csv_path}"})

    try:
        # Dynamically import to avoid circular import issues
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_model import train
        payload = train(csv_path, MODELS_DIR)
        MODEL_PAYLOAD = payload
        return {
            "success": True,
            "metrics": payload["metrics"],
            "feature_importance": payload["feature_importance"],
            "training_rows": int(len(pd.read_csv(csv_path))),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": traceback.format_exc()
        })


# ─────────────────────────────────────────────────────────────────────────────
# Metadata endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metadata")
async def get_metadata():
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    if MODEL_PAYLOAD is None:
        return {"error": "No model trained yet."}
    return {
        "feature_cols": MODEL_PAYLOAD["feature_cols"],
        "metrics": MODEL_PAYLOAD["metrics"],
        "feature_stats": MODEL_PAYLOAD["feature_stats"],
        "feature_importance": MODEL_PAYLOAD["feature_importance"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(request: Request):
    """
    Predict Remaining Useful Life (RUL) for a belt.

    Body:
    {
        "belt_properties": {
            "rubber_hardness_shore_a": 65.0,
            "belt_thickness_mm": 15.0,
            "tensile_strength_kn_m": 1000.0
        },
        "inspections": [
            {
                "hour": 0,
                "load_pct": 70.0,
                "belt_speed_mps": 3.0,
                "temperature_c": 30.0,
                "vibration_mms": 4.0,
                "wear_pct": 0.0,
                "elongation_pct": 0.0,
                "crack_length_mm": 0.0
            },
            ...  (more inspection readings)
        ]
    }

    Returns: RUL prediction for the LATEST inspection point, plus confidence
    and the full history of predictions.
    """
    if MODEL_PAYLOAD is None:
        return JSONResponse(status_code=503,
            content={"error": "Model not loaded. Train a model first via POST /train."})

    data = await request.json()
    belt_props  = data.get("belt_properties", {})
    inspections = data.get("inspections", [])

    if not inspections:
        return JSONResponse(status_code=400, content={"error": "No inspection readings provided."})

    xgb_model = MODEL_PAYLOAD["xgb_model"]
    rf_model  = MODEL_PAYLOAD["rf_model"]
    feat_cols = MODEL_PAYLOAD["feature_cols"]
    feat_stats = MODEL_PAYLOAD["feature_stats"]
    r2 = MODEL_PAYLOAD["metrics"]["ensemble"]["r2_score"]

    results = []
    for idx, insp in enumerate(inspections):
        row = {
            "rubber_hardness_shore_a": belt_props.get("rubber_hardness_shore_a", 65.0),
            "belt_thickness_mm":       belt_props.get("belt_thickness_mm", 15.0),
            "tensile_strength_kn_m":   belt_props.get("tensile_strength_kn_m", 1000.0),
            "belt_width_mm":           belt_props.get("belt_width_mm", 1000.0),
            "belt_length_mm":          belt_props.get("belt_length_mm", 50000.0),
            "cover_thickness_top_mm":  belt_props.get("cover_thickness_top_mm", 6.0),
            "pulley_alignment_mm":     belt_props.get("pulley_alignment_mm", 0.0),
            "load_pct":        insp.get("load_pct", 70.0),
            "belt_speed_mps":  insp.get("belt_speed_mps", 3.0),
            "temperature_c":   insp.get("temperature_c", 30.0),
            "humidity_pct":    insp.get("humidity_pct", 60.0),
            "vibration_mms":   insp.get("vibration_mms", 4.0),
            "torque_nm":       insp.get("torque_nm", 1000.0),
            "belt_tension_n":  insp.get("belt_tension_n", 840.0),
            "maintenance_count": insp.get("maintenance_count", 0),
            "wear_pct":        insp.get("wear_pct", 0.0),
            "elongation_pct":  insp.get("elongation_pct", 0.0),
            "crack_length_mm": insp.get("crack_length_mm", 0.0),
            "hour":            insp.get("hour", 0),
        }

        X = pd.DataFrame([row])[feat_cols]
        pred_xgb = float(xgb_model.predict(X)[0])
        pred_rf  = float(rf_model.predict(X)[0])
        pred_ens = (pred_xgb + pred_rf) / 2
        pred_rul = max(pred_ens, 0.0)

        # Physics override clamp: if any failure threshold is met, RUL is strictly 0
        if row["wear_pct"] >= 100.0 or row["elongation_pct"] >= 4.0 or row["crack_length_mm"] >= 50.0:
            pred_rul = 0.0

        # Confidence grows with more inspections
        confidence = compute_confidence(idx + 1, r2)

        # Health score: inverse of degradation ratio
        # wear threshold=100%, elongation=4%, crack=50mm
        wear_ratio  = row["wear_pct"] / 100.0
        elong_ratio = row["elongation_pct"] / 4.0
        crack_ratio = row["crack_length_mm"] / 50.0
        degradation = max(wear_ratio, elong_ratio, crack_ratio)
        health_score = round(max(1 - degradation, 0.0) * 100, 1)

        # Dominant failure driver
        drivers = {
            "Wear":       wear_ratio,
            "Elongation": elong_ratio,
            "Crack":      crack_ratio,
        }
        dominant = max(drivers, key=drivers.get)

        results.append({
            "inspection_index": int(idx + 1),
            "hour":             float(row["hour"]),
            "rul_hours":        float(round(pred_rul, 0)),
            "confidence_pct":   float(confidence),
            "health_score":     float(health_score),
            "dominant_driver":  str(dominant),
            "degradation_state": {
                "wear_pct":        float(row["wear_pct"]),
                "elongation_pct":  float(row["elongation_pct"]),
                "crack_length_mm": float(row["crack_length_mm"]),
            },
        })

    latest = results[-1]
    return {
        "belt_properties": belt_props,
        "total_inspections": len(inspections),
        "latest_prediction": latest,
        "prediction_history": results,
        "model_metrics": MODEL_PAYLOAD["metrics"]["ensemble"],
    }


if __name__ == "__main__":
    import uvicorn
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
