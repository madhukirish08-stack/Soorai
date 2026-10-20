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
  GET  /schema                - Canonical schema the model is trained on
  POST /schema/propose        - LLM (Gemini) column-mapping proposal for a dataset
  POST /schema/approve        - Validate + save a human-approved column mapping
"""

import os
import sys
import json
import asyncio
import io
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
import traceback

# Add v2 backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema_mapper as sm

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

MAPPINGS_DIR = os.path.join(BASE_DIR, "mappings")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ── Serve frontend ─────────────────────────────────────────────────────────
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui")

# ── Global model state ────────────────────────────────────────────────────────
MODEL_PAYLOAD: Optional[Dict] = None

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

def resolve_dataset(csv_path: str) -> str:
    """Datasets are always resolved inside DATA_DIR (blocks path traversal via csv_path)."""
    return os.path.join(DATA_DIR, os.path.basename(csv_path or ""))


def mapping_status(filename: str, columns) -> str:
    if sm.load_mapping(MAPPINGS_DIR, filename, columns):
        return "approved"
    if sm.is_legacy_canonical(columns):
        return "not_needed"
    return "required"


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
                    "mapping_status": mapping_status(f, cols),
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
    csv_path = resolve_dataset(body.get("csv_path", "belt_rul_training_data.csv"))

    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404,
            content={"error": f"Dataset not found: {csv_path}"})

    columns = list(pd.read_csv(csv_path, nrows=0).columns)
    saved = sm.load_mapping(MAPPINGS_DIR, csv_path, columns)
    # Human-in-the-loop gate (user requirement): a dataset whose columns are not already
    # canonical must have a human-APPROVED mapping. Do not relax this to auto-apply LLM output.
    if saved is None and not sm.is_legacy_canonical(columns):
        return JSONResponse(status_code=409, content={
            "error": "This dataset's columns must be mapped and approved before training.",
            "mapping_required": True,
        })

    try:
        # Dynamically import to avoid circular import issues
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_model import train
        payload = train(csv_path, MODELS_DIR, saved["mapping"] if saved else None)
        MODEL_PAYLOAD = payload
        return {
            "success": True,
            "metrics": payload["metrics"],
            "feature_importance": payload["feature_importance"],
            "feature_cols": payload["feature_cols"],
            "rul_source": payload["rul_source"],
            "training_rows": payload["training_rows"],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": traceback.format_exc()
        })


# ─────────────────────────────────────────────────────────────────────────────
# Schema mapping endpoints (LLM proposes, human approves)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/schema")
async def get_schema():
    return sm.load_canonical()


@app.post("/schema/propose")
def propose_schema(body: Dict[str, Any]):
    """Sync def -> FastAPI runs it in a worker thread, so the Gemini call doesn't block the loop."""
    csv_path = resolve_dataset(body.get("csv_path"))
    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"error": f"Dataset not found: {csv_path}"})
    try:
        raw = pd.read_csv(csv_path)
        saved = sm.load_mapping(MAPPINGS_DIR, csv_path, raw.columns)
        if saved:
            proposal = {"source": "approved", "mapping": saved["mapping"],
                        "profile": sm.profile_dataframe(raw), "warnings": []}
        else:
            proposal = sm.propose_mapping(raw, MAPPINGS_DIR)
        proposal["already_approved"] = saved is not None
        proposal["canonical_fields"] = sm.load_canonical()["fields"]
        return proposal
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})


@app.post("/schema/approve")
def approve_schema(body: Dict[str, Any]):
    csv_path = resolve_dataset(body.get("csv_path"))
    mapping  = body.get("mapping") or []
    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"error": f"Dataset not found: {csv_path}"})
    raw = pd.read_csv(csv_path)
    errors, warnings, rul_info = sm.validate_mapping(raw, mapping)
    if errors:
        return JSONResponse(status_code=422, content={"saved": False, "errors": errors, "warnings": warnings})
    if body.get("dry_run"):
        # UI validates first so the reviewer sees warnings BEFORE anything is saved
        return {"saved": False, "errors": [], "warnings": warnings, "rul_source": rul_info}
    sm.save_mapping(MAPPINGS_DIR, csv_path, raw.columns, mapping, body.get("proposal_source", "manual"))
    return {"saved": True, "errors": [], "warnings": warnings, "rul_source": rul_info}


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
        "rul_source": MODEL_PAYLOAD.get("rul_source"),
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
    feat_cols = MODEL_PAYLOAD["feature_cols"]
    feat_stats = MODEL_PAYLOAD["feature_stats"]
    r2 = MODEL_PAYLOAD["metrics"]["xgboost"]["r2_score"]

    # Health/failure thresholds only for degradation measures the model was trained on
    thresholds = sm.degradation_thresholds(feat_cols)
    schema_fields = sm.load_canonical()["fields"]

    results = []
    for idx, insp in enumerate(inspections):
        # Feature vector follows the TRAINED model's feature list (dataset-dependent).
        # Missing inputs fall back to the training median instead of hard-coded defaults,
        # so models trained on customer data with different columns still predict.
        row = {}
        for col in feat_cols:
            val = insp.get(col, belt_props.get(col))
            row[col] = float(val) if val is not None else feat_stats.get(col, {}).get("median", 0.0)
        hour = float(insp.get("hour", row.get("hour", 0.0)))

        X = pd.DataFrame([row])[feat_cols]
        pred_xgb = float(xgb_model.predict(X)[0])
        pred_rul = max(pred_xgb, 0.0)

        # Degradation ratios use only values actually measured in this inspection
        ratios = {}
        for col, thr in thresholds.items():
            val = insp.get(col)
            if val is not None:
                ratios[schema_fields[col]["label"]] = float(val) / thr

        # Physics override clamp: if any failure threshold is met, RUL is strictly 0
        if any(r >= 1.0 for r in ratios.values()):
            pred_rul = 0.0

        # Confidence grows with more inspections
        confidence = compute_confidence(idx + 1, r2)

        # Health score: inverse of the worst degradation ratio (None if nothing measured)
        if ratios:
            health_score = round(max(1 - max(ratios.values()), 0.0) * 100, 1)
            dominant = max(ratios, key=ratios.get)
        else:
            health_score, dominant = None, "N/A"

        results.append({
            "inspection_index": int(idx + 1),
            "hour":             hour,
            "rul_hours":        float(round(pred_rul, 0)),
            "confidence_pct":   float(confidence),
            "health_score":     health_score,
            "dominant_driver":  str(dominant),
            "degradation_state": {
                c: (float(insp[c]) if c in thresholds and insp.get(c) is not None else None)
                for c in ("wear_pct", "elongation_pct", "crack_length_mm")
            },
        })

    latest = results[-1]
    return {
        "belt_properties": belt_props,
        "total_inspections": len(inspections),
        "latest_prediction": latest,
        "prediction_history": results,
        "model_metrics": MODEL_PAYLOAD["metrics"]["xgboost"],
    }


if __name__ == "__main__":
    import uvicorn
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
