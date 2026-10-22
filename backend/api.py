"""
Belt RUL Prediction API - v2
=============================
Clean FastAPI backend built around the belt_rul_training_data.csv dataset.

Endpoints:
  GET  /metadata              - Model info and feature stats
  GET  /datasets              - List available CSV datasets in data/ folder
  POST /upload-dataset        - Upload a new CSV dataset
  POST /train                 - Train (or retrain) on the selected dataset with chosen algorithm & belt type
  POST /predict               - Predict RUL with confidence from inspection readings
  POST /generate-data         - Run training_data_5_oct.py to generate fresh data
  GET  /schema                - Canonical schema the model is trained on
  POST /schema/propose        - LLM (Gemini) column-mapping proposal for a dataset
  POST /schema/approve        - Validate + save a human-approved column mapping
  GET  /belt-types            - Get all standard + custom belt type profiles
  POST /belt-types            - Create a new custom belt type
  GET  /models                - List registered models, metrics, and active champion
  GET  /models/{id}           - Detailed model metrics, hyperparameters, and trade-offs
  POST /models/benchmark      - Zero-training evaluation of an existing model on a dataset
  POST /models/compare        - Intelligent comparison between candidate and champion models
  POST /models/{id}/promote   - Promote a candidate model to active production champion
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
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
import traceback

# Add v2 backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema_mapper as sm
import model_registry as mr

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="Belt RUL Prediction API v2", lifespan=lifespan)

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
    mr._init_registry_if_needed()
    path = os.path.join(MODELS_DIR, "rul_model.pkl")
    if os.path.exists(path):
        MODEL_PAYLOAD = joblib.load(path)
        print(f"[startup] Loaded model from {path}")
    else:
        print("[startup] No pre-trained model found. Use /train after selecting a dataset.")


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
    datasets = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith(".csv"):
            continue
        full = os.path.join(DATA_DIR, f)
        size_mb = round(os.path.getsize(full) / (1024 * 1024), 2)
        try:
            preview = pd.read_csv(full, nrows=3)
            cols = list(preview.columns)
            datasets.append({
                "filename": f,
                "path": full,
                "size_mb": size_mb,
                "columns": cols,
                "sample_row": preview.iloc[0].to_dict() if len(preview) > 0 else {},
                "mapping_status": mapping_status(full, cols),
            })
        except Exception as e:
            datasets.append({
                "filename": f, "path": full, "size_mb": size_mb,
                "columns": [], "error": str(e),
                "mapping_status": "required",
            })
    return {"datasets": datasets}


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return JSONResponse(status_code=400, content={"error": "Only CSV files are supported."})

    safe_name = os.path.basename(file.filename)
    dest = os.path.join(DATA_DIR, safe_name)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    try:
        preview = pd.read_csv(io.BytesIO(content), nrows=3)
        cols = list(preview.columns)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid CSV file: {e}"})

    return {
        "success": True,
        "filename": safe_name,
        "path": dest,
        "columns": cols,
        "mapping_status": mapping_status(dest, cols),
    }


@app.post("/generate-data")
async def generate_data(request: Request):
    """Run training_data_5_oct.py as a subprocess to produce a fresh dataset."""
    body = await request.json()
    n_belts   = int(body.get("n_belts", 300))
    max_hours = int(body.get("max_hours", 20000))
    step      = int(body.get("step_hours", 100))
    seed      = int(body.get("seed", 42))

    if not os.path.exists(TRAINING_SCRIPT):
        return JSONResponse(status_code=500, content={
            "error": f"Script not found at {TRAINING_SCRIPT}"
        })

    out_name = f"synthetic_belt_data_{n_belts}b_{seed}s.csv"
    out_path = os.path.join(DATA_DIR, out_name)

    cmd = [
        sys.executable, TRAINING_SCRIPT,
        "--belts", str(n_belts),
        "--max-hours", str(max_hours),
        "--step", str(step),
        "--seed", str(seed),
        "--output", out_path,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
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
# Belt Types endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/belt-types")
async def get_belt_types():
    return mr.load_belt_types()


@app.post("/belt-types")
async def create_belt_type(request: Request):
    data = await request.json()
    try:
        created = mr.add_custom_belt_type(data)
        return {"success": True, "belt_type": created}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry & Comparison endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/models")
async def list_models(belt_type: Optional[str] = None):
    return mr.get_model_registry(belt_type)


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    details = mr.get_model_details(model_id)
    if not details:
        return JSONResponse(status_code=404, content={"error": f"Model {model_id} not found."})
    return details


@app.post("/models/benchmark")
async def benchmark_model(request: Request):
    """Zero-training evaluation of an existing model against a dataset."""
    body = await request.json()
    csv_path = resolve_dataset(body.get("csv_path", "belt_rul_training_data.csv"))
    model_id = body.get("model_id")

    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"error": f"Dataset not found: {csv_path}"})

    columns = list(pd.read_csv(csv_path, nrows=0).columns)
    saved = sm.load_mapping(MAPPINGS_DIR, csv_path, columns)
    if saved is None and not sm.is_legacy_canonical(columns):
        return JSONResponse(status_code=409, content={
            "error": "This dataset's columns must be mapped and approved before evaluation.",
            "mapping_required": True,
        })

    try:
        result = mr.benchmark_dataset_on_model(csv_path, model_id, saved["mapping"] if saved else None)
        return {"success": True, "benchmark": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})


@app.post("/models/compare")
async def compare_models_endpoint(request: Request):
    """Compare a candidate model with a baseline or active champion."""
    body = await request.json()
    candidate_id = body.get("candidate_id")
    candidate_meta = body.get("candidate_meta")
    baseline_id = body.get("baseline_id")

    if candidate_id and not candidate_meta:
        candidate_meta = mr.get_model_details(candidate_id)
    if not candidate_meta:
        return JSONResponse(status_code=400, content={"error": "candidate_id or candidate_meta required."})

    comparison = mr.compare_models(candidate_meta, baseline_id)
    return {"success": True, "comparison": comparison}


@app.post("/models/{model_id}/promote")
async def promote_model_endpoint(model_id: str):
    global MODEL_PAYLOAD
    try:
        updated = mr.promote_to_champion(model_id)
        load_model()
        return {"success": True, "promoted": updated}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Training endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/train")
async def train_model(request: Request):
    """Train on the specified CSV dataset with chosen algorithm and belt type."""
    global MODEL_PAYLOAD
    body    = await request.json()
    csv_path = resolve_dataset(body.get("csv_path", "belt_rul_training_data.csv"))
    algorithm = body.get("algorithm", "xgboost")
    belt_type = body.get("belt_type", "flat_conveyor")
    notes = body.get("notes", "")
    promote = body.get("promote", False)

    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404,
            content={"error": f"Dataset not found: {csv_path}"})

    columns = list(pd.read_csv(csv_path, nrows=0).columns)
    saved = sm.load_mapping(MAPPINGS_DIR, csv_path, columns)
    # Human-in-the-loop gate (user requirement)
    if saved is None and not sm.is_legacy_canonical(columns):
        return JSONResponse(status_code=409, content={
            "error": "This dataset's columns must be mapped and approved before training.",
            "mapping_required": True,
        })

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_model import train
        payload = train(
            csv_path=csv_path,
            output_dir=MODELS_DIR,
            mapping=saved["mapping"] if saved else None,
            algorithm=algorithm,
            belt_type=belt_type,
            notes=notes,
            promote_to_champion=promote
        )
        if payload.get("is_champion", False):
            MODEL_PAYLOAD = payload

        comparison = mr.compare_models(payload)

        return {
            "success": True,
            "model_id": payload["model_id"],
            "algorithm": payload["algorithm"],
            "belt_type": payload["belt_type"],
            "is_champion": payload["is_champion"],
            "metrics": payload["metrics"],
            "feature_importance": payload["feature_importance"],
            "feature_cols": payload["feature_cols"],
            "rul_source": payload["rul_source"],
            "training_rows": payload["training_rows"],
            "comparison": comparison,
            "tradeoffs": mr.get_algorithm_tradeoffs(algorithm)
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
    algo = MODEL_PAYLOAD.get("algorithm", "xgboost")
    return {
        "model_id": MODEL_PAYLOAD.get("model_id"),
        "algorithm": algo,
        "belt_type": MODEL_PAYLOAD.get("belt_type"),
        "feature_cols": MODEL_PAYLOAD["feature_cols"],
        "rul_source": MODEL_PAYLOAD.get("rul_source"),
        "metrics": MODEL_PAYLOAD["metrics"],
        "feature_stats": MODEL_PAYLOAD["feature_stats"],
        "feature_importance": MODEL_PAYLOAD["feature_importance"],
        "tradeoffs": mr.get_algorithm_tradeoffs(algo)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(request: Request):
    """
    Predict Remaining Useful Life (RUL) for a belt.
    Works with any trained model algorithm (XGBoost, Random Forest, Gradient Boosting).
    """
    if MODEL_PAYLOAD is None:
        return JSONResponse(status_code=503,
            content={"error": "Model not loaded. Train a model first via POST /train."})

    data = await request.json()
    belt_props  = data.get("belt_properties", {})
    inspections = data.get("inspections", [])

    if not inspections:
        return JSONResponse(status_code=400, content={"error": "No inspection readings provided."})

    model = MODEL_PAYLOAD.get("model") or MODEL_PAYLOAD.get("xgb_model")
    feat_cols = MODEL_PAYLOAD["feature_cols"]
    feat_stats = MODEL_PAYLOAD["feature_stats"]
    algo = MODEL_PAYLOAD.get("algorithm", "xgboost")
    algo_metrics = MODEL_PAYLOAD.get("metrics", {}).get(algo) or next(iter(MODEL_PAYLOAD.get("metrics", {}).values()), {})
    r2 = float(algo_metrics.get("r2_score", 0.90))

    thresholds = sm.degradation_thresholds(feat_cols)
    schema_fields = sm.load_canonical()["fields"]

    results = []
    for idx, insp in enumerate(inspections):
        row = {}
        for col in feat_cols:
            val = insp.get(col, belt_props.get(col))
            row[col] = float(val) if val is not None else feat_stats.get(col, {}).get("median", 0.0)
        hour = float(insp.get("hour", row.get("hour", 0.0)))

        X = pd.DataFrame([row])[feat_cols]
        pred_raw = float(model.predict(X)[0])
        pred_rul = max(pred_raw, 0.0)

        ratios = {}
        for col, thr in thresholds.items():
            val = insp.get(col)
            if val is not None:
                ratios[schema_fields[col]["label"]] = float(val) / thr

        if any(r >= 1.0 for r in ratios.values()):
            pred_rul = 0.0

        confidence = compute_confidence(idx + 1, r2)

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
        "model_metrics": algo_metrics,
        "algorithm": algo,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
