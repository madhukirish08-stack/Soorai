"""
Model Registry & Belt Type Intelligence
=======================================
Manages:
1. Belt type definitions (Flat, V-Belt, Ribbed, Timing, + Custom).
2. Model catalog with versioned runs, hyperparameters, and test metrics.
3. Champion promotion with backward-compatible export to models_v2/rul_model.pkl.
4. Model comparison, automated performance verdict, and algorithmic trade-offs.
5. Zero-training dataset benchmarking on existing models.
"""

import os
import json
import shutil
import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models_v2")
RUNS_DIR = os.path.join(MODELS_DIR, "runs")
REGISTRY_PATH = os.path.join(MODELS_DIR, "models_registry.json")
BELT_TYPES_PATH = os.path.join(os.path.dirname(__file__), "belt_types.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Belt Types Management
# ─────────────────────────────────────────────────────────────────────────────

def load_belt_types() -> Dict[str, Any]:
    """Load all configured belt types (defaults + custom)."""
    if os.path.exists(BELT_TYPES_PATH):
        try:
            with open(BELT_TYPES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"types": {}}


def add_custom_belt_type(belt_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add or update a custom belt type."""
    registry = load_belt_types()
    belt_id = belt_data.get("id") or belt_data.get("name", "").lower().replace(" ", "_").replace("/", "_")
    if not belt_id:
        raise ValueError("Belt type must have a valid name or id.")

    entry = {
        "id": belt_id,
        "name": belt_data.get("name", belt_id),
        "geometry": belt_data.get("geometry", "Custom Geometry"),
        "icon": belt_data.get("icon", "⚙️"),
        "description": belt_data.get("description", "Custom configured belt profile."),
        "nominal_hardness": float(belt_data.get("nominal_hardness", 65.0)),
        "nominal_thickness": float(belt_data.get("nominal_thickness", 15.0)),
        "nominal_tensile": float(belt_data.get("nominal_tensile", 1000.0)),
        "typical_failure_modes": belt_data.get("typical_failure_modes", ["Surface Wear", "Fatigue"]),
        "is_custom": True,
        "created_at": datetime.datetime.now().isoformat(),
    }
    registry["types"][belt_id] = entry

    with open(BELT_TYPES_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    return entry


# ─────────────────────────────────────────────────────────────────────────────
# Algorithmic Trade-offs Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHM_PROFILES = {
    "xgboost": {
        "name": "XGBoost Regressor",
        "category": "Gradient Boosted Decision Trees",
        "advantages": [
            "High predictive accuracy capturing complex non-linear wear & crack acceleration curves.",
            "Built-in L1/L2 regularization prevents overfitting on high-dimensional sensor data.",
            "Efficient handling of skewed telemetry distributions and missing values."
        ],
        "disadvantages": [
            "Sensitive to extreme out-of-distribution sensor noise without feature clipping.",
            "Higher hyperparameter sensitivity (tree depth, subsampling rates).",
            "Black-box tree ensemble requires SHAP/importance extraction for explainability."
        ],
        "best_for": "Heavy-duty industrial applications with sufficient historical data where peak accuracy is critical."
    },
    "random_forest": {
        "name": "Random Forest Regressor",
        "category": "Bagging Ensemble of Decision Trees",
        "advantages": [
            "High robustness against noisy or erratic field sensors (temperature spikes, vibration blips).",
            "Low risk of overfitting due to independent bootstrap aggregation.",
            "Minimal hyperparameter tuning required out-of-the-box."
        ],
        "disadvantages": [
            "Cannot extrapolate RUL beyond the maximum target values observed in training data.",
            "Larger model footprint and slightly slower inference latency on large tree counts.",
            "Slightly lower peak accuracy compared to fine-tuned gradient boosting."
        ],
        "best_for": "Environments with unpredictable operational noise, varying loads, or smaller training sets."
    },
    "gradient_boosting": {
        "name": "Gradient Boosting Regressor (Scikit-Learn)",
        "category": "Sequential Stage-wise Additive Model",
        "advantages": [
            "Smooth continuous regression fits that perform well on steady wear degradation trends.",
            "Lightweight dependencies without requiring external C++ bindings.",
            "Predictable convergence with robust shrinkage (learning rate) control."
        ],
        "disadvantages": [
            "Slower training times on large datasets compared to histogram-optimized XGBoost.",
            "Greedy sequential nature can propagate errors if early splits are misleading."
        ],
        "best_for": "Standard belt degradation with consistent physical deterioration rates."
    }
}


def get_algorithm_tradeoffs(algorithm: str) -> Dict[str, Any]:
    key = (algorithm or "xgboost").lower()
    return ALGORITHM_PROFILES.get(key, ALGORITHM_PROFILES["xgboost"])


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry Management
# ─────────────────────────────────────────────────────────────────────────────

def _init_registry_if_needed():
    """If registry file doesn't exist, bootstrap from existing models_v2/ files."""
    if not os.path.exists(REGISTRY_PATH):
        legacy_model = os.path.join(MODELS_DIR, "rul_model.pkl")
        legacy_meta = os.path.join(MODELS_DIR, "model_metadata.json")
        registry = {
            "active_champion_id": None,
            "models": {}
        }
        if os.path.exists(legacy_model) and os.path.exists(legacy_meta):
            try:
                with open(legacy_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                boot_id = "run_initial_champion"
                run_dir = os.path.join(RUNS_DIR, boot_id)
                os.makedirs(run_dir, exist_ok=True)
                shutil.copy2(legacy_model, os.path.join(run_dir, "rul_model.pkl"))
                shutil.copy2(legacy_meta, os.path.join(run_dir, "model_metadata.json"))

                registry["active_champion_id"] = boot_id
                registry["models"][boot_id] = {
                    "model_id": boot_id,
                    "created_at": datetime.datetime.now().isoformat(),
                    "belt_type": "flat_conveyor",
                    "algorithm": "xgboost",
                    "dataset": meta.get("dataset", "belt_rul_training_data.csv"),
                    "training_rows": meta.get("training_rows", 0),
                    "feature_cols": meta.get("feature_cols", []),
                    "metrics": meta.get("metrics", {}),
                    "is_champion": True,
                    "notes": "Bootstrapped from existing pre-trained model."
                }
            except Exception as e:
                print(f"[registry] Warning bootstrapping legacy model: {e}")

        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)


def get_model_registry(belt_type: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve all registered models, optionally filtered by belt_type."""
    _init_registry_if_needed()
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except Exception:
        registry = {"active_champion_id": None, "models": {}}

    models_list = list(registry.get("models", {}).values())
    if belt_type and belt_type != "all":
        models_list = [m for m in models_list if m.get("belt_type") == belt_type or m.get("belt_type") is None]

    # Sort newest first
    models_list.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return {
        "active_champion_id": registry.get("active_champion_id"),
        "models": models_list
    }


def get_model_details(model_id: str) -> Optional[Dict[str, Any]]:
    """Get metadata and feature importances for a specific model."""
    _init_registry_if_needed()
    meta_path = os.path.join(RUNS_DIR, model_id, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["model_id"] = model_id
        meta["tradeoffs"] = get_algorithm_tradeoffs(meta.get("algorithm", "xgboost"))
        return meta
    return None


def register_trained_model(
    model_id: str,
    payload: Dict[str, Any],
    metadata: Dict[str, Any],
    promote_to_champion: bool = False
) -> Dict[str, Any]:
    """Save versioned artifacts to models_v2/runs/{model_id}/ and update registry."""
    _init_registry_if_needed()
    run_dir = os.path.join(RUNS_DIR, model_id)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "rul_model.pkl")
    meta_path = os.path.join(run_dir, "model_metadata.json")

    joblib.dump(payload, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # First model registered automatically becomes champion if none exists
    if not registry.get("active_champion_id") or promote_to_champion:
        registry["active_champion_id"] = model_id
        is_champ = True
    else:
        is_champ = False

    entry = {
        "model_id": model_id,
        "created_at": metadata.get("created_at", datetime.datetime.now().isoformat()),
        "belt_type": metadata.get("belt_type", "flat_conveyor"),
        "algorithm": metadata.get("algorithm", "xgboost"),
        "dataset": metadata.get("dataset", ""),
        "training_rows": metadata.get("training_rows", 0),
        "feature_cols": metadata.get("feature_cols", []),
        "metrics": metadata.get("metrics", {}),
        "is_champion": is_champ,
        "notes": metadata.get("notes", "")
    }

    # Update championship flags
    if is_champ:
        for mid, m in registry["models"].items():
            m["is_champion"] = (mid == model_id)

    registry["models"][model_id] = entry

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    # If champion, copy to legacy paths for backwards compatibility
    if is_champ:
        shutil.copy2(model_path, os.path.join(MODELS_DIR, "rul_model.pkl"))
        shutil.copy2(meta_path, os.path.join(MODELS_DIR, "model_metadata.json"))

    return entry


def promote_to_champion(model_id: str) -> Dict[str, Any]:
    """Promote a registered model to the active champion model."""
    _init_registry_if_needed()
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if model_id not in registry.get("models", {}):
        raise ValueError(f"Model {model_id} not found in registry.")

    run_dir = os.path.join(RUNS_DIR, model_id)
    model_path = os.path.join(run_dir, "rul_model.pkl")
    meta_path = os.path.join(run_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found for {model_id}.")

    # Copy to production locations
    shutil.copy2(model_path, os.path.join(MODELS_DIR, "rul_model.pkl"))
    shutil.copy2(meta_path, os.path.join(MODELS_DIR, "model_metadata.json"))

    registry["active_champion_id"] = model_id
    for mid, m in registry["models"].items():
        m["is_champion"] = (mid == model_id)

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    return registry["models"][model_id]


# ─────────────────────────────────────────────────────────────────────────────
# Model Comparison & Intelligence Engine
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(
    candidate_meta: Dict[str, Any],
    baseline_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compares candidate model metrics against baseline/champion and generates:
    1. Metric deltas (MAE, RMSE, R2)
    2. Performance Verdict (OUTPERFORMED, COMPARABLE, DEGRADED)
    3. Algorithmic trade-offs (advantages & disadvantages)
    4. Intelligence summary & deployment recommendation
    """
    _init_registry_if_needed()
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    target_baseline_id = baseline_id or registry.get("active_champion_id")
    baseline_meta = None
    if target_baseline_id and target_baseline_id in registry.get("models", {}):
        baseline_meta = get_model_details(target_baseline_id)

    # Extract primary metrics
    cand_algo = candidate_meta.get("algorithm", "xgboost")
    cand_metrics = candidate_meta.get("metrics", {}).get(cand_algo, {})
    if not cand_metrics and candidate_meta.get("metrics"):
        # Fallback to first metric dictionary
        cand_metrics = next(iter(candidate_meta["metrics"].values()))

    c_r2 = cand_metrics.get("r2_score", 0.0)
    c_mae = cand_metrics.get("mae_hours", 0.0)
    c_rmse = cand_metrics.get("rmse_hours", 0.0)

    comparison = {
        "candidate": {
            "model_id": candidate_meta.get("model_id", "candidate"),
            "algorithm": cand_algo,
            "belt_type": candidate_meta.get("belt_type", "flat_conveyor"),
            "dataset": candidate_meta.get("dataset", ""),
            "r2_score": c_r2,
            "mae_hours": c_mae,
            "rmse_hours": c_rmse,
            "training_rows": candidate_meta.get("training_rows", 0),
            "tradeoffs": get_algorithm_tradeoffs(cand_algo)
        },
        "baseline": None,
        "deltas": None,
        "verdict": "CHAMPION_CANDIDATE",
        "verdict_label": "Champion Candidate",
        "verdict_color": "green",
        "recommendation": ""
    }

    if baseline_meta:
        base_algo = baseline_meta.get("algorithm", "xgboost")
        base_metrics = baseline_meta.get("metrics", {}).get(base_algo, {})
        if not base_metrics and baseline_meta.get("metrics"):
            base_metrics = next(iter(baseline_meta["metrics"].values()))

        b_r2 = base_metrics.get("r2_score", 0.0)
        b_mae = base_metrics.get("mae_hours", 0.0)
        b_rmse = base_metrics.get("rmse_hours", 0.0)

        delta_r2 = c_r2 - b_r2
        delta_mae = c_mae - b_mae
        delta_rmse = c_rmse - b_rmse

        comparison["baseline"] = {
            "model_id": target_baseline_id,
            "algorithm": base_algo,
            "belt_type": baseline_meta.get("belt_type", "flat_conveyor"),
            "dataset": baseline_meta.get("dataset", ""),
            "r2_score": b_r2,
            "mae_hours": b_mae,
            "rmse_hours": b_rmse,
            "training_rows": baseline_meta.get("training_rows", 0),
            "tradeoffs": get_algorithm_tradeoffs(base_algo)
        }

        comparison["deltas"] = {
            "r2_diff": round(float(delta_r2), 4),
            "mae_diff": round(float(delta_mae), 1),
            "rmse_diff": round(float(delta_rmse), 1),
            "r2_improved": delta_r2 > 0.005,
            "mae_improved": delta_mae < -10.0
        }

        # Determine Verdict
        if delta_r2 > 0.01 or (delta_r2 >= -0.005 and delta_mae < -20):
            comparison["verdict"] = "OUTPERFORMED"
            comparison["verdict_label"] = f"Outperformed Active Champion (+{delta_r2*100:.1f}% R²)"
            comparison["verdict_color"] = "green"
            comparison["recommendation"] = (
                f"Candidate model ({cand_algo.upper()}) demonstrates superior RUL accuracy with a "
                f"{abs(delta_mae):.0f}-hour reduction in MAE. Recommended to promote to active production."
            )
        elif abs(delta_r2) <= 0.015:
            comparison["verdict"] = "COMPARABLE"
            comparison["verdict_label"] = "Comparable Performance"
            comparison["verdict_color"] = "yellow"
            comparison["recommendation"] = (
                f"Candidate performance is on par with the existing champion. Evaluate algorithm trade-offs "
                f"({cand_algo} vs {base_algo}) before deciding whether to switch."
            )
        else:
            comparison["verdict"] = "DEGRADED"
            comparison["verdict_label"] = f"Underperformed Champion ({delta_r2*100:.1f}% R²)"
            comparison["verdict_color"] = "red"
            comparison["recommendation"] = (
                f"Candidate model exhibited lower generalization score than the active champion. "
                f"Keep current champion in production or adjust dataset preprocessing."
            )
    else:
        comparison["recommendation"] = (
            f"No existing champion model found for baseline comparison. "
            f"This {cand_algo.upper()} model will serve as the initial champion."
        )

    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# Zero-Training Dataset Benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_dataset_on_model(
    csv_path: str,
    model_id: Optional[str] = None,
    mapping: Optional[list] = None
) -> Dict[str, Any]:
    """
    Evaluates an existing model on a given dataset without retraining.
    Calculates R2, MAE, RMSE, and sample predictions.
    """
    from train_model import load_and_prepare

    _init_registry_if_needed()
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    target_id = model_id or registry.get("active_champion_id")
    if not target_id:
        raise ValueError("No model available to benchmark against.")

    run_dir = os.path.join(RUNS_DIR, target_id)
    model_path = os.path.join(run_dir, "rul_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "rul_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")

    payload = joblib.load(model_path)
    model = payload.get("xgb_model") or payload.get("model")
    feature_cols = payload.get("feature_cols", [])

    X, y, df, rul_info = load_and_prepare(csv_path, mapping)

    # Ensure all required features are present
    missing = [c for c in feature_cols if c not in X.columns]
    for c in missing:
        X[c] = payload.get("feature_stats", {}).get(c, {}).get("median", 0.0)

    X_eval = X[feature_cols]
    y_pred = model.predict(X_eval)

    mae = float(mean_absolute_error(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2 = float(r2_score(y, y_pred))

    samples = []
    sample_indices = np.linspace(0, len(y) - 1, min(10, len(y)), dtype=int)
    for idx in sample_indices:
        samples.append({
            "actual_rul_hours": round(float(y.iloc[idx]), 1),
            "predicted_rul_hours": round(float(y_pred[idx]), 1),
            "error_hours": round(float(abs(y.iloc[idx] - y_pred[idx])), 1)
        })

    return {
        "model_id": target_id,
        "algorithm": payload.get("algorithm", "xgboost"),
        "dataset": os.path.basename(csv_path),
        "rows_evaluated": int(len(X_eval)),
        "metrics": {
            "r2_score": round(r2, 4),
            "mae_hours": round(mae, 1),
            "rmse_hours": round(rmse, 1)
        },
        "samples": samples,
        "tradeoffs": get_algorithm_tradeoffs(payload.get("algorithm", "xgboost"))
    }
