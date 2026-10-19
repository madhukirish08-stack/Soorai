"""
Belt RUL Model Trainer
======================
Trains an XGBoost regressor on belt_rul_training_data.csv (produced by training_data_5_oct.py).

Features  : rubber_hardness_shore_a, belt_thickness_mm, tensile_strength_kn_m,
            load_pct, belt_speed_mps, temperature_c, vibration_mms,
            wear_pct, elongation_pct, crack_length_mm, hour
Target    : rul_hours  (Remaining Useful Life in hours)

Usage:
    python train_model.py --csv path/to/belt_rul_training_data.csv
"""

import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

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
TARGET_COL = "rul_hours"


def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Keep ALL rows — censored rows still have valid rul_hours
    # (they represent belts that survived to max_hours with known time remaining)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y, df


def compute_feature_stats(X: pd.DataFrame) -> dict:
    stats = {}
    for col in X.columns:
        stats[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "median": float(X[col].median()),
            "std": float(X[col].std()),
            "p25": float(X[col].quantile(0.25)),
            "p75": float(X[col].quantile(0.75)),
        }
    return stats


def train(csv_path: str, output_dir: str = "models_v2"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {csv_path}")
    X, y, df = load_and_prepare(csv_path)
    print(f"Training samples: {len(X):,}  |  RUL range: {y.min():.0f} - {y.max():.0f} hrs")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # XGBoost
    print("\nTraining XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    # Random Forest
    print("\nTraining Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Evaluate ensemble
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_rf  = rf_model.predict(X_test)
    y_pred_ens = (y_pred_xgb + y_pred_rf) / 2

    def get_metrics(y_true, y_pred, label):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = r2_score(y_true, y_pred)
        print(f"  [{label}]  MAE={mae:,.0f} hrs  RMSE={rmse:,.0f} hrs  R2={r2:.4f}")
        return {"mae_hours": round(float(mae), 1), "rmse_hours": round(float(rmse), 1), "r2_score": round(float(r2), 4)}

    print("\n-- Test-set Metrics --")
    xgb_metrics = get_metrics(y_test, y_pred_xgb, "XGBoost ")
    rf_metrics  = get_metrics(y_test, y_pred_rf,  "RF      ")
    ens_metrics = get_metrics(y_test, y_pred_ens, "Ensemble")

    feat_imp = dict(zip(FEATURE_COLS, [float(x) for x in xgb_model.feature_importances_]))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    feature_stats = compute_feature_stats(X)

    payload = {
        "xgb_model": xgb_model,
        "rf_model": rf_model,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "feature_stats": feature_stats,
        "metrics": {"xgboost": xgb_metrics, "random_forest": rf_metrics, "ensemble": ens_metrics},
        "feature_importance": feat_imp,
        "y_max": float(y.max()),
        "y_mean": float(y.mean()),
    }
    model_path = os.path.join(output_dir, "rul_model.pkl")
    joblib.dump(payload, model_path)

    metadata = {
        "training_rows": int(len(X)),
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "feature_stats": feature_stats,
        "feature_importance": feat_imp,
        "metrics": payload["metrics"],
        "y_max": float(y.max()),
        "y_mean": float(y.mean()),
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] Model saved   : {model_path}")
    print(f"[OK] Metadata saved: {meta_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Belt RUL Model")
    parser.add_argument("--csv",    default="../data/belt_rul_training_data.csv")
    parser.add_argument("--output", default="../models_v2")
    args = parser.parse_args()
    train(args.csv, args.output)
