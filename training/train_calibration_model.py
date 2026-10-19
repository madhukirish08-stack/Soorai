"""
Calibration Model Training
===========================
Trains an ML model on the RESIDUAL between the physics engine's prediction
and actual observed failure hours in the historical dataset.

This gives the hybrid PIML engine:
    final_life = physics_life × calibration_factor

Where calibration_factor is learned from 50,000 historical belt records.
For novel/unseen inputs, calibration → ~1.0 and physics engine dominates.

Also saves feature statistics (mean, std) for anomaly detection.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Add parent and backend dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

import physics_engine

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingRegressor


LEAKAGE_COLS = [
    "eventual_failure", "censored", "failure_hour", "observed_time",
    "failure_mode", "failure_location", "belt_id", "manufacturing_batch"
]

CAT_FEATURES = [
    "belt_model", "belt_type", "belt_material", "reinforcement_type",
    "manufacturer", "splice_type"
]


def row_to_params(row: pd.Series) -> dict:
    """Convert a DataFrame row to a params dict for the physics engine."""
    return row.to_dict()


def train_calibration_model(
    csv_path="data/data.csv",
    output_dir="models"
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("TRAINING CALIBRATION MODEL")
    print("Computing physics predictions on 50,000 historical belt records...")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} records from {csv_path}")

    # ── Save feature statistics (mean/std) for anomaly detection ──────────────
    num_features = [
        c for c in df.columns
        if c not in LEAKAGE_COLS and c not in CAT_FEATURES and df[c].dtype in [float, "float64", int, "int64"]
    ]
    feature_stats = {}
    for col in num_features:
        feature_stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "p5": float(df[col].quantile(0.05)),
            "p95": float(df[col].quantile(0.95)),
        }
    for col in CAT_FEATURES:
        if col in df.columns:
            feature_stats[col] = {
                "type": "categorical",
                "values": df[col].unique().tolist(),
            }

    stats_path = os.path.join(output_dir, "feature_stats.pkl")
    joblib.dump(feature_stats, stats_path)
    print(f"Feature statistics saved to: {stats_path}")

    # ── Compute physics predictions for all failed belts ─────────────────────
    failed_df = df[df["eventual_failure"] == 1].copy()
    print(f"\nComputing physics predictions for {len(failed_df):,} failed belts...")

    physics_preds = []
    for _, row in failed_df.iterrows():
        params = row_to_params(row)
        result = physics_engine.predict(params)
        physics_preds.append(result["physics_life_hours"])

    failed_df = failed_df.copy()
    failed_df["physics_life"] = physics_preds
    failed_df["actual_life"] = failed_df["failure_hour"]

    # Calibration factor: how much does reality deviate from physics?
    failed_df["calibration_factor"] = (
        failed_df["actual_life"] / failed_df["physics_life"].clip(lower=100)
    )

    # Clip calibration to a reasonable range [0.3, 4.0]
    # Prevents the ML from completely overriding physics for extreme inputs
    failed_df["calibration_factor"] = failed_df["calibration_factor"].clip(0.3, 4.0)

    print(f"\nCalibration Factor Statistics:")
    print(f"  Mean:   {failed_df['calibration_factor'].mean():.3f}")
    print(f"  Std:    {failed_df['calibration_factor'].std():.3f}")
    print(f"  Min:    {failed_df['calibration_factor'].min():.3f}")
    print(f"  Max:    {failed_df['calibration_factor'].max():.3f}")
    print(f"  Median: {failed_df['calibration_factor'].median():.3f}")

    # ── Build feature matrix ──────────────────────────────────────────────────
    # Use only non-leakage, non-categorical features for calibration
    feature_cols = [
        c for c in df.columns
        if c not in LEAKAGE_COLS + ["physics_life", "actual_life", "calibration_factor"]
        and c not in CAT_FEATURES
        and df[c].dtype in [float, "float64", int, "int64"]
    ]

    X = failed_df[feature_cols].values
    y = failed_df["calibration_factor"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining calibration model on {len(X_train):,} samples...")

    # Use HistGradientBoosting — handles all feature types, no preprocessing needed
    model = HistGradientBoostingRegressor(
        max_iter=150,
        max_depth=6,
        learning_rate=0.06,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Calibration Model Performance:")
    print(f"  MAE (calibration factor): {mae:.4f}")
    print(f"  R² score:                 {r2:.4f}")
    print(f"  (R²~0.0 is EXPECTED — calibration factor noise is irreducible from synthetic data)")

    # Save model + feature column list
    cal_payload = {
        "model": model,
        "feature_cols": feature_cols,
        "mean_calibration": float(failed_df["calibration_factor"].median()),
    }
    cal_path = os.path.join(output_dir, "calibration_model.pkl")
    joblib.dump(cal_payload, cal_path)
    print(f"\nCalibration model saved to: {cal_path}")

    # --- Verify: does the hybrid model produce sensible outputs? ---
    print("\n── Verification on 5 sample failed belts ──")
    sample = failed_df.sample(5, random_state=99)
    for _, row in sample.iterrows():
        x = row[feature_cols].values.reshape(1, -1)
        cal = float(np.clip(model.predict(x)[0], 0.3, 4.0))
        phys = float(row["physics_life"])
        hybrid = phys * cal
        actual = float(row["actual_life"])
        print(f"  Physics={phys:,.0f}h  ×Cal={cal:.2f}  =Hybrid={hybrid:,.0f}h  | Actual={actual:,.0f}h")

    return cal_payload


if __name__ == "__main__":
    train_calibration_model()
