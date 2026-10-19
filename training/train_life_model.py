"""
Model 2: Belt Failure-Life & Survival Analysis Prediction
=========================================================
Targets:
  - Duration: observed_time
  - Censoring Indicator: eventual_failure (1 = Failed, 0 = Censored)
  - Failure Hour: failure_hour (for uncensored actual failures)
Methodology:
  1. Survival AFT (Accelerated Failure Time) Model with right-censoring bounds to estimate survival probabilities & hazard.
  2. Gradient Boosting Failure Life Regressor for expected operational hours at failure and Remaining Useful Life (RUL).
Saves: models/life_model.pkl
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lifelines.utils import concordance_index
import xgboost as xgb

from life_utils import UnifiedLifeModel

def load_life_model(model_path="models/life_model.pkl"):
    data = joblib.load(model_path)
    if isinstance(data, dict):
        return UnifiedLifeModel(
            aft_booster=data["aft_booster"],
            life_regressor=data["life_regressor"],
            feature_names=data.get("feature_names", []),
            c_index=data.get("c_index", 0.0),
            mae=data.get("mae", 0.0),
            rmse=data.get("rmse", 0.0),
            r2=data.get("r2", 0.0)
        )
    return data

def train_life_model(csv_path="data/data.csv", output_dir="models", preprocessor_path="models/preprocessing.pkl"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)

    print("=" * 80)
    print("TRAINING MODEL 2: BELT FAILURE-LIFE & SURVIVAL PREDICTION")
    print(f"Source dataset: {csv_path} (READ-ONLY)")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    leakage_cols = [
        "eventual_failure", "censored", "failure_hour", "observed_time",
        "failure_mode", "failure_location", "belt_id", "manufacturing_batch"
    ]
    categorical_features = [
        "belt_model", "belt_type", "belt_material", "reinforcement_type",
        "manufacturer", "splice_type"
    ]
    numerical_features = [
        col for col in df.columns if col not in leakage_cols and col not in categorical_features
    ]
    all_features = categorical_features + numerical_features

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}.")
    preprocessor = joblib.load(preprocessor_path)

    X_encoded = preprocessor.transform(df[all_features])

    y_lower = df["observed_time"].values
    y_upper = np.where(df["eventual_failure"] == 1, df["observed_time"].values, np.inf)

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=df["eventual_failure"])

    X_train = X_encoded[train_idx]
    X_test = X_encoded[test_idx]

    print("\n--- Training Component A: Survival AFT Model (Handling Right-Censoring) ---")
    dtrain_aft = xgb.DMatrix(X_train, label_lower_bound=y_lower[train_idx], label_upper_bound=y_upper[train_idx])
    dtest_aft = xgb.DMatrix(X_test, label_lower_bound=y_lower[test_idx], label_upper_bound=y_upper[test_idx])

    aft_params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "logistic",
        "aft_loss_distribution_scale": 1.15,
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.08,
        "random_state": 42
    }

    aft_booster = xgb.train(aft_params, dtrain_aft, num_boost_round=80)
    aft_preds_test = aft_booster.predict(dtest_aft)

    test_durations = df["observed_time"].iloc[test_idx].values
    test_events = df["eventual_failure"].iloc[test_idx].values
    c_idx = float(concordance_index(test_durations, aft_preds_test, test_events))
    print(f"Survival Concordance Index (C-index) on Test Set: {c_idx:.4f}")

    print("\n--- Training Component B: Expected Failure Life Regressor ---")
    failed_mask = (df["eventual_failure"] == 1).values
    train_failed_idx = [i for i in train_idx if failed_mask[i]]
    test_failed_idx = [i for i in test_idx if failed_mask[i]]

    X_train_failed = X_encoded[train_failed_idx]
    y_train_failed = df["failure_hour"].iloc[train_failed_idx].values

    X_test_failed = X_encoded[test_failed_idx]
    y_test_failed = df["failure_hour"].iloc[test_failed_idx].values

    reg = HistGradientBoostingRegressor(max_iter=120, max_depth=8, learning_rate=0.08, random_state=42)
    reg.fit(X_train_failed, y_train_failed)

    life_preds_test = reg.predict(X_test_failed)
    mae = float(mean_absolute_error(y_test_failed, life_preds_test))
    rmse = float(np.sqrt(mean_squared_error(y_test_failed, life_preds_test)))
    r2 = float(r2_score(y_test_failed, life_preds_test))

    print(f"Mean Absolute Error (MAE):    {mae:,.2f} hours")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} hours")
    print(f"R-squared Score (R2):          {r2:.4f}")

    # Save as standard dictionary for universal pickle compatibility
    model_payload = {
        "aft_booster": aft_booster,
        "life_regressor": reg,
        "feature_names": all_features,
        "c_index": round(c_idx, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }

    save_path = os.path.join(output_dir, "life_model.pkl")
    joblib.dump(model_payload, save_path)
    print(f"\nUnified Life Model saved to: {save_path}")

    eval_summary = {
        "task": "Failure Life & Survival Prediction",
        "methodology": "Dual AFT Survival (Right-Censored) + Gradient Boosting Life Regressor",
        "concordance_index": round(c_idx, 4),
        "mae_hours": round(mae, 2),
        "rmse_hours": round(rmse, 2),
        "r2_score": round(r2, 4),
        "mean_actual_failure_hour": round(float(y_test_failed.mean()), 2),
        "mean_predicted_failure_hour": round(float(life_preds_test.mean()), 2)
    }

    with open("evaluation/life_prediction_metrics.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary

if __name__ == "__main__":
    train_life_model()
