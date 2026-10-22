"""
Belt RUL Model Trainer
======================
Trains an XGBoost, Random Forest, or Gradient Boosting regressor on any belt dataset.
Tracks trained runs, model metrics, hyperparameters, and algorithmic trade-offs
in the model registry (models_v2/models_registry.json).

Usage:
    python train_model.py --csv path/to/data.csv [--mapping path/to/data.csv.mapping.json] [--algorithm xgboost] [--belt-type flat_conveyor]
"""

import argparse
import os
import sys
import json
import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema_mapper as sm
import model_registry as mr

TARGET_COL = "rul_hours"
# Columns with less coverage than this are dropped instead of imputed.
MIN_FEATURE_COVERAGE = 0.5


def select_feature_cols(df: pd.DataFrame, schema: dict) -> list:
    """Canonical features in schema order, then extra customer features."""
    canonical = sm.fields_by_role(schema, "feature", "degradation", "time")
    extras = sorted(c for c in df.columns if c.startswith(sm.EXTRA_PREFIX))
    return [
        c for c in canonical + extras
        if c in df.columns and df[c].notna().mean() >= MIN_FEATURE_COVERAGE
    ]


def load_and_prepare(csv_path: str, mapping: list = None) -> tuple:
    raw = pd.read_csv(csv_path)
    schema = sm.load_canonical()

    if mapping:
        df, rul_info = sm.apply_mapping(raw, mapping)
    else:
        df = raw.copy()
        if TARGET_COL in df.columns:
            rul_info = {"source": "measured", "message": "Using measured rul_hours column from CSV"}
        elif "failure_hour" in df.columns and "hour" in df.columns:
            df[TARGET_COL] = (df["failure_hour"] - df["hour"]).clip(lower=0)
            rul_info = {"source": "derived_exact", "message": "Computed RUL as failure_hour - hour"}
        else:
            df, rul_info = sm.derive_rul(df)

    feature_cols = select_feature_cols(df, schema)
    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET_COL]
    return X, y, df, rul_info


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


def train(
    csv_path: str,
    output_dir: str = "models_v2",
    mapping: list = None,
    algorithm: str = "xgboost",
    belt_type: str = "flat_conveyor",
    notes: str = "",
    promote_to_champion: bool = False
):
    os.makedirs(output_dir, exist_ok=True)

    algorithm = (algorithm or "xgboost").lower()
    print(f"Loading data from: {csv_path}")
    print(f"Algorithm: {algorithm.upper()} | Belt Type: {belt_type}")
    X, y, df, rul_info = load_and_prepare(csv_path, mapping)
    feature_cols = list(X.columns)
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"RUL source: {rul_info['message']}")
    print(f"Training samples: {len(X):,}  |  RUL range: {y.min():.0f} - {y.max():.0f} hrs")

    # Split by BELT, not by row to avoid data leakage
    if "belt_id" in df.columns and df["belt_id"].nunique() >= 5:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=df["belt_id"]))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        split = "by_belt"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        split = "by_row"
    print(f"Test split: {split} ({len(X_test):,} test rows)")

    # Model instantiation based on selected algorithm
    if algorithm == "random_forest":
        print("\nTraining Random Forest Regressor...")
        model = RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    elif algorithm == "gradient_boosting":
        print("\nTraining Gradient Boosting Regressor...")
        model = GradientBoostingRegressor(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            random_state=42
        )
        model.fit(X_train, y_train)
    else:
        # Default: XGBoost
        algorithm = "xgboost"
        print("\nTraining XGBoost Regressor...")
        model = xgb.XGBRegressor(
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
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)

    def get_metrics(y_true, y_pred, label):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = r2_score(y_true, y_pred)
        print(f"  [{label}]  MAE={mae:,.0f} hrs  RMSE={rmse:,.0f} hrs  R2={r2:.4f}")
        return {"mae_hours": round(float(mae), 1), "rmse_hours": round(float(rmse), 1), "r2_score": round(float(r2), 4)}

    print(f"\n-- Test-set Metrics ({algorithm}) --")
    metrics_summary = get_metrics(y_test, y_pred, algorithm)

    if hasattr(model, "feature_importances_"):
        feat_imp = dict(zip(feature_cols, [float(x) for x in model.feature_importances_]))
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    else:
        feat_imp = {}

    feature_stats = compute_feature_stats(X)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"run_{timestamp_str}_{algorithm}"

    payload = {
        "model_id": model_id,
        "algorithm": algorithm,
        "belt_type": belt_type,
        "model": model,
        "xgb_model": model,  # Backwards compatibility with existing api.py predict endpoint
        "training_rows": int(len(X)),
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "feature_stats": feature_stats,
        "rul_source": rul_info,
        "mapping": mapping,
        "test_split": split,
        "metrics": {algorithm: metrics_summary},
        "feature_importance": feat_imp,
        "y_max": float(y.max()),
        "y_mean": float(y.mean()),
        "created_at": datetime.datetime.now().isoformat(),
    }

    metadata = {
        "model_id": model_id,
        "algorithm": algorithm,
        "belt_type": belt_type,
        "training_rows": int(len(X)),
        "dataset": os.path.basename(csv_path),
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "feature_stats": feature_stats,
        "rul_source": rul_info,
        "mapping": mapping,
        "test_split": split,
        "feature_importance": feat_imp,
        "metrics": payload["metrics"],
        "y_max": float(y.max()),
        "y_mean": float(y.mean()),
        "created_at": payload["created_at"],
        "notes": notes,
    }

    # Register in registry & save versioned run
    reg_entry = mr.register_trained_model(
        model_id=model_id,
        payload=payload,
        metadata=metadata,
        promote_to_champion=promote_to_champion
    )
    payload["is_champion"] = reg_entry.get("is_champion", False)

    print(f"\n[OK] Model registered: {model_id} (Champion: {payload['is_champion']})")
    return payload


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_csv = os.path.join(BASE_DIR, "data", "belt_rul_training_data.csv")
    default_output = os.path.join(BASE_DIR, "models_v2")

    parser = argparse.ArgumentParser(description="Train Belt RUL Model")
    parser.add_argument("--csv", default=default_csv, help="Path to CSV dataset")
    parser.add_argument("--output", default=default_output, help="Output directory for model and metadata")
    parser.add_argument("--mapping", default=None, help="Approved *.mapping.json (omit for canonical-named CSVs)")
    parser.add_argument("--algorithm", default="xgboost", choices=["xgboost", "random_forest", "gradient_boosting"], help="Algorithm to train")
    parser.add_argument("--belt-type", default="flat_conveyor", help="Belt type ID")
    parser.add_argument("--promote", action="store_true", help="Promote directly to active champion model")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path) and os.path.exists(os.path.join(BASE_DIR, csv_path)):
        csv_path = os.path.join(BASE_DIR, csv_path)

    mapping = None
    if args.mapping:
        mapping_path = args.mapping
        if not os.path.exists(mapping_path) and os.path.exists(os.path.join(BASE_DIR, mapping_path)):
            mapping_path = os.path.join(BASE_DIR, mapping_path)
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)["mapping"]
    train(csv_path, args.output, mapping, algorithm=args.algorithm, belt_type=args.belt_type, promote_to_champion=args.promote)
