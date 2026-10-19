"""
Master Training Pipeline for Belt Fatigue Testing & Failure Prediction
======================================================================
Executes end-to-end training of:
  - Model 1: Failure Classification (eventual_failure)
  - Model 2: Failure-Life & Survival Analysis (observed_time, censored, failure_hour)
  - Model 3: Failure-Mode Multiclass Prediction (failure_mode)
Generates evaluation plots and creates models/model_metadata.json.
"""

import os
import sys

# Ensure training directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Import sub-modules
from train_failure_model import train_failure_classification
from train_life_model import train_life_model, UnifiedLifeModel
from train_failure_mode_model import train_failure_mode_model

def run_full_pipeline(csv_path="data/data.csv", output_dir="models"):
    print("=" * 80)
    print("STARTING FULL END-TO-END MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)

    # 1. Train Model 1 (Classification + Preprocessor)
    print("\n>>> STEP 1/3: Training Failure Classification Model...")
    cls_metrics = train_failure_classification(csv_path, output_dir)

    # 2. Train Model 2 (Survival AFT + Life Regressor)
    print("\n>>> STEP 2/3: Training Failure-Life & Survival Prediction Model...")
    life_metrics = train_life_model(csv_path, output_dir)

    # 3. Train Model 3 (Failure Mode Multiclass)
    print("\n>>> STEP 3/3: Training Failure-Mode Multiclass Model...")
    mode_metrics = train_failure_mode_model(csv_path, output_dir)

    # 4. Generate Metadata & Feature Schema for API & UI
    print("\n>>> STEP 4: Compiling Model Metadata and Feature Schema...")
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

    cat_options = {col: sorted(df[col].dropna().unique().tolist()) for col in categorical_features}
    num_stats = {}
    for col in numerical_features:
        s = df[col].describe()
        num_stats[col] = {
            "min": round(float(s["min"]), 2),
            "max": round(float(s["max"]), 2),
            "mean": round(float(s["mean"]), 2),
            "median": round(float(s["50%"]), 2),
            "std": round(float(s["std"]), 2),
            "default": round(float(s["50%"]), 2)
        }

    # Consolidated Metadata
    metadata = {
        "dataset_rows": len(df),
        "total_input_features": len(categorical_features) + len(numerical_features),
        "categorical_features": categorical_features,
        "categorical_options": cat_options,
        "numerical_features": numerical_features,
        "numerical_statistics": num_stats,
        "leakage_columns_excluded": leakage_cols,
        "models": {
            "failure_classification": {
                "target": "eventual_failure",
                "selected_algorithm": cls_metrics["selected_model"],
                "metrics": cls_metrics["best_metrics"],
                "compared": {k: {m: v[m] for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]} for k, v in cls_metrics["models_compared"].items()},
                "artifact": "models/failure_model.pkl"
            },
            "failure_life_survival": {
                "target": "observed_time / failure_hour",
                "methodology": "Dual AFT Survival (Right-Censored) + Gradient Boosting Life Regressor",
                "metrics": {
                    "concordance_index": life_metrics["concordance_index"],
                    "mae_hours": life_metrics["mae_hours"],
                    "rmse_hours": life_metrics["rmse_hours"],
                    "r2_score": life_metrics["r2_score"]
                },
                "artifact": "models/life_model.pkl"
            },
            "failure_mode": {
                "target": "failure_mode",
                "selected_algorithm": mode_metrics["selected_model"],
                "classes": mode_metrics["classes"],
                "metrics": {
                    "test_accuracy": mode_metrics["test_accuracy"],
                    "macro_f1": mode_metrics["macro_f1"]
                },
                "artifact": "models/failure_mode_model.pkl"
            }
        },
        "top_predictive_features": cls_metrics["top_features"]
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")

    with open("evaluation/all_metrics.json", "w") as f:
        json.dump(metadata["models"], f, indent=2)

    # 5. Generate Visual Diagnostic Plots
    print("\n>>> STEP 5: Generating Evaluation Plots...")
    try:
        # Plot 1: Failure Classification Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm_fail = np.array(cls_metrics["best_metrics"]["confusion_matrix"])
        sns.heatmap(cm_fail, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Predicted No Failure (0)", "Predicted Failure (1)"],
                    yticklabels=["Actual No Failure (0)", "Actual Failure (1)"])
        plt.title(f"Model 1: Failure Prediction Confusion Matrix\n({cls_metrics['selected_model']})")
        plt.tight_layout()
        plt.savefig("evaluation/failure_confusion_matrix.png", dpi=150)
        plt.close()

        # Plot 2: Top 15 Feature Importances
        plt.figure(figsize=(10, 6))
        top_items = list(cls_metrics["top_features"].items())[:15]
        top_items.reverse()
        names = [item[0] for item in top_items]
        values = [item[1] for item in top_items]
        plt.barh(names, values, color="#1f77b4")
        plt.xlabel("Relative Importance / Weight Magnitude")
        plt.title("Model 1: Top 15 Features for Failure Prediction")
        plt.tight_layout()
        plt.savefig("evaluation/feature_importance.png", dpi=150)
        plt.close()

        # Plot 3: Failure Mode Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm_mode = np.array(mode_metrics["confusion_matrix"])
        classes = mode_metrics["classes"]
        sns.heatmap(cm_mode, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=classes, yticklabels=classes)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Model 3: Failure-Mode Confusion Matrix\n({mode_metrics['selected_model']})")
        plt.tight_layout()
        plt.savefig("evaluation/failure_mode_confusion_matrix.png", dpi=150)
        plt.close()

        print("Visual plots saved in evaluation/ folder.")
    except Exception as e:
        print(f"Warning: Plot generation encountered non-fatal error: {e}")

    print("\n" + "=" * 80)
    print("ALL 3 MODELS SUCCESSFULLY TRAINED AND SAVED!")
    print("=" * 80)

if __name__ == "__main__":
    run_full_pipeline()
