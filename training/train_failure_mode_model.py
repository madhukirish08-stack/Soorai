"""
Model 3: Belt Failure-Mode Multiclass Prediction
================================================
Target: failure_mode (6 classes: Crack, Abrasion, Splice_Failure, Delamination, Edge_Damage, Thermal_Damage)
Sample population: Belts with eventual_failure == 1 (22,402 cases)
Saves: models/failure_mode_model.pkl, models/failure_mode_encoder.pkl
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

def train_failure_mode_model(csv_path="data/data.csv", output_dir="models", preprocessor_path="models/preprocessing.pkl"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)

    print("=" * 80)
    print("TRAINING MODEL 3: BELT FAILURE-MODE PREDICTION")
    print(f"Source dataset: {csv_path} (READ-ONLY)")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    # Filter to failed belts where a true failure mode occurred
    failed_df = df[df["eventual_failure"] == 1].copy()
    print(f"Total failed belts available for training: {len(failed_df):,}")

    # Strict leakage exclusions
    leakage_cols = [
        "eventual_failure", "censored", "failure_hour", "observed_time",
        "failure_mode", "failure_location", "belt_id", "manufacturing_batch"
    ]
    categorical_features = [
        "belt_model", "belt_type", "belt_material", "reinforcement_type",
        "manufacturer", "splice_type"
    ]
    numerical_features = [
        col for col in failed_df.columns if col not in leakage_cols and col not in categorical_features
    ]
    all_features = categorical_features + numerical_features

    # Load preprocessor
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}.")
    preprocessor = joblib.load(preprocessor_path)

    # Encode multiclass labels
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(failed_df["failure_mode"])
    classes = label_encoder.classes_.tolist()
    print(f"Failure Mode classes ({len(classes)}): {classes}")

    # Transform features
    X_encoded = preprocessor.transform(failed_df[all_features])

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    candidate_models = {
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=120, max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    best_model_name = None
    best_acc = -1
    best_clf = None
    model_evals = {}

    for name, clf in candidate_models.items():
        print(f"\nTraining candidate: {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        model_evals[name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4)
        }
        print(f"  -> Accuracy:    {acc*100:.2f}%")
        print(f"  -> Macro F1:    {macro_f1:.4f}")
        print(f"  -> Weighted F1: {weighted_f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_clf = clf

    print("\n" + "=" * 80)
    print(f"SELECTED BEST FAILURE-MODE CLASSIFIER: {best_model_name} (Accuracy: {best_acc*100:.2f}%)")
    print("=" * 80)

    # Save best model and label encoder
    model_path = os.path.join(output_dir, "failure_mode_model.pkl")
    encoder_path = os.path.join(output_dir, "failure_mode_encoder.pkl")
    joblib.dump(best_clf, model_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"Saved failure mode model to: {model_path}")
    print(f"Saved label encoder to: {encoder_path}")

    # Full evaluation report
    y_test_pred = best_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred).tolist()
    report = classification_report(y_test, y_test_pred, target_names=classes, output_dict=True, zero_division=0)

    eval_summary = {
        "task": "Failure Mode Multiclass Prediction",
        "selected_model": best_model_name,
        "classes": classes,
        "models_compared": model_evals,
        "test_accuracy": round(best_acc, 4),
        "macro_f1": round(float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": cm,
        "classification_report": report
    }

    with open("evaluation/failure_mode_metrics.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary

if __name__ == "__main__":
    train_failure_mode_model()
