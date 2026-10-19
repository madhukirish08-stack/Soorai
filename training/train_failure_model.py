"""
Model 1: Belt Failure Classification
=====================================
Target: eventual_failure (0 = No failure / Censored, 1 = Failed)
Compares: Logistic Regression, Random Forest, HistGradientBoosting, XGBoost
Saves: models/failure_model.pkl, models/preprocessing.pkl
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def train_failure_classification(csv_path="data/data.csv", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)

    print("=" * 80)
    print("TRAINING MODEL 1: BELT FAILURE CLASSIFICATION")
    print(f"Source dataset: {csv_path} (READ-ONLY)")
    print("=" * 80)

    df = pd.read_csv(csv_path)

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
        col for col in df.columns if col not in leakage_cols and col not in categorical_features
    ]

    all_features = categorical_features + numerical_features
    X = df[all_features]
    y = df["eventual_failure"]

    print(f"Total samples: {len(df):,}")
    print(f"Features used: {len(all_features)} ({len(categorical_features)} categorical, {len(numerical_features)} numerical)")
    print(f"Target distribution: 0 -> {sum(y == 0):,} ({sum(y == 0)/len(y)*100:.1f}%), 1 -> {sum(y == 1):,} ({sum(y == 1)/len(y)*100:.1f}%)")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Reproducible Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop"
    )

    # Fit preprocessor on training set only to prevent test set data leakage
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessing.pkl"))
    print(f"Preprocessing pipeline saved to {os.path.join(output_dir, 'preprocessing.pkl')}")

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss")
    }

    results = {}
    best_model_name = None
    best_roc_auc = -1
    best_pipeline = None

    for name, clf in candidate_models.items():
        print(f"\nTraining candidate: {name}...")
        pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", clf)
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm
        }

        print(f"  -> Accuracy:  {acc*100:.2f}%")
        print(f"  -> Precision: {prec*100:.2f}%")
        print(f"  -> Recall:    {rec*100:.2f}%")
        print(f"  -> F1-Score:  {f1:.4f}")
        print(f"  -> ROC-AUC:   {auc:.4f}")

        if auc > best_roc_auc:
            best_roc_auc = auc
            best_model_name = name
            best_pipeline = pipe

    print("\n" + "=" * 80)
    print(f"SELECTED BEST CLASSIFIER: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    print("=" * 80)

    # Save the selected model
    model_save_path = os.path.join(output_dir, "failure_model.pkl")
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved best model to: {model_save_path}")

    # Extract Feature Importances / Coefficients
    final_clf = best_pipeline.named_steps["clf"]
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names = numerical_features + encoded_cat_names

    feature_importances = {}
    if hasattr(final_clf, "coef_"):
        weights = final_clf.coef_[0]
        for fname, w in zip(feature_names, weights):
            feature_importances[fname] = float(np.abs(w))
    elif hasattr(final_clf, "feature_importances_"):
        weights = final_clf.feature_importances_
        for fname, w in zip(feature_names, weights):
            feature_importances[fname] = float(w)

    top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 15 Most Important Features in Failure Prediction:")
    for rank, (fname, score) in enumerate(top_features[:15], 1):
        print(f"  {rank:2d}. {fname:35s}: {score:.5f}")

    eval_summary = {
        "task": "Failure Classification",
        "target": "eventual_failure",
        "selected_model": best_model_name,
        "models_compared": results,
        "best_metrics": results[best_model_name],
        "top_features": dict(top_features[:20])
    }

    with open("evaluation/failure_classification_metrics.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary

if __name__ == "__main__":
    train_failure_classification()
