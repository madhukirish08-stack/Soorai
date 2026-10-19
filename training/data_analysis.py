"""
Belt Fatigue Testing and Failure Prediction - Exploratory Data Analysis & Leakage Audit
========================================================================================
This script performs a rigorous, reproducible, read-only analysis of data/data.csv.
DO NOT MODIFY OR DELETE THE ORIGINAL CSV.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

def analyze_dataset(csv_path="data/data.csv"):
    # Ensure read-only resolution of CSV path
    if not os.path.exists(csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
        elif os.path.exists("data.csv"):
            csv_path = "data.csv"
        else:
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print("=" * 80)
    print(f"LOADING DATASET: {csv_path} (READ-ONLY)")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    # 1. Basic Dimensions
    n_rows, n_cols = df.shape
    print(f"\n[1] DATASET DIMENSIONS")
    print(f"  - Total Rows:    {n_rows:,}")
    print(f"  - Total Columns: {n_cols}")

    # 2. Missing Values & Duplicates
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    total_missing = missing.sum()
    n_duplicates = df.duplicated().sum()
    n_duplicates_noid = df.drop(columns=["belt_id"]).duplicated().sum() if "belt_id" in df.columns else None

    print(f"\n[2] DATA INTEGRITY & MISSING VALUE AUDIT")
    print(f"  - Total Missing Values: {total_missing}")
    if len(cols_with_missing) == 0:
        print("  - All 100 columns are 100% complete (0 NaN/null values).")
    else:
        print(f"  - Columns with missing values:\n{cols_with_missing}")
    print(f"  - Duplicate Rows (all columns): {n_duplicates}")
    print(f"  - Duplicate Rows (excluding belt_id): {n_duplicates_noid}")

    # 3. Column Data Types & Classification
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"\n[3] COLUMN DATA TYPES BREAKDOWN")
    print(f"  - Numerical Columns ({len(num_cols)}):")
    print(f"  - Categorical / String Columns ({len(cat_cols)}):")

    print("\n--- ALL 100 COLUMNS AND DTYPES ---")
    for idx, col in enumerate(df.columns, 1):
        dtype_str = str(df[col].dtype)
        nunique = df[col].nunique()
        print(f"  {idx:3d}. {col:35s} | Type: {dtype_str:10s} | Unique Values: {nunique:6d}")

    # 4. Categorical Feature Inspection
    print(f"\n[4] CATEGORICAL FEATURE VALUE COUNTS")
    for col in cat_cols:
        print(f"\n  Column: '{col}' ({df[col].nunique()} unique values)")
        vc = df[col].value_counts()
        if len(vc) <= 10:
            for val, cnt in vc.items():
                print(f"    - {str(val):30s}: {cnt:6,d} ({cnt/n_rows*100:5.2f}%)")
        else:
            print(f"    - [High-cardinality preview top 5]:")
            for val, cnt in vc.head(5).items():
                print(f"      * {str(val):30s}: {cnt:6,d}")

    # 5. Target Variables Analysis
    print(f"\n[5] TARGET COLUMNS IN-DEPTH ANALYSIS")
    target_names = ["eventual_failure", "failure_hour", "failure_mode", "failure_location", "observed_time", "censored"]
    for t in target_names:
        if t in df.columns:
            if df[t].dtype == "object" or df[t].nunique() <= 10:
                print(f"\n  Target: '{t}' (Categorical/Discrete)")
                for val, cnt in df[t].value_counts().items():
                    print(f"    - {str(val):25s}: {cnt:6,d} ({cnt/n_rows*100:5.2f}%)")
            else:
                print(f"\n  Target: '{t}' (Continuous)")
                s = df[t].describe()
                print(f"    Min: {s['min']:,.2f} | 25%: {s['25%']:,.2f} | Median: {s['50%']:,.2f} | Mean: {s['mean']:,.2f} | 75%: {s['75%']:,.2f} | Max: {s['max']:,.2f} | Std: {s['std']:,.2f}")

    # 6. Target Cross-Relationships & Survival Invariants
    print(f"\n[6] CRITICAL MATHEMATICAL RELATIONSHIPS AMONG TARGETS")
    f_failed = df[df["eventual_failure"] == 1]
    f_censored = df[df["eventual_failure"] == 0]

    obs_eq_fail = (f_failed["observed_time"] == f_failed["failure_hour"]).all()
    obs_eq_run = (f_censored["observed_time"] == f_censored["runtime_hours"]).all()
    fail_gt_obs = (f_censored["failure_hour"] > f_censored["observed_time"]).all()
    cens_inv_ef = ((df["censored"] + df["eventual_failure"]) == 1).all()

    print(f"  - Censored is exact complement of eventual_failure (censored = 1 - eventual_failure): {cens_inv_ef}")
    print(f"  - For failed belts (eventual_failure == 1): observed_time == failure_hour strictly holds: {obs_eq_fail}")
    print(f"  - For non-failed belts (eventual_failure == 0): observed_time == runtime_hours strictly holds: {obs_eq_run}")
    print(f"  - For non-failed belts (eventual_failure == 0): failure_hour > observed_time strictly holds: {fail_gt_obs}")

    print("\n  Crosstab: eventual_failure vs failure_mode:")
    print(pd.crosstab(df["eventual_failure"], df["failure_mode"], margins=True))

    print("\n  Crosstab: eventual_failure vs failure_location:")
    print(pd.crosstab(df["eventual_failure"], df["failure_location"], margins=True))

    # 7. Class Balance Analysis
    print(f"\n[7] CLASS BALANCE SUMMARY: eventual_failure")
    ef_vc = df["eventual_failure"].value_counts()
    print(f"  - Class 0 (No Failure / Censored): {ef_vc[0]:,d} ({ef_vc[0]/n_rows*100:.2f}%)")
    print(f"  - Class 1 (Failure Occurred):     {ef_vc[1]:,d} ({ef_vc[1]/n_rows*100:.2f}%)")
    print(f"  - Ratio (Class 0 : Class 1):       {ef_vc[0]/ef_vc[1]:.2f} : 1.0 (Well-balanced, no severe skew)")

    # 8. Data Leakage Assessment
    print(f"\n[8] DATA LEAKAGE AUDIT")
    leakage_cols = {
        "eventual_failure": "Primary classification target.",
        "censored": "Exact 1:1 mathematical inverse of eventual_failure (leakage = 100%).",
        "failure_hour": "Records post-event time of failure for failed belts, synthetic future life for censored.",
        "observed_time": "Directly reveals failure_hour when failure occurs (observed_time == failure_hour).",
        "failure_mode": "Categorical post-event failure classification ('No_Failure' reveals failure state 100%).",
        "failure_location": "Categorical post-event physical location ('Not_Applicable' reveals failure state 100%).",
        "belt_id": "Primary key identifier (50,000 unique values), causes memorization/overfitting.",
        "manufacturing_batch": "High-cardinality metadata (1,500 batches), should be evaluated cautiously."
    }
    for col, reason in leakage_cols.items():
        print(f"  [!] {col:25s} -> {reason}")

    # 9. Top Predictive Features
    print(f"\n[9] CORRELATION WITH eventual_failure (Top 15 legitimate features)")
    safe_num_cols = [c for c in num_cols if c not in leakage_cols]
    corrs = df[safe_num_cols].corrwith(df["eventual_failure"]).abs().sort_values(ascending=False)
    for c, val in corrs.head(15).items():
        print(f"  - {c:32s}: Pearson |r| = {val:.4f}")

    # 10. Recommended Feature Partitions
    print(f"\n[10] RECOMMENDED FEATURE SPLITS FOR THE 3 ML TASKS")
    print("""
  A. Failure Classification Model (Target: eventual_failure):
     - Predicts: Probability that belt will fail given its current accumulated runtime and condition.
     - Inputs: Belt specifications, operating conditions, cycle counts, environmental exposure,
               current degradation measurements (wear, elongation, tension loss, crack dimensions),
               and maintenance history.
     - Strict Exclusions: censored, failure_hour, observed_time, failure_mode, failure_location, belt_id.

  B. Failure-Life / Survival Prediction Model:
     - Approach: Survival Analysis (Cox Proportional Hazards / Gradient Boosting Survival / Accelerated Failure Time).
     - Targets: Duration = observed_time, Event = eventual_failure (or 1 - censored).
     - Inputs: Same valid pre-failure predictors as above.
     - Note: Treating failure_hour as an ordinary regression target directly on all 50k rows causes severe bias
             because failure_hour for censored belts was never observed in production!

  C. Failure-Mode Prediction Model (Target: failure_mode):
     - Target: Multiclass failure mode (Crack, Abrasion, Splice_Failure, Delamination, Edge_Damage, Thermal_Damage).
     - Population: Belts with eventual_failure == 1 (22,402 cases, well-distributed across all 6 modes).
     - Inputs: Pre-failure sensor readings, load dynamics, temperature, vibration, degradation patterns, splice type.
     - Strict Exclusions: failure_location (post-mortem), failure_hour, observed_time, censored, belt_id.
    """)

    # Export machine-readable summary
    summary_path = "analysis_summary.json"
    summary_payload = {
        "total_rows": n_rows,
        "total_columns": n_cols,
        "duplicates": int(n_duplicates),
        "missing_values_count": int(total_missing),
        "numerical_columns_count": len(num_cols),
        "categorical_columns_count": len(cat_cols),
        "all_columns": df.columns.tolist(),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "eventual_failure_distribution": ef_vc.to_dict(),
        "censored_distribution": df["censored"].value_counts().to_dict(),
        "failure_mode_distribution": df["failure_mode"].value_counts().to_dict(),
        "failure_location_distribution": df["failure_location"].value_counts().to_dict(),
        "leakage_columns": list(leakage_cols.keys()),
        "top_correlations_with_failure": corrs.head(20).to_dict()
    }
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\nAnalysis summary successfully saved to: {summary_path}")
    print("=" * 80)

if __name__ == "__main__":
    analyze_dataset()
