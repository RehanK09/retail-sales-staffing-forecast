"""
train.py — End-to-end model training and evaluation script.

Executes the complete ML pipeline:
  1. [DATA PREPARATION] Load and merge genuine Walmart CSV files.
  2. [FEATURE ENGINEERING & VALIDATION] Temporal feature extraction & chronological split.
  3. [SUPERVISED MODEL 1] Train Linear Regression (baseline on scaled features).
  4. [SUPERVISED MODEL 2] Train Decision Tree Regressor (nonlinear model on raw features).
  5. [MODEL EVALUATION] Evaluate MAE, RMSE, R² (overall, holiday, non-holiday).
  6. [ARTIFACT PERSISTENCE] Save models, scaler, and evaluation results.

Run from project root:
    .venv/Scripts/python.exe train.py
"""

import sys
import os
import time

# Ensure src/ is on the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader    import load_merged_train, dataset_summary
from src.preprocessing  import prepare_data
from src.models         import (
    train_linear_regression, train_decision_tree,
    save_all, MODELS_DIR
)
from src.evaluation import evaluate_both_models, save_results, print_results


def main():
    t0 = time.time()
    print("=" * 70)
    print("  WALMART SALES FORECASTING — END-TO-END MACHINE LEARNING PIPELINE")
    print("  Student: Rehan Khan | Reg No: 12526856 | Course: CSE252")
    print("=" * 70)

    # =======================================================================
    # [STAGE 1: Data Ingestion & Relational Merging]
    # =======================================================================
    print("\n[STAGE 1/5] Ingesting and Merging Multi-Table Datasets...")
    merged_df = load_merged_train()
    summary   = dataset_summary(merged_df)
    print(f"      Merged Shape : {merged_df.shape[0]:,} records × {merged_df.shape[1]} features")
    print(f"      Store Count  : {summary['stores']} (Types A, B, C)")
    print(f"      Dept Count   : {summary['departments']}")
    print(f"      Date Range   : {summary['date_min']}  →  {summary['date_max']}")

    # =======================================================================
    # [STAGE 2: Feature Engineering & Time-Series Chronological Split]
    # =======================================================================
    print("\n[STAGE 2/5] Engineering Features & Performing Chronological Validation Split...")
    data = prepare_data(merged_df, val_fraction=0.20)

    X_train        = data["X_train"]
    y_train        = data["y_train"]
    X_val          = data["X_val"]
    y_val          = data["y_val"]
    X_train_scaled = data["X_train_scaled"]
    X_val_scaled   = data["X_val_scaled"]
    scaler         = data["scaler"]
    feature_cols   = data["feature_cols"]
    train_df       = data["train_df"]
    val_df         = data["val_df"]
    cutoff         = data["cutoff_date"]

    print(f"      Feature Count  : {len(feature_cols)} predictor columns")
    print(f"      Training Set   : {len(X_train):,} observations (dates strictly before {cutoff})")
    print(f"      Validation Set : {len(X_val):,} observations (dates from {cutoff} onwards)")
    print(f"      Holiday Subset : {val_df['IsHoliday'].sum():,} weeks")
    print(f"      Non-Hol Subset : {(~val_df['IsHoliday']).sum():,} weeks")

    # =======================================================================
    # [STAGE 3: Supervised Model 1 — Linear Regression Baseline]
    # =======================================================================
    print("\n[STAGE 3/5] Training Linear Regression (Standardized Baseline Model)...")
    t_lr = time.time()
    lr_model = train_linear_regression(X_train_scaled, y_train)
    print(f"      Linear Regression trained in {time.time()-t_lr:.2f}s")

    # =======================================================================
    # [STAGE 4: Supervised Model 2 — Decision Tree Regression]
    # =======================================================================
    print("\n[STAGE 4/5] Training Decision Tree Regressor (Regularized Nonlinear Model)...")
    t_dt = time.time()
    dt_model = train_decision_tree(X_train, y_train)
    print(f"      Decision Tree trained in {time.time()-t_dt:.2f}s")

    # =======================================================================
    # [STAGE 5: Model Evaluation, Holiday Gap Analysis & Artifact Persistence]
    # =======================================================================
    print("\n[STAGE 5/5] Evaluating Models on Chronological Hold-out Validation Set...")
    lr_pred = lr_model.predict(X_val_scaled)
    dt_pred = dt_model.predict(X_val)

    results = evaluate_both_models(val_df, y_val, lr_pred, dt_pred)
    print_results(results)

    print("\n--- Persisting Model Artifacts & Evaluation Outputs ---")
    save_all(lr_model, dt_model, scaler, feature_cols)
    rpath = save_results(results)
    print(f"Evaluation metrics saved to: {rpath}")

    print(f"\n✓ Pipeline successfully completed in {time.time()-t0:.1f}s total.")
    print(f"  Artifacts directory: {MODELS_DIR}")


if __name__ == "__main__":
    main()
