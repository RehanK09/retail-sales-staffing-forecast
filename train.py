"""
train.py — End-to-end model training and evaluation script.

Executes the complete ML pipeline:
  1. Load and merge genuine Walmart CSV files.
  2. Engineer features.
  3. Chronological train / validation split.
  4. Train Linear Regression (on scaled features).
  5. Train Decision Tree (on unscaled features).
  6. Evaluate both models: overall, holiday, and non-holiday.
  7. Save models, scaler, feature column list, and evaluation results.

Run from the project root:
    .venv/Scripts/python.exe train.py
"""

import sys
import os
import time

# Ensure src/ is on the path when run from project root
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
    print("=" * 65)
    print("  Walmart Sales Forecasting — Model Training Pipeline")
    print("  Student: Rehan Khan | Reg No: 12526856 | CSE252")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading and merging datasets...")
    merged_df = load_merged_train()
    summary   = dataset_summary(merged_df)
    print(f"      Merged shape : {merged_df.shape[0]:,} rows × {merged_df.shape[1]} cols")
    print(f"      Stores       : {summary['stores']}")
    print(f"      Departments  : {summary['departments']}")
    print(f"      Date range   : {summary['date_min']}  →  {summary['date_max']}")

    # ------------------------------------------------------------------
    # Step 2: Preprocessing + temporal split
    # ------------------------------------------------------------------
    print("\n[2/5] Engineering features and performing chronological split...")
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

    print(f"      Features used  : {len(feature_cols)}")
    print(f"      Train samples  : {len(X_train):,}  (dates before {cutoff})")
    print(f"      Val   samples  : {len(X_val):,}  (dates from  {cutoff})")
    print(f"      Holiday (val)  : {val_df['IsHoliday'].sum():,}")
    print(f"      Non-hol (val)  : {(~val_df['IsHoliday']).sum():,}")

    # ------------------------------------------------------------------
    # Step 3: Train Linear Regression
    # ------------------------------------------------------------------
    print("\n[3/5] Training Linear Regression (baseline)...")
    t_lr = time.time()
    lr_model = train_linear_regression(X_train_scaled, y_train)
    print(f"      Done in {time.time()-t_lr:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Train Decision Tree
    # ------------------------------------------------------------------
    print("\n[4/5] Training Decision Tree Regressor...")
    t_dt = time.time()
    dt_model = train_decision_tree(X_train, y_train)
    print(f"      Done in {time.time()-t_dt:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: Evaluate
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating on chronological validation set...")
    lr_pred = lr_model.predict(X_val_scaled)
    dt_pred = dt_model.predict(X_val)

    results = evaluate_both_models(val_df, y_val, lr_pred, dt_pred)
    print_results(results)

    # ------------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------------
    print("\n--- Saving artifacts ---")
    save_all(lr_model, dt_model, scaler, feature_cols)
    rpath = save_results(results)
    print(f"Evaluation results saved to: {rpath}")

    print(f"\n✓  Training pipeline complete in {time.time()-t0:.1f}s total.")
    print(f"   Models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
