"""
evaluation.py — Compute and report model evaluation metrics.

Evaluates both models on:
  - Overall validation set
  - Holiday weeks only
  - Non-holiday weeks only

Metrics: MAE, RMSE, R²

Results are returned as a structured dict and also saved to outputs/.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(OUTPUTS_DIR, "evaluation_results.json")


# ---------------------------------------------------------------------------
# Core metric function
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with MAE, RMSE, R² and sample count."""
    n = len(y_true)
    if n == 0:
        return {"MAE": None, "RMSE": None, "R2": None, "n": 0}
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4), "n": n}


# ---------------------------------------------------------------------------
# Holiday / non-holiday sub-group evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_holiday: np.ndarray,
    model_name: str,
) -> dict:
    """
    Evaluate a model across three groups:
      - overall
      - holiday weeks (IsHoliday == True)
      - non-holiday weeks (IsHoliday == False)
    """
    holiday_mask = is_holiday.astype(bool)

    overall     = compute_metrics(y_true, y_pred)
    holiday     = compute_metrics(y_true[holiday_mask],  y_pred[holiday_mask])
    non_holiday = compute_metrics(y_true[~holiday_mask], y_pred[~holiday_mask])

    # Calculate holiday vs non-holiday performance gap
    gap = {}
    for metric in ["MAE", "RMSE"]:
        if holiday[metric] is not None and non_holiday[metric] is not None:
            gap[f"{metric}_gap"] = round(holiday[metric] - non_holiday[metric], 2)
        else:
            gap[f"{metric}_gap"] = None

    return {
        "model_name":   model_name,
        "overall":      overall,
        "holiday":      holiday,
        "non_holiday":  non_holiday,
        "holiday_gap":  gap,
    }


# ---------------------------------------------------------------------------
# Evaluate both models together
# ---------------------------------------------------------------------------

def evaluate_both_models(
    val_df:        pd.DataFrame,
    y_val:         np.ndarray,
    lr_pred:       np.ndarray,
    dt_pred:       np.ndarray,
) -> dict:
    """
    Evaluate Linear Regression and Decision Tree on the validation DataFrame.

    val_df must contain the IsHoliday column for segment evaluation.
    """
    is_holiday = val_df["IsHoliday"].values.astype(bool)

    lr_results = evaluate_model(y_val, lr_pred, is_holiday, "Linear Regression")
    dt_results = evaluate_model(y_val, dt_pred, is_holiday, "Decision Tree")

    results = {
        "Linear Regression": lr_results,
        "Decision Tree":     dt_results,
    }
    return results


# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------

def save_results(results: dict) -> str:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return RESULTS_PATH


def load_results() -> dict | None:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Pretty printer (CLI use)
# ---------------------------------------------------------------------------

def print_results(results: dict) -> None:
    for model_name, res in results.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")
        _print_group("Overall",     res["overall"])
        _print_group("Holiday",     res["holiday"])
        _print_group("Non-Holiday", res["non_holiday"])
        print(f"\n  Holiday Performance Gap:")
        for k, v in res["holiday_gap"].items():
            print(f"    {k}: {v:+.2f}")


def _print_group(label: str, metrics: dict) -> None:
    n = metrics.get("n", "?")
    print(f"\n  {label}  (n={n:,})")
    print(f"    MAE : {metrics['MAE']}")
    print(f"    RMSE: {metrics['RMSE']}")
    print(f"    R²  : {metrics['R2']}")
