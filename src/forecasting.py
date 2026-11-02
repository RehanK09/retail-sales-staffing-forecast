"""
forecasting.py — Generate sales predictions for a given store/dept/date.

Uses the saved trained models. Prediction inputs are:
  - store_id, dept_id, date, is_holiday
  - Engineered temporal features (derived from date)
  - Store metadata (type, size) from stores.csv
  - External features from features.csv (most recent known values used)

The prediction pipeline:
  1. Build a one-row DataFrame matching the training feature schema.
  2. Apply the saved scaler for Linear Regression.
  3. Produce predictions from both LR and DT models.
"""

import numpy as np
import pandas as pd

from src.data_loader import load_stores, load_features
from src.preprocessing import engineer_features
from src.models import load_all, models_exist


# ---------------------------------------------------------------------------
# Helper: build a feature row for a given store/dept/date
# ---------------------------------------------------------------------------

def _build_input_row(
    store_id:   int,
    dept_id:    int,
    date:       pd.Timestamp,
    is_holiday: bool,
    stores_df:  pd.DataFrame,
    features_df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Construct a 2-D numpy array (shape: 1 × len(feature_cols)) representing
    a single prediction request.

    External features are filled from the nearest historical date
    for the selected store.
    """
    # 1. Nearest historical feature row for this store
    store_feats = features_df[features_df["Store"] == store_id].copy()
    if not store_feats.empty:
        store_feats = store_feats.sort_values("Date")
        # Use the most recent row that is <= requested date; else earliest row
        past = store_feats[store_feats["Date"] <= date]
        feat_row = past.iloc[-1] if not past.empty else store_feats.iloc[0]
    else:
        feat_row = pd.Series(dtype=float)

    # 2. Store metadata
    store_meta = stores_df[stores_df["Store"] == store_id]
    store_type = store_meta["Type"].values[0] if not store_meta.empty else "A"
    store_size = int(store_meta["Size"].values[0]) if not store_meta.empty else 130000

    # 3. Build base row dict
    row = {
        "Store":         store_id,
        "Dept":          dept_id,
        "Date":          date,
        "Weekly_Sales":  0.0,          # placeholder (target column, not used for features)
        "IsHoliday":     is_holiday,
        "Type":          store_type,
        "Size":          store_size,
        "Temperature":   float(feat_row.get("Temperature",   60.0)),
        "Fuel_Price":    float(feat_row.get("Fuel_Price",     3.5)),
        "MarkDown1":     float(feat_row.get("MarkDown1",      0.0)) if pd.notna(feat_row.get("MarkDown1", 0)) else 0.0,
        "MarkDown2":     float(feat_row.get("MarkDown2",      0.0)) if pd.notna(feat_row.get("MarkDown2", 0)) else 0.0,
        "MarkDown3":     float(feat_row.get("MarkDown3",      0.0)) if pd.notna(feat_row.get("MarkDown3", 0)) else 0.0,
        "MarkDown4":     float(feat_row.get("MarkDown4",      0.0)) if pd.notna(feat_row.get("MarkDown4", 0)) else 0.0,
        "MarkDown5":     float(feat_row.get("MarkDown5",      0.0)) if pd.notna(feat_row.get("MarkDown5", 0)) else 0.0,
        "CPI":           float(feat_row.get("CPI",          211.0)),
        "Unemployment":  float(feat_row.get("Unemployment",   7.5)),
    }

    df_row = pd.DataFrame([row])

    # 4. Run through the feature engineering function (same as training)
    df_row = engineer_features(df_row)

    # 5. Keep only the feature columns used during training; fill any extras with 0
    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0.0
    X = df_row[feature_cols].values.astype(np.float64)
    return X


# ---------------------------------------------------------------------------
# Public prediction function
# ---------------------------------------------------------------------------

def predict(
    store_id:   int,
    dept_id:    int,
    date:       pd.Timestamp,
    is_holiday: bool = False,
) -> dict:
    """
    Predict weekly sales for a given store / department / date.

    Returns a dict with:
      lr_pred: float  — Linear Regression prediction
      dt_pred: float  — Decision Tree prediction
    """
    if not models_exist():
        raise RuntimeError("Trained models not found. Run train.py first.")

    lr_model, dt_model, scaler, feature_cols = load_all()
    stores_df   = load_stores()
    features_df = load_features()
    # Parse dates in features_df if not already done
    if features_df["Date"].dtype == object:
        features_df["Date"] = pd.to_datetime(features_df["Date"])

    X = _build_input_row(
        store_id, dept_id, date, is_holiday,
        stores_df, features_df, feature_cols,
    )

    X_scaled = scaler.transform(X)

    lr_pred = float(lr_model.predict(X_scaled)[0])
    dt_pred = float(dt_model.predict(X)[0])

    return {
        "lr_pred": round(lr_pred, 2),
        "dt_pred": round(dt_pred, 2),
    }


# ---------------------------------------------------------------------------
# Batch prediction on a range of dates
# ---------------------------------------------------------------------------

def predict_range(
    store_id:   int,
    dept_id:    int,
    start_date: pd.Timestamp,
    end_date:   pd.Timestamp,
    is_holiday_fn=None,
) -> pd.DataFrame:
    """
    Predict weekly sales from start_date to end_date (weekly frequency).

    is_holiday_fn: callable(date) -> bool  (defaults to always False)
    Returns a DataFrame with columns: Date, lr_pred, dt_pred
    """
    if not models_exist():
        raise RuntimeError("Trained models not found. Run train.py first.")

    lr_model, dt_model, scaler, feature_cols = load_all()
    stores_df   = load_stores()
    features_df = load_features()
    if features_df["Date"].dtype == object:
        features_df["Date"] = pd.to_datetime(features_df["Date"])

    dates = pd.date_range(start=start_date, end=end_date, freq="W-FRI")
    records = []
    for d in dates:
        is_holiday = is_holiday_fn(d) if is_holiday_fn else False
        X = _build_input_row(store_id, dept_id, d, is_holiday,
                             stores_df, features_df, feature_cols)
        X_sc   = scaler.transform(X)
        lr_p   = float(lr_model.predict(X_sc)[0])
        dt_p   = float(dt_model.predict(X)[0])
        records.append({"Date": d, "LR_Prediction": round(lr_p, 2), "DT_Prediction": round(dt_p, 2)})

    return pd.DataFrame(records)
