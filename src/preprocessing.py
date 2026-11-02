"""
preprocessing.py — Data cleaning, feature engineering and temporal split.

Key rules enforced:
  - No random shuffling of time-series records.
  - All transformers (scalers, encoders) are fitted on train only.
  - Temporal split: last 20 % of weeks (chronologically) are held out for validation.
  - MarkDown missing values → filled with 0 + binary indicator flags.
  - Categorical features → one-hot encoded (Store Type).
  - Negative sales rows are kept (they are real returns) but flagged.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MARKDOWN_COLS  = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
CATEGORICAL    = ["Type"]
DROP_AFTER_ENC = ["Type", "Date"]

# The fraction of the chronological data used as the validation window
VALIDATION_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal and business features to the merged DataFrame.
    Does NOT modify the original; returns a new DataFrame.
    """
    df = df.copy()

    # ---- Temporal features -------------------------------------------------
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["WeekOfYear"]  = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]     = df["Date"].dt.quarter
    df["DayOfYear"]   = df["Date"].dt.dayofyear

    # Season: 1=Winter, 2=Spring, 3=Summer, 4=Autumn
    df["Season"] = df["Month"].map({
        12: 1, 1: 1, 2: 1,
        3: 2,  4: 2, 5: 2,
        6: 3,  7: 3, 8: 3,
        9: 4, 10: 4, 11: 4,
    })

    # ---- Holiday proximity flag  -------------------------------------------
    df["IsHoliday_int"] = df["IsHoliday"].astype(int)

    # ---- MarkDown handling: fill NaN with 0 + indicator flag ---------------
    for col in MARKDOWN_COLS:
        if col in df.columns:
            df[f"{col}_flag"] = df[col].notna().astype(int)
            df[col]           = df[col].fillna(0.0)

    # ---- CPI & Unemployment: fill with median per store --------------------
    for col in ["CPI", "Unemployment"]:
        if col in df.columns:
            store_medians = df.groupby("Store")[col].transform("median")
            df[col]       = df[col].fillna(store_medians)
            # If still missing (store had all NaN) use global median
            df[col]       = df[col].fillna(df[col].median())

    # ---- Negative sales flag -----------------------------------------------
    if "Weekly_Sales" in df.columns:
        df["IsReturn"] = (df["Weekly_Sales"] < 0).astype(int)

    # ---- Store Type one-hot encoding ----------------------------------------
    if "Type" in df.columns:
        type_dummies = pd.get_dummies(df["Type"], prefix="StoreType", drop_first=False, dtype=int)
        df = pd.concat([df, type_dummies], axis=1)

    return df


# ---------------------------------------------------------------------------
# Feature list (resolved lazily after engineering)
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the ordered list of numeric feature column names to use as X.
    Excludes target, identifier, and date columns.
    """
    exclude = {
        "Weekly_Sales", "Date", "Type",
        # keep Store and Dept as numeric identifiers (they encode group info)
    }
    # Candidate columns
    candidates = [c for c in df.columns if c not in exclude]
    # Drop columns that are object/categorical dtype (shouldn't exist after encoding)
    candidates = [c for c in candidates if df[c].dtype != object]
    return candidates


# ---------------------------------------------------------------------------
# Temporal train / validation split
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, val_fraction: float = VALIDATION_FRACTION):
    """
    Split the DataFrame chronologically.

    Returns (train_df, val_df) where val_df contains the last `val_fraction`
    of unique dates.  This ensures no future data leaks into training.
    """
    sorted_dates = sorted(df["Date"].unique())
    n_val        = max(1, int(len(sorted_dates) * val_fraction))
    cutoff_date  = sorted_dates[-n_val]

    train_df = df[df["Date"] < cutoff_date].copy()
    val_df   = df[df["Date"] >= cutoff_date].copy()
    return train_df, val_df


# ---------------------------------------------------------------------------
# Build X / y matrices
# ---------------------------------------------------------------------------

def build_Xy(df: pd.DataFrame, feature_cols: list[str]):
    """Return (X, y) numpy arrays from the engineered DataFrame."""
    X = df[feature_cols].values.astype(np.float64)
    y = df["Weekly_Sales"].values.astype(np.float64)
    return X, y


# ---------------------------------------------------------------------------
# Scaler (fit on train only)
# ---------------------------------------------------------------------------

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)


# ---------------------------------------------------------------------------
# Full pipeline runner (called by train.py)
# ---------------------------------------------------------------------------

def prepare_data(merged_df: pd.DataFrame, val_fraction: float = VALIDATION_FRACTION):
    """
    End-to-end preprocessing that returns everything needed for training.

    Returns dict with keys:
      X_train, y_train, X_val, y_val
      X_train_scaled, X_val_scaled
      scaler
      feature_cols
      train_df, val_df
      cutoff_date (str)
    """
    # 1. Feature engineering
    df = engineer_features(merged_df)

    # 2. Temporal split BEFORE any fitting
    train_df, val_df = temporal_split(df, val_fraction)

    # 3. Resolve feature columns from TRAINING set only
    feature_cols = get_feature_columns(train_df)

    # 4. Build matrices
    X_train, y_train = build_Xy(train_df, feature_cols)
    X_val,   y_val   = build_Xy(val_df,   feature_cols)

    # 5. Fit scaler on training data only; transform both
    scaler        = fit_scaler(X_train)
    X_train_sc    = apply_scaler(scaler, X_train)
    X_val_sc      = apply_scaler(scaler, X_val)

    cutoff = str(val_df["Date"].min().date())

    return dict(
        X_train        = X_train,
        y_train        = y_train,
        X_val          = X_val,
        y_val          = y_val,
        X_train_scaled = X_train_sc,
        X_val_scaled   = X_val_sc,
        scaler         = scaler,
        feature_cols   = feature_cols,
        train_df       = train_df,
        val_df         = val_df,
        cutoff_date    = cutoff,
    )
