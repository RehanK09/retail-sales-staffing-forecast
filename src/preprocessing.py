"""
preprocessing.py — Data cleaning, feature engineering and temporal split.

Key rules enforced:
  - No random shuffling of time-series records.
  - All transformers (scalers, encoders) are fitted on train only.
  - Temporal split: last 20% of weeks (chronologically) are held out for validation.
  - MarkDown missing values → filled with 0 + binary indicator flags.
  - Categorical features → one-hot encoded (Store Type).
  - Negative sales rows are kept (they are real returns) but flagged.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# [TOPIC: Pipeline Constants & Operational Parameters]
# ===========================================================================
MARKDOWN_COLS       = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
CATEGORICAL         = ["Type"]
DROP_AFTER_ENC      = ["Type", "Date"]
VALIDATION_FRACTION = 0.20  # Chronological hold-out fraction (last 20% of timeline)


# ===========================================================================
# [TOPIC: Feature Engineering & Preprocessing Workflow]
# ===========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply comprehensive feature engineering and cleaning transformations.
    Does NOT modify the input DataFrame; returns a transformed copy.
    """
    df = df.copy()

    # -----------------------------------------------------------------------
    # [SUB-TOPIC: Temporal & Calendar Feature Extraction]
    # -----------------------------------------------------------------------
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["WeekOfYear"]  = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]     = df["Date"].dt.quarter
    df["DayOfYear"]   = df["Date"].dt.dayofyear

    # Seasonal mapping: 1=Winter, 2=Spring, 3=Summer, 4=Autumn
    df["Season"] = df["Month"].map({
        12: 1, 1: 1, 2: 1,
        3: 2,  4: 2, 5: 2,
        6: 3,  7: 3, 8: 3,
        9: 4, 10: 4, 11: 4,
    })

    # -----------------------------------------------------------------------
    # [SUB-TOPIC: Holiday Feature Engineering]
    # -----------------------------------------------------------------------
    df["IsHoliday_int"] = df["IsHoliday"].astype(int)

    # -----------------------------------------------------------------------
    # [SUB-TOPIC: Missing Value Treatment — Promotional MarkDowns]
    # Impute missing promotional markdowns with 0.0 + binary indicator flag
    # (Since Walmart only introduced markdown tracking from November 2011)
    # -----------------------------------------------------------------------
    for col in MARKDOWN_COLS:
        if col in df.columns:
            df[f"{col}_flag"] = df[col].notna().astype(int)
            df[col]           = df[col].fillna(0.0)

    # -----------------------------------------------------------------------
    # [SUB-TOPIC: Missing Value Treatment — Store-Grouped Median Imputation]
    # Impute CPI and Unemployment using store-level median grouping
    # -----------------------------------------------------------------------
    for col in ["CPI", "Unemployment"]:
        if col in df.columns:
            store_medians = df.groupby("Store")[col].transform("median")
            df[col]       = df[col].fillna(store_medians)
            # Global median fallback if a store has entirely null values
            df[col]       = df[col].fillna(df[col].median())

    # -----------------------------------------------------------------------
    # [SUB-TOPIC: Outlier & Return Detection]
    # Negative sales represent merchandise returns/refunds; flag them explicitly
    # -----------------------------------------------------------------------
    if "Weekly_Sales" in df.columns:
        df["IsReturn"] = (df["Weekly_Sales"] < 0).astype(int)

    # -----------------------------------------------------------------------
    # [SUB-TOPIC: Feature Encoding — One-Hot Categorical Encoding]
    # Encode store types ('A', 'B', 'C') into discrete binary columns
    # -----------------------------------------------------------------------
    if "Type" in df.columns:
        type_dummies = pd.get_dummies(df["Type"], prefix="StoreType", drop_first=False, dtype=int)
        df = pd.concat([df, type_dummies], axis=1)

    return df


# ===========================================================================
# [TOPIC: Feature Selection & Input Matrix Resolution]
# ===========================================================================

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the ordered list of numeric predictor features (X).
    Excludes the target variable (Weekly_Sales), date, and unencoded text columns.
    """
    exclude = {
        "Weekly_Sales", "Date", "Type",
    }
    candidates = [c for c in df.columns if c not in exclude]
    candidates = [c for c in candidates if df[c].dtype != object]
    return candidates


def build_Xy(df: pd.DataFrame, feature_cols: list[str]):
    """Construct (X, y) 2D NumPy arrays from the engineered DataFrame."""
    X = df[feature_cols].values.astype(np.float64)
    y = df["Weekly_Sales"].values.astype(np.float64)
    return X, y


# ===========================================================================
# [TOPIC: Chronological Validation Split (Time-Series Train-Test Split)]
# ===========================================================================

def temporal_split(df: pd.DataFrame, val_fraction: float = VALIDATION_FRACTION):
    """
    Split the dataset chronologically (no random shuffling).

    Guarantees:
      - Training observations strictly precede validation observations.
      - Eliminates lookahead bias and data leakage in time-series evaluation.
    """
    sorted_dates = sorted(df["Date"].unique())
    n_val        = max(1, int(len(sorted_dates) * val_fraction))
    cutoff_date  = sorted_dates[-n_val]

    train_df = df[df["Date"] < cutoff_date].copy()
    val_df   = df[df["Date"] >= cutoff_date].copy()
    return train_df, val_df


# ===========================================================================
# [TOPIC: Feature Scaling (StandardScaler — Fit on Train Only)]
# ===========================================================================

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler strictly on training set to prevent data leakage."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Transform feature matrix using the pre-fitted scaler."""
    return scaler.transform(X)


# ===========================================================================
# [TOPIC: End-to-End Preprocessing Pipeline Execution]
# ===========================================================================

def prepare_data(merged_df: pd.DataFrame, val_fraction: float = VALIDATION_FRACTION):
    """
    Execute complete data preprocessing, feature engineering, chronological split,
    and feature scaling for model training.
    """
    # 1. Feature Engineering
    df = engineer_features(merged_df)

    # 2. Chronological Split (Before any transformation fitting)
    train_df, val_df = temporal_split(df, val_fraction)

    # 3. Resolve Feature Columns from Training Split
    feature_cols = get_feature_columns(train_df)

    # 4. Construct Matrices
    X_train, y_train = build_Xy(train_df, feature_cols)
    X_val,   y_val   = build_Xy(val_df,   feature_cols)

    # 5. Fit Scaler Strictly on Training Data Only
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
