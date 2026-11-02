"""
data_loader.py — Load and merge the Walmart Kaggle dataset files.

Merges train.csv + stores.csv + features.csv into a single analysis-ready DataFrame.
All merges are left-joins on store/date keys. No data is fabricated.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(_ROOT, "data")


def _data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_train() -> pd.DataFrame:
    """Load train.csv and parse the Date column."""
    df = pd.read_csv(_data_path("train.csv"), parse_dates=["Date"])
    return df


def load_test() -> pd.DataFrame:
    """Load test.csv (no Weekly_Sales column)."""
    df = pd.read_csv(_data_path("test.csv"), parse_dates=["Date"])
    return df


def load_stores() -> pd.DataFrame:
    """Load stores.csv — 45 rows with Store, Type, Size."""
    return pd.read_csv(_data_path("stores.csv"))


def load_features() -> pd.DataFrame:
    """Load features.csv — store-level weekly external data."""
    df = pd.read_csv(_data_path("features.csv"), parse_dates=["Date"])
    return df


# ---------------------------------------------------------------------------
# Master merged loader
# ---------------------------------------------------------------------------

def load_merged_train() -> pd.DataFrame:
    """
    Returns a single DataFrame merging train + stores + features.

    Join keys:
      train  LEFT JOIN stores   ON Store
      result LEFT JOIN features ON [Store, Date]

    The IsHoliday column from features is dropped (train.csv is authoritative).
    """
    train    = load_train()
    stores   = load_stores()
    features = load_features()

    # Drop the duplicate IsHoliday from features (train.csv is source of truth)
    features = features.drop(columns=["IsHoliday"], errors="ignore")

    df = (
        train
        .merge(stores,   on="Store",          how="left")
        .merge(features, on=["Store", "Date"], how="left")
    )
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a quick summary dict for the loaded DataFrame."""
    return {
        "rows":         len(df),
        "columns":      len(df.columns),
        "stores":       df["Store"].nunique(),
        "departments":  df["Dept"].nunique(),
        "date_min":     str(df["Date"].min().date()),
        "date_max":     str(df["Date"].max().date()),
        "unique_weeks": df["Date"].nunique(),
        "holiday_rows": int(df["IsHoliday"].sum()),
        "non_holiday_rows": int((~df["IsHoliday"]).sum()),
        "missing_values": df.isnull().sum().to_dict(),
        "total_sales":  float(df["Weekly_Sales"].sum()),
        "mean_sales":   float(df["Weekly_Sales"].mean()),
        "negative_sales": int((df["Weekly_Sales"] < 0).sum()),
    }


if __name__ == "__main__":
    df = load_merged_train()
    print("Merged shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Date range:", df["Date"].min(), "→", df["Date"].max())
    print("Missing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
