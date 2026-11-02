"""
models.py — Model definitions and training for the two required algorithms.

Model 1: Linear Regression
  - Simple interpretable baseline; assumes a linear relationship between
    engineered features and weekly sales.
  - Requires feature scaling (StandardScaler already applied upstream).

Model 2: Decision Tree Regression
  - Captures non-linear, hierarchical interactions between features.
  - Uses max_depth and min_samples_leaf to control complexity and prevent
    over-fitting that would produce unrealistically perfect training scores.
  - Does NOT require scaling (tree-based model is scale-invariant).
"""

import joblib
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LR_PATH  = os.path.join(MODELS_DIR, "linear_regression.joblib")
DT_PATH  = os.path.join(MODELS_DIR, "decision_tree.joblib")
SC_PATH  = os.path.join(MODELS_DIR, "scaler.joblib")
FC_PATH  = os.path.join(MODELS_DIR, "feature_cols.joblib")


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_linear_regression() -> LinearRegression:
    """
    Build the Linear Regression baseline model.
    Fitting is performed on scaled features (StandardScaler applied upstream).
    """
    return LinearRegression(n_jobs=-1)


def build_decision_tree(
    max_depth: int = 12,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> DecisionTreeRegressor:
    """
    Build a Decision Tree Regressor with sensible regularisation.

    max_depth=12 allows the tree to model complex seasonal and store-level
    patterns while avoiding extreme over-fitting.
    min_samples_leaf=10 prevents very small terminal nodes from fitting noise.
    """
    return DecisionTreeRegressor(
        max_depth         = max_depth,
        min_samples_leaf  = min_samples_leaf,
        random_state      = random_state,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_linear_regression(X_scaled: np.ndarray, y: np.ndarray) -> LinearRegression:
    model = build_linear_regression()
    model.fit(X_scaled, y)
    return model


def train_decision_tree(X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
    """Decision Tree does not need scaled features."""
    model = build_decision_tree()
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(obj, path: str) -> None:
    joblib.dump(obj, path)


def load_model(path: str):
    return joblib.load(path)


def save_all(lr_model, dt_model, scaler, feature_cols) -> None:
    save_model(lr_model,     LR_PATH)
    save_model(dt_model,     DT_PATH)
    save_model(scaler,       SC_PATH)
    save_model(feature_cols, FC_PATH)
    print(f"Models saved to {MODELS_DIR}")


def load_all():
    """Load all persisted artifacts. Returns (lr_model, dt_model, scaler, feature_cols)."""
    lr_model     = load_model(LR_PATH)
    dt_model     = load_model(DT_PATH)
    scaler       = load_model(SC_PATH)
    feature_cols = load_model(FC_PATH)
    return lr_model, dt_model, scaler, feature_cols


def models_exist() -> bool:
    return all(os.path.exists(p) for p in [LR_PATH, DT_PATH, SC_PATH, FC_PATH])
