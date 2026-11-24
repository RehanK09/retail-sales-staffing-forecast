# Code Directory & Reference Guide

**Course:** CSE252 – Artificial Intelligence and Machine Learning  
**Project:** Retail Store Weekly Sales Forecasting and Staffing Demand Analysis System  
**Student:** Rehan Khan | **Registration No:** 12526856 | **Roll No:** 29  

---

This index maps every academic topic in the project to its **exact file**, **function name**, and **line number range**.

---

## 📌 Quick Topic Navigation Map

| Academic Topic | Primary File | Primary Function / Section |
| :--- | :--- | :--- |
| **1. Dataset Ingestion & Multi-Table Joins** | `src/data_loader.py` | `load_merged_train()` |
| **2. Exploratory Data Analysis & Statistics** | `src/data_loader.py` | `dataset_summary()` |
| **3. Temporal & Calendar Feature Engineering** | `src/preprocessing.py` | `engineer_features()` |
| **4. Missing Value Treatment (Imputation)** | `src/preprocessing.py` | `engineer_features()` |
| **5. Outlier & Return Sales Handling** | `src/preprocessing.py` | `engineer_features()` (`IsReturn`) |
| **6. Categorical One-Hot Encoding** | `src/preprocessing.py` | `engineer_features()` (`pd.get_dummies`) |
| **7. Chronological Validation Split (No Shuffling)** | `src/preprocessing.py` | `temporal_split()` |
| **8. Feature Scaling (StandardScaler)** | `src/preprocessing.py` | `fit_scaler()`, `apply_scaler()` |
| **9. Linear Regression (Baseline Supervised Model)** | `src/models.py` | `build_linear_regression()`, `train_linear_regression()` |
| **10. Decision Tree Regression (Regularization)** | `src/models.py` | `build_decision_tree()`, `train_decision_tree()` |
| **11. Model Persistence (Serialization)** | `src/models.py` | `save_all()`, `load_all()` |
| **12. Regression Metrics (MAE, RMSE, R²)** | `src/evaluation.py` | `compute_metrics()` |
| **13. Holiday vs Non-Holiday Subgroup Analysis** | `src/evaluation.py` | `evaluate_model()` |
| **14. End-to-End Training Orchestration** | `train.py` | `main()` |
| **15. Single-Point & Multi-Week Forecasting** | `src/forecasting.py` | `predict()`, `predict_range()` |
| **16. Decision-Support Staffing Demand Module** | `src/staffing.py` | `estimate_staffing()`, `staffing_from_forecast_df()` |
| **17. Interactive Web Dashboard UI** | `app.py` | Streamlit multi-page router & reactive calculations |

---

## 🔍 Detailed Topic-by-Topic Walkthrough

### 1. Data Ingestion & Relational Merging
* **File:** [`src/data_loader.py`](../src/data_loader.py)
* **Header Tag:** `[TOPIC: Relational Dataset Merging (Left Joins on Store & Date)]`
* **Function:** `load_merged_train()`
* **What to Show:**
  * Shows how `train.csv` (421,570 records) is left-joined with `stores.csv` on `Store`.
  * Shows how `features.csv` is merged on compound key `['Store', 'Date']`.
  * Safe dropping of duplicate `IsHoliday` column from `features.csv`.

---

### 2. Missing Value Imputation
* **File:** [`src/preprocessing.py`](../src/preprocessing.py)
* **Header Tag:** `[SUB-TOPIC: Missing Value Treatment — Promotional MarkDowns]` and `[SUB-TOPIC: Missing Value Treatment — Store-Grouped Median Imputation]`
* **Function:** `engineer_features(df)`
* **What to Show:**
  1. **MarkDown1–5:** Filled with `0.0` while creating binary presence flags (`MarkDownX_flag = 1/0`) because promotional tracking was only added by Walmart in November 2011.
  2. **CPI & Unemployment:** Filled using **store-specific medians** (`df.groupby("Store")[col].transform("median")`), preventing global distortion.

---

### 3. Feature Engineering & Temporal Extraction
* **File:** [`src/preprocessing.py`](../src/preprocessing.py)
* **Header Tag:** `[SUB-TOPIC: Temporal & Calendar Feature Extraction]`
* **Function:** `engineer_features(df)`
* **What to Show:**
  * Extracts calendar signals: `Year`, `Month`, `WeekOfYear`, `Quarter`, `DayOfYear`.
  * Implements `Season` mapping: Winter (1), Spring (2), Summer (3), Autumn (4).
  * Binary conversion of `IsHoliday` to `IsHoliday_int`.

---

### 4. Categorical Feature Encoding (One-Hot Encoding)
* **File:** [`src/preprocessing.py`](../src/preprocessing.py)
* **Header Tag:** `[SUB-TOPIC: Feature Encoding — One-Hot Categorical Encoding]`
* **Function:** `engineer_features(df)`
* **What to Show:**
  * Shows `pd.get_dummies(df["Type"], prefix="StoreType", drop_first=False, dtype=int)`.
  * Converts store categories A, B, and C into distinct binary columns without implicit ordinal bias.

---

### 5. Time-Series Chronological Train-Validation Split
* **File:** [`src/preprocessing.py`](../src/preprocessing.py)
* **Header Tag:** `[TOPIC: Chronological Validation Split (Time-Series Train-Test Split)]`
* **Function:** `temporal_split(df, val_fraction=0.20)`
* **What to Show:**
  * **Zero Random Shuffling:** Demonstrates that the last 20% of unique dates (from `2012-04-20` onwards) are reserved strictly for validation.
  * Ensures past data predicts future data, preventing temporal data leakage.

---

### 6. Feature Scaling (StandardScaler without Data Leakage)
* **File:** [`src/preprocessing.py`](../src/preprocessing.py)
* **Header Tag:** `[TOPIC: Feature Scaling (StandardScaler — Fit on Train Only)]`
* **Functions:** `fit_scaler(X_train)` and `apply_scaler(scaler, X)`
* **What to Show:**
  * Shows `scaler.fit(X_train)` is called **strictly on the training split**.
  * Shows validation data `X_val` is only transformed (`scaler.transform(X_val)`), preserving true out-of-sample evaluation.

---

### 7. Linear Regression Baseline Model
* **File:** [`src/models.py`](../src/models.py)
* **Header Tag:** `[TOPIC: Supervised Learning — Linear Regression (Baseline Parametric Model)]`
* **Functions:** `build_linear_regression()`, `train_linear_regression()`
* **What to Show:**
  * Fitted on scaled features (`X_train_scaled`).
  * Acts as the simple, interpretable parametric baseline ($R^2 \approx 0.0968$).

---

### 8. Decision Tree Regressor & Overfitting Regularization
* **File:** [`src/models.py`](../src/models.py)
* **Header Tag:** `[TOPIC: Supervised Learning — Decision Tree Regression & Overfitting Control]`
* **Functions:** `build_decision_tree()`, `train_decision_tree()`
* **What to Show:**
  * Non-parametric tree model capturing nonlinear feature interactions.
  * **Overfitting Hyperparameters:**
    * `max_depth=12`: Prevents unbounded tree depth and memorization.
    * `min_samples_leaf=10`: Ensures leaf nodes have statistical mass.
    * `random_state=42`: Guarantees reproducible splits.
  * Achieves high validation accuracy ($R^2 = 0.9186$).

---

### 9. Regression Evaluation Metrics (MAE, RMSE, R²)
* **File:** [`src/evaluation.py`](../src/evaluation.py)
* **Header Tag:** `[TOPIC: Regression Metrics Formulation (MAE, RMSE, R² Score)]`
* **Function:** `compute_metrics(y_true, y_pred)`
* **What to Show:**
  * Computes Mean Absolute Error (`mean_absolute_error`).
  * Computes Root Mean Squared Error (`np.sqrt(mean_squared_error)`).
  * Computes Coefficient of Determination ($R^2$ via `r2_score`).

---

### 10. Holiday vs Non-Holiday Performance Gap Analysis
* **File:** [`src/evaluation.py`](../src/evaluation.py)
* **Header Tag:** `[TOPIC: Subgroup Evaluation & Holiday Performance Gap Analysis]`
* **Function:** `evaluate_model(y_true, y_pred, is_holiday, model_name)`
* **What to Show:**
  * Subsets test data by `holiday_mask = is_holiday.astype(bool)`.
  * Computes separate MAE, RMSE, and $R^2$ for holiday weeks vs non-holiday weeks.
  * Quantifies the performance gap: $\text{MAE}_{\text{holiday}} - \text{MAE}_{\text{non-holiday}}$.

---

### 11. Decision-Support Staffing Demand Module
* **File:** [`src/staffing.py`](../src/staffing.py)
* **Header Tag:** `[TOPIC: Sales-to-Labor Mathematical Formulation (FTE Staffing Estimation)]`
* **Function:** `estimate_staffing(forecasted_sales, sales_per_hour, hours_per_week, min_staff)`
* **What to Show:**
  * Formula 1: $\text{Labour Hours} = \frac{\text{Forecasted Sales}}{\text{Sales per Associate-Hour}}$ (Default: \$300/hr)
  * Formula 2: $\text{FTE Staff} = \frac{\text{Labour Hours}}{\text{Working Hours per Week}}$ (Default: 40 hrs/wk)
  * Formula 3: $\text{Recommended Staff} = \max(\text{min\_staff}, \text{round}(\text{FTE Staff}))$
  * Emphasizes that this is a configurable operational estimate, not employee ground-truth.

---

### 12. End-to-End Pipeline CLI Execution
* **File:** [`train.py`](../train.py)
* **Header Tag:** Stages 1 through 5
* **Function:** `main()`
* **What to Show:**
  * Orchestrates the entire pipeline: Ingestion $\rightarrow$ Preprocessing $\rightarrow$ LR Training $\rightarrow$ DT Training $\rightarrow$ Evaluation $\rightarrow$ Artifact Persistence (`models/`, `outputs/evaluation_results.json`).

---

### 13. Interactive Dashboard & Visualizations
* **File:** [`app.py`](../app.py)
* **What to Show:**
  * Built using Streamlit + Plotly.
  * Theme-adaptive CSS for both light and dark mode.
  * Reactive Staffing Estimator calculator (auto-calculates live on any input change).
  * Historical vs forecast interactive trend chart with holiday diamonds.
