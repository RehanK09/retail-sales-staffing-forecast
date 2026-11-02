# Phase 0: Project Initialization & Environment Handover Report

**Project Title:** Retail Store Weekly Sales Forecasting and Staffing Demand Analysis System  
**Course:** CSE252 – Artificial Intelligence and Machine Learning  
**Student:** Rehan Khan  
**Registration Number:** 12526856  
**Roll Number:** 29  
**Batch:** K3P25RA  
**Problem Statement ID:** KAGGLE-WALMART-SALES  
**Target Platform:** Streamlit Community Cloud / Local Python 3.13  
**Date:** November 2, 2026  
**Status:** **PHASE 0 FULLY COMPLETED & VERIFIED — READY FOR PHASE 1 IMPLEMENTATION**

---

## 1. Environment Details
- **Operating System:** Windows 11 (64-bit)
- **Base Python Version:** Python 3.13.15
- **Git Version:** 2.55.0.windows.5
- **Configured Git User:** `RehanK09` (`rehankhanty8349@gmail.com`)
- **Package Manager:** pip 26.2.1
- **GitHub CLI (`gh`):** Version 2.103.3 (installed at `C:\Program Files\GitHub CLI\gh.exe`)
- **Git Credential Manager:** Version 2.9.1

---

## 2. Python Virtual Environment & Verified Dependencies
A dedicated virtual environment is active at:
`Walmart_Sales_Forecasting\.venv`

### Operational Package Inventory
| Package | Installed Version | Verification Status | Primary Role |
| :--- | :--- | :--- | :--- |
| `pandas` | 3.1.0 | Verified | Data manipulation, date parsing, tabular processing |
| `numpy` | 2.5.4 | Verified | Numerical computations and vector operations |
| `scikit-learn` | 1.9.1 | Verified | Classical ML (Linear Regression, Decision Tree), metrics |
| `streamlit` | 1.66.1 | Verified | Web application framework for interactive dashboard |
| `plotly` | 7.1.0 | Verified | Interactive charts for sales and staffing visualizations |
| `joblib` | 1.6.0 | Verified | Model serialization and deserialization |
| `matplotlib` | 3.11.2 | Verified | Static plotting, exploratory visual analysis |
| `seaborn` | 0.13.2 | Verified | Statistical distribution and correlation plots |
| `kaggle` | 2.2.5 | Verified | Kaggle dataset CLI integration |
| `six` | 1.17.0 | Verified | Compatibility helper for Kaggle SDK |

All dependencies import cleanly with zero missing modules or deprecation errors.

---

## 3. Dataset Download & Integrity Verification Status
- **Acquisition Strategy:** Manual Kaggle Download (completed by student).
- **Original Archives:** Preserved in `walmart-recruiting-store-sales-forecasting/` (excluded in `.gitignore`).
- **Active Working Datasets:** Extracted and validated in `Walmart_Sales_Forecasting/data/`.
- **Git Protection:** `data/*.csv` added to `.gitignore` to prevent raw Kaggle datasets from being pushed to GitHub.

### Verified Dataset Inventory (Pandas Read Test Passed)
| File Name | File Size | Dimensions | Key Columns | Date Range | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train.csv` | 12.25 MB | 421,570 × 5 | `Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday` | 2010-02-05 to 2012-10-26 | 0 nulls, 0 duplicates. 45 stores, 81 depts. 1,285 negative sales rows (returns). |
| `features.csv` | 0.56 MB | 8,190 × 12 | `Store`, `Date`, `Temperature`, `Fuel_Price`, `MarkDown1-5`, `CPI`, `Unemployment`, `IsHoliday` | 2010-02-05 to 2013-07-26 | 0 duplicates. MarkDowns missing prior to Nov 2011 (standard). |
| `stores.csv` | 532 B | 45 × 3 | `Store`, `Type`, `Size` | N/A | Types: A (22), B (17), C (6). Sizes: 34,875 to 219,622 sq ft. |
| `test.csv` | 2.48 MB | 115,064 × 4 | `Store`, `Dept`, `Date`, `IsHoliday` | 2012-11-02 to 2013-07-26 | Future evaluation period. |
| `sampleSubmission.csv` | 2.05 MB | 115,064 × 2 | `Id`, `Weekly_Sales` | N/A | Kaggle benchmark format. |

Full dataset inspection details are saved in `docs/dataset_inspection.txt`.

---

## 4. Git & GitHub Repository Status
- **Local Git Repository:** Initialized successfully in `Walmart_Sales_Forecasting/`.
- **Default Branch:** `master`
- **Remote Origin URL:** `https://github.com/RehanK09/retail-sales-staffing-forecast.git`
- **Remote Verification:** Tested with `git ls-remote origin` — **Successfully Connected & Accessible (Exit Code 0)**.
- **Repository Visibility:** Public (verified on GitHub).
- **Staging / Commits Status:** 0 commits made (clean state; no premature commits or unfinished code).
- **Data Leak Prevention:** `.gitignore` excludes:
  - Python virtual environment (`.venv/`)
  - Original archives directory (`walmart-recruiting-store-sales-forecasting/`)
  - Raw CSV datasets (`data/*.csv`)
  - Cache, logs, credentials (`kaggle.json`, `*.log`, `__pycache__/`)
- **GitHub Authentication:**
  - Git Credential Manager (GCM v2.9.1) is enabled on Windows. When ready to push the initial finished commit, Git will prompt for browser-based authorization automatically.
  - Alternatively, GitHub CLI can be authenticated with one simple command:
    ```powershell
    & "C:\Program Files\GitHub CLI\gh.exe" auth login
    ```
    Select: `GitHub.com` -> `HTTPS` -> `Yes` (authenticate Git with GitHub credentials) -> `Login with a web browser`.

---

## 5. Verified Project Directory Structure
```text
Walmart_Sales_Forecasting/
│
├── .git/                      # Local Git repository (remote: origin -> retail-sales-staffing-forecast)
├── .gitignore                 # Ignore rules (venv, logs, archives, data/*.csv, credentials)
├── .venv/                     # Python 3.13 isolated virtual environment
├── app.py                     # Dashboard entry point (placeholder)
├── requirements.txt           # Project dependencies
├── README.md                  # Project overview and run guidelines
│
├── data/                      # Active working CSV dataset files (git-ignored)
│   ├── .gitkeep               # Preserves folder in Git without committing CSVs
│   ├── features.csv
│   ├── sampleSubmission.csv
│   ├── stores.csv
│   ├── test.csv
│   └── train.csv
│
├── src/                       # Core Python modules (to be built in Phase 1)
│   ├── __init__.py
│   └── .gitkeep
├── models/                    # Serialized trained model binaries (.joblib)
│   └── .gitkeep
├── outputs/                   # Metric outputs, comparison reports, evaluation graphs
│   └── .gitkeep
├── screenshots/               # Dashboard screenshots for report and presentation
│   └── .gitkeep
└── docs/                      # Handover report, dataset notes, and academic documentation
    ├── PHASE_0_HANDOVER.md    # Master Phase 0 handover document
    ├── dataset_inspection.txt # Detailed dataset inspection log
    └── inspect_dataset.py     # Verification script
```

---

## 6. Approved Operational & Academic Parameters
- **Project Title:** Retail Store Weekly Sales Forecasting and Staffing Demand Analysis System
- **Algorithm Comparison:** Linear Regression (Baseline) vs. Decision Tree Regression (Nonlinear).
- **Validation Methodology:** Strict chronological / temporal splitting (no random shuffling).
- **Staffing Estimation Method:** Configurable operational assumptions:
  $$\text{Estimated Labour Hours} = \frac{\text{Forecasted Weekly Sales}}{\text{Expected Sales per Associate-Hour}}$$
  $$\text{Estimated FTE Staff} = \frac{\text{Estimated Labour Hours}}{\text{Weekly Working Hours per Associate (e.g., 40 hrs)}}$$

---

## 7. Exact Next Steps for Phase 1 (Main Development Model)
With all prerequisites, data files, virtual environment, and public repository verified, Phase 1 can now begin:

1. **`src/data_preprocessing.py`:**
   - Merge `train.csv`, `stores.csv`, and `features.csv`.
   - Feature engineering: Year, Month, Week of Year, Quarter, IsHoliday encoding, Store Type one-hot encoding, MarkDown missing value imputation (median/zero-fill with indicator), negative return sales treatment.
   - Enforce temporal train-test split (train on earliest period, test on subsequent holdout period).
2. **`src/models.py` & `src/evaluation.py`:**
   - Train Linear Regression and Decision Tree Regressor.
   - Calculate MAE, RMSE, and $R^2$ scores across overall test set.
   - Calculate Holiday vs. Non-Holiday breakdown and quantify the holiday error gap.
   - Serialize models to `models/linear_regression.joblib` and `models/decision_tree.joblib`.
3. **`src/staffing.py`:**
   - Implement the parameterized staffing analysis functions.
4. **`app.py`:**
   - Build the interactive Streamlit decision-support application.
5. **Documentation & University Deliverables:**
   - Update README, create academic report, generate presentation, record `PROJECT_LINKS.txt`, and package into `CSE252_12526856.zip`.
