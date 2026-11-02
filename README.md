# Retail Store Weekly Sales Forecasting and Staffing Demand Analysis System

> **CSE252 – Artificial Intelligence and Machine Learning | Individual Project**  
> **Student:** Rehan Khan | **Roll No.:** 29 | **Reg. No.:** 12526856 | **Batch:** K3P25RA  
> **Problem ID:** KAGGLE-WALMART-SALES

---

## 📋 Project Description
An end-to-end interactive Machine Learning application that forecasts weekly retail sales for Walmart stores using genuine historical data from the Kaggle Walmart Recruiting competition. The system compares **Linear Regression** and **Decision Tree Regression**, evaluates performance on holiday vs. non-holiday weeks, and translates forecasts into practical staffing recommendations through an interactive Streamlit dashboard.

---

## 🎯 Objectives
- Compare Linear Regression (baseline) vs. Decision Tree Regression (nonlinear) on 421,570 weekly sales observations.
- Evaluate forecasting accuracy using MAE, RMSE, and R² — separately for holiday and non-holiday weeks.
- Build an interactive retail analytics dashboard with forecasting, staffing planning, and EDA capabilities.
- Use strict chronological validation to prevent data leakage.

---

## ✨ Key Features
- ✅ **Genuine Walmart dataset** — no fabricated data.
- ✅ **Temporal (chronological) train-test split** — no random shuffling.
- ✅ **Holiday vs. Non-Holiday evaluation** — quantified performance gap.
- ✅ **Interactive Streamlit dashboard** with 8 navigation pages.
- ✅ **Configurable staffing estimates** with clear assumption labelling.
- ✅ **Forecast range** — predict multiple weeks in one click.
- ✅ **Multi-week staffing planner** with charts and tables.

---

## 🛠️ Technology Stack
| Component | Technology |
|:---|:---|
| Language | Python 3.13 |
| Data Processing | Pandas 3.1, NumPy 2.5 |
| Machine Learning | Scikit-learn 1.9 |
| Dashboard | Streamlit 1.66 |
| Visualisation | Plotly 7.1, Matplotlib 3.11, Seaborn 0.13 |
| Model Persistence | Joblib 1.6 |
| Version Control | Git 2.55 |

---

## 📦 Dataset
**Source:** [Kaggle — Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)

| File | Records | Description |
|:---|:---|:---|
| `train.csv` | 421,570 | Historical weekly sales by store/department |
| `features.csv` | 8,190 | Store-level external data (CPI, Unemployment, MarkDowns, etc.) |
| `stores.csv` | 45 | Store metadata (Type, Size) |
| `test.csv` | 115,064 | Future prediction schedule (no sales labels) |

### Dataset Setup
1. Download from the Kaggle competition page (requires free Kaggle account + rule acceptance).
2. Extract and place all CSV files into `Walmart_Sales_Forecasting/data/`.

> **Note:** The dataset is **not** committed to GitHub (size/privacy). You must download it separately.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/RehanK09/retail-sales-staffing-forecast.git
cd retail-sales-staffing-forecast
```

### 2. Create virtual environment and install dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Place the dataset files
Place `train.csv`, `features.csv`, `stores.csv`, `test.csv` inside the `data/` directory.

---

## 🚀 Running the Project

### Train the models
```powershell
.\.venv\Scripts\python.exe train.py
```
This will:
- Load and merge all datasets.
- Engineer 29 features (temporal, store, markdown, external indicators).
- Perform chronological train/validation split (80/20 by date).
- Train Linear Regression and Decision Tree models.
- Save models to `models/` and evaluation results to `outputs/evaluation_results.json`.

### Launch the dashboard
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 📊 Model Results (Validation Set — Chronological Hold-out)

| Model | MAE | RMSE | R² |
|:---|:---|:---|:---|
| **Linear Regression** | \$14,422 | \$20,877 | 0.0968 |
| **Decision Tree** | \$3,352 | \$6,266 | **0.9186** |

### Holiday vs. Non-Holiday Performance

| Model | Holiday MAE | Non-Holiday MAE | Gap |
|:---|:---|:---|:---|
| Linear Regression | \$15,156 | \$14,395 | +\$761 |
| Decision Tree | \$3,534 | \$3,346 | **+\$189** |

> **Interpretation:** The Decision Tree significantly outperforms Linear Regression (R² 0.92 vs 0.10). Both models show slightly higher error on holiday weeks due to unusual demand spikes. The Decision Tree handles this much better.

---

## 📁 Project Structure
```text
Walmart_Sales_Forecasting/
│
├── app.py               # Streamlit dashboard (8-page interactive application)
├── train.py             # End-to-end model training script
├── requirements.txt     # Python dependencies
├── README.md
├── PROJECT_LINKS.txt
│
├── src/
│   ├── data_loader.py       # Load and merge CSV datasets
│   ├── preprocessing.py     # Feature engineering + chronological split
│   ├── models.py            # Model definitions and persistence
│   ├── evaluation.py        # MAE / RMSE / R² evaluation (holiday breakdowns)
│   ├── forecasting.py       # Single & range prediction using saved models
│   └── staffing.py          # Sales-to-staffing estimation logic
│
├── data/                # Dataset CSVs (not committed — download from Kaggle)
├── models/              # Saved model artifacts (.joblib files)
├── outputs/             # evaluation_results.json and prediction outputs
├── screenshots/         # Dashboard screenshots
└── docs/                # Phase handover reports and dataset inspection
```

---

## ⚠️ Limitations
- The Kaggle test set has no actual sales labels; evaluation uses a chronological hold-out from training data only.
- MarkDown promotional features (MarkDown1–5) were only collected from November 2011 onward.
- Staffing estimates are based on configurable assumptions — not actual Walmart employee records.
- Linear Regression performs significantly worse on this dataset due to non-linear sales patterns.
- No deep learning or ensemble methods are used (per course scope).

---

## 👤 Author
**Rehan Khan** | Roll No. 29 | Reg. No. 12526856  
B.Tech CSE — Batch K3P25RA  
CSE252 – Artificial Intelligence and Machine Learning  
Email: rehankhanty8349@gmail.com  
GitHub: [RehanK09](https://github.com/RehanK09)
