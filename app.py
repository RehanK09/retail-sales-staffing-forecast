"""
app.py — Retail Store Weekly Sales Forecasting & Staffing Demand Analysis System
CSE252 – Artificial Intelligence and Machine Learning | Individual Project

Author : Rehan Khan
Reg No : 12526856 | Roll No: 29 | Batch: K3P25RA
Dataset: Kaggle Walmart Recruiting – Store Sales Forecasting (2010-02-05 to 2012-10-26)

Note: All forecasts are model predictions based on the historical training dataset.
They represent decision-support estimates and not live 2026 operational records.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import load_merged_train, load_stores, load_features
from src.preprocessing import engineer_features
from src.models import models_exist, load_all
from src.evaluation import load_results
from src.staffing import (
    estimate_staffing,
    staffing_from_forecast_df,
    staffing_recommendation,
    DEFAULT_SALES_PER_HOUR,
    DEFAULT_HOURS_PER_WEEK,
)
from src.forecasting import predict, predict_range

# ─────────────────────────────────────────────────────────────────────────────
# Page Config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Sales Forecasting | CSE252",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_START = pd.Timestamp("2010-02-05")
DATA_END   = pd.Timestamp("2012-10-26")

PAGES = ["Overview", "Sales Forecast", "Model Performance", "Staffing Analysis",
         "Holiday Analysis", "Dataset Explorer", "Inventory Insights"]

# ─────────────────────────────────────────────────────────────────────────────
# Full CSS — no sidebar, clean horizontal nav, high-contrast cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide sidebar completely ───────────────────────────────────────────── */
[data-testid="stSidebar"]          { display: none !important; }
[data-testid="collapsedControl"]   { display: none !important; }
button[kind="header"]              { display: none !important; }

/* ── App background ────────────────────────────────────────────────────── */
.stApp { background: #F7F9FC; }
.stApp [data-testid="stAppViewContainer"] { background: #F7F9FC; }

/* ── Top navigation bar ────────────────────────────────────────────────── */
.top-nav {
    background: #FFFFFF;
    border-bottom: 2px solid #E3E8F0;
    padding: 0 24px;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.nav-brand {
    font-size: 1.1rem;
    font-weight: 800;
    color: #1565C0;
    padding: 14px 16px 14px 0;
    border-right: 2px solid #E3E8F0;
    margin-right: 10px;
    white-space: nowrap;
    letter-spacing: -0.3px;
}
.nav-brand span { color: #37474F; font-weight: 400; font-size: 0.75rem; display: block; }
.nav-btn {
    background: transparent;
    border: none;
    color: #546E7A;
    padding: 10px 16px;
    font-size: 0.88rem;
    font-weight: 600;
    border-radius: 6px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s ease;
}
.nav-btn:hover { background: #EEF2FF; color: #1565C0; }
.nav-btn.active { background: #1565C0; color: #FFFFFF; }
.nav-status {
    margin-left: auto;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    white-space: nowrap;
}
.nav-status.ok  { background: #E8F5E9; color: #1B5E20; border: 1px solid #A5D6A7; }
.nav-status.err { background: #FFEBEE; color: #B71C1C; border: 1px solid #EF9A9A; }

/* ── Page heading ──────────────────────────────────────────────────────── */
.page-heading {
    font-size: 1.75rem;
    font-weight: 800;
    color: #1A237E;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.page-subhead {
    font-size: 0.85rem;
    color: #546E7A;
    margin-bottom: 20px;
}

/* ── KPI cards ─────────────────────────────────────────────────────────── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E3E8F0;
    border-left: 4px solid #1976D2;
    border-radius: 10px;
    padding: 16px 18px 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.kpi-card.green  { border-left-color: #2E7D32; }
.kpi-card.amber  { border-left-color: #F57F17; }
.kpi-card.red    { border-left-color: #C62828; }
.kpi-card.purple { border-left-color: #6A1B9A; }
.kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #546E7A;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: #1A237E;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #78909C;
    margin-top: 5px;
}

/* ── Section headers ───────────────────────────────────────────────────── */
.sec-hdr {
    font-size: 1.0rem;
    font-weight: 700;
    color: #1565C0;
    border-bottom: 2px solid #BBDEFB;
    padding-bottom: 5px;
    margin: 22px 0 14px 0;
}

/* ── Divider ───────────────────────────────────────────────────────────── */
.thin-hr { border: none; border-top: 1px solid #E3E8F0; margin: 18px 0; }

/* ── Info banner ────────────────────────────────────────────────────────── */
.info-banner {
    background: #E3F2FD;
    border: 1px solid #90CAF9;
    border-left: 4px solid #1565C0;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.84rem;
    color: #0D47A1;
    margin-bottom: 18px;
}
.warn-banner {
    background: #FFF8E1;
    border: 1px solid #FFD54F;
    border-left: 4px solid #F57F17;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.84rem;
    color: #4E342E;
    margin-bottom: 18px;
}

/* ── Streamlit element overrides ─────────────────────────────────────────── */
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
div[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.84rem;
}

/* ── Force main content full width ─────────────────────────────────────── */
.block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State — navigation
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Overview"


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_dataset():
    return load_merged_train()

@st.cache_data(show_spinner=False)
def get_engineered():
    return engineer_features(get_dataset())

@st.cache_data(show_spinner=False)
def get_stores_df():
    return load_stores()

@st.cache_data(show_spinner=False)
def get_eval_results():
    return load_results()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def kpi(label: str, value: str, sub: str = "", color: str = ""):
    cls = f"kpi-card {color}".strip()
    st.markdown(f"""
    <div class="{cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def sec(title: str):
    st.markdown(f'<div class="sec-hdr">{title}</div>', unsafe_allow_html=True)

def fmt_c(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"${val:,.2f}"

def safe_forecast(store, dept, date, is_holiday):
    try:
        res = predict(store, dept, date, is_holiday)
        return res["lr_pred"], res["dt_pred"]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def info(text: str):
    st.markdown(f'<div class="info-banner">ℹ️ {text}</div>', unsafe_allow_html=True)

def warn(text: str):
    st.markdown(f'<div class="warn-banner">⚠️ {text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOP NAVIGATION BAR — rendered as HTML + Streamlit buttons
# ─────────────────────────────────────────────────────────────────────────────
models_ok = models_exist()
eval_res  = get_eval_results()

status_cls  = "ok"  if models_ok else "err"
status_text = "✓ Models Active" if models_ok else "⚠ Models Missing"

# Brand area
st.markdown(f"""
<div class="top-nav">
  <div class="nav-brand">🛒 Retail Sales AI<span>CSE252 · Rehan Khan · 12526856</span></div>
</div>
""", unsafe_allow_html=True)

# Nav buttons in columns  (8 cols: 7 pages + 1 status pill)
_cols = st.columns([1.1, 1.1, 1.3, 1.4, 1.3, 1.3, 1.3, 1])
_page_labels = [
    ("Overview",          "🏠 Overview",          _cols[0]),
    ("Sales Forecast",    "🔮 Forecast",           _cols[1]),
    ("Model Performance", "📊 Models",             _cols[2]),
    ("Staffing Analysis", "👥 Staffing",           _cols[3]),
    ("Holiday Analysis",  "🎄 Holidays",           _cols[4]),
    ("Dataset Explorer",  "🗃️ Dataset",            _cols[5]),
    ("Inventory Insights","📦 Inventory",          _cols[6]),
]

for page_id, label, col in _page_labels:
    with col:
        is_active = st.session_state.page == page_id
        btn_type  = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_{page_id}", type=btn_type, use_container_width=True):
            st.session_state.page = page_id
            st.rerun()

with _cols[7]:
    st.markdown(
        f'<div class="nav-status {status_cls}" style="margin-top:6px;">{status_text}</div>',
        unsafe_allow_html=True
    )

st.markdown('<hr class="thin-hr">', unsafe_allow_html=True)

current = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if current == "Overview":
    df = get_dataset()

    st.markdown('<div class="page-heading">🛒 Retail Sales Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subhead">Walmart Recruiting Dataset · {DATA_START.date()} to {DATA_END.date()} · 45 Stores · 81 Departments · Historical ML Predictions</div>',
        unsafe_allow_html=True
    )

    if not models_ok:
        st.error("⚠️ Trained models not found. Please run `python train.py` first.")
        st.stop()

    # ── Quick forecast controls ───────────────────────────────────────────
    sec("⚡ Quick Forecast")
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1.4, 0.9, 1.1])
    with c1:
        q_store = st.selectbox("Store", sorted(df["Store"].unique()), key="ov_store")
    with c2:
        q_dept  = st.selectbox("Department", sorted(df["Dept"].unique()), key="ov_dept")
    with c3:
        q_date  = st.date_input("Forecast Week (Ending Friday)",
                                value=pd.Timestamp("2012-11-02").date(),
                                min_value=pd.Timestamp("2010-01-01").date(),
                                max_value=pd.Timestamp("2014-12-31").date(),
                                key="ov_date")
    with c4:
        q_hol   = st.checkbox("Holiday Week", value=False, key="ov_hol")
    with c5:
        q_model = st.radio("Model", ["Decision Tree", "Linear Regression"],
                           key="ov_mdl", help="Decision Tree R²=0.9186 · LR R²=0.0968")

    target_date  = pd.Timestamp(q_date)
    lr_w, dt_w   = safe_forecast(q_store, q_dept, target_date, q_hol)
    if lr_w is None:
        st.stop()

    fc_week = dt_w if "Decision" in q_model else lr_w

    month_weeks = pd.date_range(target_date, periods=4, freq="W-FRI")
    month_sum_dt = month_sum_lr = 0.0
    for w in month_weeks:
        lw, dw = safe_forecast(q_store, q_dept, w, q_hol)
        if dw is not None:
            month_sum_dt += dw
            month_sum_lr += lw
    fc_month = month_sum_dt if "Decision" in q_model else month_sum_lr

    hist_sub = df[(df["Store"] == q_store) & (df["Dept"] == q_dept)]
    hist_avg  = hist_sub["Weekly_Sales"].mean() if not hist_sub.empty else 0.0
    diff_pct  = ((fc_week - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0.0

    # ── 5 KPI cards ───────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi("Next Week Predicted Sales", fmt_c(fc_week),
            f"Store {q_store} · Dept {q_dept} · {target_date.date()}")
    with k2:
        kpi("Next Month Est. Demand", fmt_c(fc_month),
            f"4-week total from {target_date.date()}", "green")
    with k3:
        dc = "green" if diff_pct >= 0 else "red"
        kpi("vs Historical Average", f"{diff_pct:+.1f}%",
            f"Hist Avg: {fmt_c(hist_avg)}/wk", dc)
    with k4:
        kpi("Best Model R²", "0.9186", "Decision Tree · Chronological Validation", "green")
    with k5:
        hol_txt = "🎄 Holiday Week" if q_hol else "Regular Week"
        kpi("Week Classification", hol_txt, "Walmart Calendar", "amber" if q_hol else "")

    # ── Historical vs Predicted Trend ─────────────────────────────────────
    sec("📈 Historical Sales vs Model Forecast")
    hist_sorted = hist_sub.sort_values("Date")
    fc_range    = predict_range(q_store, q_dept, target_date, target_date + pd.Timedelta(weeks=7))
    pred_col    = "DT_Prediction" if "Decision" in q_model else "LR_Prediction"

    fig = go.Figure()
    if not hist_sorted.empty:
        fig.add_trace(go.Scatter(
            x=hist_sorted["Date"], y=hist_sorted["Weekly_Sales"],
            name="Historical Sales", mode="lines",
            line=dict(color="#90A4AE", width=1.5),
            fill="tozeroy", fillcolor="rgba(144,164,174,0.07)",
        ))
        hol_pts = hist_sorted[hist_sorted["IsHoliday"]]
        if not hol_pts.empty:
            fig.add_trace(go.Scatter(
                x=hol_pts["Date"], y=hol_pts["Weekly_Sales"],
                name="Holiday Weeks", mode="markers",
                marker=dict(color="#E53935", size=6, symbol="diamond"),
            ))
    fig.add_trace(go.Scatter(
        x=fc_range["Date"], y=fc_range[pred_col],
        name=f"{q_model} Forecast", mode="lines+markers",
        line=dict(color="#1565C0", width=2.5, dash="dash"),
        marker=dict(size=7, color="#1565C0"),
    ))
    fig.add_trace(go.Scatter(
        x=[target_date], y=[fc_week],
        name="Target Week", mode="markers+text",
        text=[f"  {fmt_c(fc_week)}"],
        textposition="top right",
        marker=dict(size=14, color="#0D47A1", symbol="star"),
    ))
    fig.add_vline(x=DATA_END, line_color="#FB8C00", line_dash="dot", line_width=1.5)
    fig.add_annotation(x=DATA_END, y=0.05, yref="paper",
                       text="Dataset End (Oct 2012)", showarrow=False,
                       font=dict(color="#FB8C00", size=10))
    fig.update_layout(
        height=380, margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Timeline", yaxis_title="Weekly Sales ($)",
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", y=1.08, x=0),
        font=dict(color="#37474F"),
        xaxis=dict(gridcolor="#F0F4F8"), yaxis=dict(gridcolor="#F0F4F8"),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption("Dashed blue line = model forecast beyond historical boundary (dotted orange).")

    # ── 4-week table ──────────────────────────────────────────────────────
    sec("📅 Next 4-Week Forecast Breakdown")
    rows = []
    for w in month_weeks:
        l, d = safe_forecast(q_store, q_dept, w, q_hol)
        if d is not None:
            rows.append({
                "Week Ending": str(w.date()),
                "Linear Regression": fmt_c(l),
                "Decision Tree": fmt_c(d),
                "Active Forecast": fmt_c(d if "Decision" in q_model else l),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Model performance strip ───────────────────────────────────────────
    if eval_res:
        sec("🏅 Model Performance Summary (Chronological Validation)")
        dt_v = eval_res["Decision Tree"]["overall"]
        lr_v = eval_res["Linear Regression"]["overall"]
        p1, p2, p3, p4 = st.columns(4)
        with p1:  kpi("Decision Tree R²",   f"{dt_v['R2']:.4f}",        "Validation accuracy",    "green")
        with p2:  kpi("Decision Tree MAE",  fmt_c(dt_v["MAE"]),          "Mean Absolute Error",    "green")
        with p3:  kpi("Linear Reg R²",      f"{lr_v['R2']:.4f}",        "Baseline linear model",   "red")
        with p4:  kpi("Linear Reg MAE",     fmt_c(lr_v["MAE"]),          "Mean Absolute Error",     "red")

    # ── Overview charts ───────────────────────────────────────────────────
    sec("📊 Dataset Overview — Store-Wise Average Weekly Sales")
    store_avg = df.groupby("Store")["Weekly_Sales"].mean().reset_index()
    fig_stores = px.bar(
        store_avg, x="Store", y="Weekly_Sales",
        labels={"Weekly_Sales": "Avg Weekly Sales ($)", "Store": "Store ID"},
        color="Weekly_Sales",
        color_continuous_scale=["#90CAF9", "#1565C0"],
    )
    fig_stores.update_layout(
        height=320, margin=dict(t=10, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        coloraxis_showscale=False,
        font=dict(color="#37474F"),
        xaxis=dict(gridcolor="#F0F4F8"), yaxis=dict(gridcolor="#F0F4F8"),
    )
    st.plotly_chart(fig_stores, use_container_width=True, theme=None)

    c_left, c_right = st.columns(2)
    with c_left:
        sec("🗓️ Holiday vs Non-Holiday Weekly Sales")
        hol_comp = df.groupby("IsHoliday")["Weekly_Sales"].mean().reset_index()
        hol_comp["Period"] = hol_comp["IsHoliday"].map({True: "Holiday Weeks", False: "Non-Holiday"})
        fig_h = px.bar(
            hol_comp, x="Period", y="Weekly_Sales",
            color="Period",
            color_discrete_map={"Holiday Weeks": "#E53935", "Non-Holiday": "#1976D2"},
            text_auto=True,
            labels={"Weekly_Sales": "Avg Weekly Sales ($)"},
        )
        fig_h.update_layout(
            height=300, margin=dict(t=10, b=10), showlegend=False,
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(color="#37474F"),
            yaxis=dict(gridcolor="#F0F4F8"),
        )
        st.plotly_chart(fig_h, use_container_width=True, theme=None)

    with c_right:
        sec("📉 Weekly Sales Distribution")
        fig_dist = px.histogram(
            df, x="Weekly_Sales", nbins=80,
            labels={"Weekly_Sales": "Weekly Sales ($)", "count": "Frequency"},
            color_discrete_sequence=["#1976D2"],
        )
        fig_dist.update_layout(
            height=300, margin=dict(t=10, b=10),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(color="#37474F"),
            xaxis=dict(gridcolor="#F0F4F8"), yaxis=dict(gridcolor="#F0F4F8"),
        )
        st.plotly_chart(fig_dist, use_container_width=True, theme=None)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SALES FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif current == "Sales Forecast":
    df = get_dataset()

    st.markdown('<div class="page-heading">🔮 Sales Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subhead">Generate store-department forecasts using trained ML models.</div>',
                unsafe_allow_html=True)
    info(f"Training boundary: {DATA_START.date()} to {DATA_END.date()}. "
         "Predictions beyond Oct 2012 extrapolate learned seasonal patterns.")

    if not models_ok:
        st.error("Trained models missing — run `python train.py` first.")
        st.stop()

    # Controls
    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1.1])
    with c1:
        f_store = st.selectbox("Store ID", sorted(df["Store"].unique()), key="fc_s")
    with c2:
        f_dept  = st.selectbox("Department ID", sorted(df["Dept"].unique()), key="fc_d")
    with c3:
        f_date  = st.date_input("Forecast Week Ending",
                                value=pd.Timestamp("2012-11-02").date(),
                                min_value=pd.Timestamp("2010-01-01").date(),
                                max_value=pd.Timestamp("2014-12-31").date(),
                                key="fc_dt")
    with c4:
        f_model = st.radio("Model", ["Decision Tree", "Linear Regression"], key="fc_mdl")

    f_hol = st.checkbox("Mark as Holiday Week", value=False, key="fc_hol")

    target_dt  = pd.Timestamp(f_date)
    lr_p, dt_p = safe_forecast(f_store, f_dept, target_dt, f_hol)
    if lr_p is None:
        st.stop()

    active_val = dt_p if "Decision" in f_model else lr_p
    range_end  = target_dt + pd.Timedelta(weeks=4)
    range_df   = predict_range(f_store, f_dept, target_dt, range_end)
    month_tot  = range_df["DT_Prediction"].sum() if "Decision" in f_model else range_df["LR_Prediction"].sum()
    hist_data  = df[(df["Store"] == f_store) & (df["Dept"] == f_dept)]
    hist_mean  = hist_data["Weekly_Sales"].mean() if not hist_data.empty else 0.0

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("Target Week Forecast", fmt_c(active_val),
            f"{f_model} · {target_dt.date()}")
    with k2:
        kpi("4-Week Estimated Demand", fmt_c(month_tot),
            "Summed 4-week projection", "green")
    with k3:
        kpi("Historical Dept Average", fmt_c(hist_mean),
            f"Store {f_store} · Dept {f_dept}")

    sec(f"📈 Projection: Store {f_store} · Department {f_dept}")
    fig = go.Figure()
    if not hist_data.empty:
        hd = hist_data.sort_values("Date")
        fig.add_trace(go.Scatter(
            x=hd["Date"], y=hd["Weekly_Sales"],
            name="Historical", mode="lines",
            line=dict(color="#90A4AE", width=1.5),
        ))
    col_pred = "DT_Prediction" if "Decision" in f_model else "LR_Prediction"
    fig.add_trace(go.Scatter(
        x=range_df["Date"], y=range_df[col_pred],
        name=f"{f_model} Forecast", mode="lines+markers",
        line=dict(color="#1565C0", width=2.5, dash="dash"),
        marker=dict(size=7, color="#1565C0"),
    ))
    fig.add_vline(x=DATA_END, line_color="#FB8C00", line_dash="dot", line_width=1.5)
    fig.update_layout(
        height=380, margin=dict(t=10, b=10, l=10, r=10),
        yaxis_title="Weekly Sales ($)",
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#37474F"),
        xaxis=dict(gridcolor="#F0F4F8"), yaxis=dict(gridcolor="#F0F4F8"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    sec("📅 Weekly Breakdown")
    disp = range_df.copy()
    disp["Date"]          = disp["Date"].dt.date.astype(str)
    disp["LR_Prediction"] = disp["LR_Prediction"].apply(fmt_c)
    disp["DT_Prediction"] = disp["DT_Prediction"].apply(fmt_c)
    disp.columns = ["Week Ending", "Linear Regression Forecast", "Decision Tree Forecast"]
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif current == "Model Performance":
    st.markdown('<div class="page-heading">📊 ML Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subhead">Chronological hold-out validation · No random shuffling · 80/20 time split</div>',
                unsafe_allow_html=True)
    info("Evaluation is performed on a strict chronological hold-out set (last 20% of weeks). "
         "Training data never sees future observations — this mirrors real deployment conditions.")

    if not eval_res:
        st.error("Evaluation results missing. Run `python train.py` first.")
        st.stop()

    lr_m = eval_res["Linear Regression"]
    dt_m = eval_res["Decision Tree"]

    sec("📋 Overall Metrics Comparison")
    comp_df = pd.DataFrame({
        "Metric": ["MAE (↓ better)", "RMSE (↓ better)", "R² Score (↑ better)", "Validation Samples"],
        "Linear Regression (Baseline)": [
            fmt_c(lr_m["overall"]["MAE"]),
            fmt_c(lr_m["overall"]["RMSE"]),
            f"{lr_m['overall']['R2']:.4f}",
            f"{lr_m['overall']['n']:,}",
        ],
        "Decision Tree Regressor": [
            fmt_c(dt_m["overall"]["MAE"]),
            fmt_c(dt_m["overall"]["RMSE"]),
            f"{dt_m['overall']['R2']:.4f}",
            f"{dt_m['overall']['n']:,}",
        ],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    chart_data = pd.DataFrame({
        "Model":    ["Linear Regression", "Decision Tree"] * 2,
        "Metric":   ["MAE", "MAE", "RMSE", "RMSE"],
        "Error ($)": [
            lr_m["overall"]["MAE"], dt_m["overall"]["MAE"],
            lr_m["overall"]["RMSE"], dt_m["overall"]["RMSE"],
        ],
    })
    fig_bar = px.bar(
        chart_data, x="Metric", y="Error ($)", color="Model", barmode="group",
        color_discrete_map={"Linear Regression": "#C62828", "Decision Tree": "#1565C0"},
        text_auto=True,
    )
    fig_bar.update_layout(
        height=360, margin=dict(t=10, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#37474F"),
        yaxis=dict(gridcolor="#F0F4F8"),
    )
    st.plotly_chart(fig_bar, use_container_width=True, theme=None)

    sec("🎄 Holiday vs Non-Holiday Breakdown")
    h_rows = []
    for mname, mdata in [("Linear Regression", lr_m), ("Decision Tree", dt_m)]:
        for period, label in [("holiday", "Holiday Weeks"), ("non_holiday", "Non-Holiday")]:
            h_rows.append({
                "Model": mname, "Period": label,
                "MAE ($)": fmt_c(mdata[period]["MAE"]),
                "RMSE ($)": fmt_c(mdata[period]["RMSE"]),
                "R²": f"{mdata[period]['R2']:.4f}",
                "Samples": f"{mdata[period]['n']:,}",
            })
    st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        kpi("LR Holiday MAE Gap", f"+${lr_m['holiday_gap']['MAE_gap']:,.0f}",
            "Holiday vs Non-Holiday error", "red")
    with c2:
        kpi("DT Holiday MAE Gap", f"+${dt_m['holiday_gap']['MAE_gap']:,.0f}",
            "Holiday vs Non-Holiday error", "amber")

    sec("🔬 Validation Methodology")
    st.markdown("""
    | Property | Detail |
    |---|---|
    | Split strategy | Strict chronological — no shuffling |
    | Training period | Feb 2010 → Apr 2012 (≈80%) |
    | Validation period | Apr 2012 → Oct 2012 (≈20%) |
    | Validation records | 82,832 store-dept-week observations |
    | Feature scaling | StandardScaler fitted **only** on training partition |
    | Holiday validation rows | 2,966 (3.6%) |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STAFFING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif current == "Staffing Analysis":
    df = get_dataset()

    st.markdown('<div class="page-heading">👥 Staffing Demand Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subhead">Translate sales forecasts into labor-hour and FTE estimates using operational assumptions.</div>',
                unsafe_allow_html=True)
    warn("ESTIMATES ONLY. This tool uses configurable operational assumptions to translate "
         "forecasted sales into staffing estimates. The Walmart dataset does NOT contain employee "
         "headcount, labour-hours, or attendance records. These outputs are <b>decision-support "
         "planning aids</b> — not verified staffing requirements.")

    if not models_ok:
        st.error("Trained models not found. Run `python train.py` first.")
        st.stop()

    # ── Parameter controls ────────────────────────────────────────────────
    sec("⚙️ Forecast & Operational Parameters")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("**Store Selection**")
        st_store = st.selectbox("Store ID", sorted(df["Store"].unique()), key="stf_store")
        st_dept  = st.selectbox("Department ID", sorted(df["Dept"].unique()), key="stf_dept")
        st_hol   = st.checkbox("Holiday Week Surge", value=False, key="stf_hol")

    with p2:
        st.markdown("**Forecast Date & Model**")
        st_date = st.date_input("Forecast Week Ending",
                                value=pd.Timestamp("2012-11-02").date(), key="stf_date")
        st_mdl  = st.radio("Forecast Model", ["Decision Tree", "Linear Regression"], key="stf_mdl")

    with p3:
        st.markdown("**Operational Assumptions**")
        st_sph = st.number_input(
            "Sales per Associate-Hour ($)",
            min_value=50.0, max_value=1500.0,
            value=float(DEFAULT_SALES_PER_HOUR), step=25.0, key="stf_sph",
            help="US general retail benchmark: $250–$400/labor-hour",
        )
        st_hrs = st.number_input(
            "Hours per FTE / Week",
            min_value=10.0, max_value=60.0,
            value=float(DEFAULT_HOURS_PER_WEEK), step=1.0, key="stf_hrs",
        )
        st_min = st.number_input(
            "Minimum Staff Floor",
            min_value=1, max_value=20, value=2, step=1, key="stf_min",
        )

    # ── Reactive calculation ───────────────────────────────────────────────
    st_target  = pd.Timestamp(st_date)
    s_lr, s_dt = safe_forecast(st_store, st_dept, st_target, st_hol)
    if s_dt is None:
        st.stop()

    active_sales = s_dt if "Decision" in st_mdl else s_lr
    est          = estimate_staffing(active_sales, st_sph, st_hrs, int(st_min))
    recommend    = staffing_recommendation(est["estimated_staff"], st_hol)

    st.markdown('<hr class="thin-hr">', unsafe_allow_html=True)
    sec("📊 Calculated Staffing Requirements")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        kpi("Forecasted Weekly Sales",  fmt_c(est["forecasted_sales"]),    f"Model: {st_mdl}",           "green")
    with r2:
        kpi("Estimated Labor Hours",    f"{est['labour_hours']:,.1f} hrs", f"Sales ÷ ${st_sph:.0f}/hr")
    with r3:
        kpi("FTE Equivalent",           f"{est['fte_staff']:.1f} FTE",    f"Hours ÷ {st_hrs:.0f} hrs/wk")
    with r4:
        kpi("Recommended Staff Count",  f"{est['estimated_staff']} Associates",
            f"Min Floor: {st_min}", "amber" if st_hol else "green")

    info(f"📋 Planning Guidance: {recommend}")

    # ── Donut + summary ────────────────────────────────────────────────────
    labor_used   = est["labour_hours"]
    total_avail  = max(labor_used, est["estimated_staff"] * st_hrs)
    buffer       = max(0.0, total_avail - labor_used)

    c_chart, c_notes = st.columns([1.2, 1])
    with c_chart:
        fig_donut = go.Figure(go.Pie(
            labels=["Scheduled Labor Hours", "Available Shift Buffer"],
            values=[labor_used, buffer],
            hole=0.6,
            marker_colors=["#1565C0", "#B0BEC5"],
            textinfo="label+percent",
        ))
        fig_donut.add_annotation(
            text=f"<b>{est['estimated_staff']}</b><br>Staff",
            x=0.5, y=0.5, font_size=18, showarrow=False,
        )
        fig_donut.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="#FFFFFF", font=dict(color="#37474F"),
        )
        st.plotly_chart(fig_donut, use_container_width=True, theme=None)

    with c_notes:
        st.markdown(f"""
        **Calculation Breakdown:**

        | Parameter | Value |
        |---|---|
        | Weekly Sales Forecast | `{fmt_c(est['forecasted_sales'])}` |
        | Productivity Rate | `${st_sph:.0f}` per associate-hour |
        | Total Hours Required | `{est['labour_hours']:.1f}` hrs |
        | FTE Standard | `{st_hrs:.0f}` hrs/week/associate |
        | Calculated FTEs | `{est['fte_staff']:.2f}` |
        | Minimum Floor | `{st_min}` associates |
        | **Final Recommendation** | **`{est['estimated_staff']} Associates`** |
        """)
        st.caption(est["assumption_note"])

    # ── Multi-week planner ─────────────────────────────────────────────────
    sec("📅 Multi-Week Staffing Planner")
    with st.expander("Generate Multi-Week Staffing Schedule", expanded=False):
        m1, m2, m3 = st.columns([1, 1, 1])
        with m1:
            mw_start = st.date_input("Schedule Start", value=pd.Timestamp("2012-11-02").date(), key="mws")
        with m2:
            mw_end   = st.date_input("Schedule End",   value=pd.Timestamp("2013-01-25").date(), key="mwe")
        with m3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            run_multi = st.button("▶ Generate Staffing Plan", type="primary", use_container_width=True, key="run_multi")

        if run_multi:
            with st.spinner("Computing multi-week staffing schedule…"):
                m_range = predict_range(st_store, st_dept, pd.Timestamp(mw_start), pd.Timestamp(mw_end))
                m_staff = staffing_from_forecast_df(m_range, "DT_Prediction", st_sph, st_hrs, int(st_min))

            fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=["Weekly Sales Forecast ($)", "Recommended Staff Count"])
            fig_m.add_trace(go.Bar(x=m_staff["Date"], y=m_staff["DT_Prediction"],
                                   marker_color="#1565C0", name="Sales ($)"), row=1, col=1)
            fig_m.add_trace(go.Bar(x=m_staff["Date"], y=m_staff["Estimated_Staff"],
                                   marker_color="#2E7D32", name="Staff"), row=2, col=1)
            fig_m.update_layout(
                height=420, margin=dict(t=30, b=10), showlegend=False,
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(color="#37474F"),
            )
            st.plotly_chart(fig_m, use_container_width=True, theme=None)

            tbl = m_staff.copy()
            tbl["Date"]          = tbl["Date"].dt.date.astype(str)
            tbl["LR_Prediction"] = tbl["LR_Prediction"].apply(fmt_c)
            tbl["DT_Prediction"] = tbl["DT_Prediction"].apply(fmt_c)
            tbl = tbl.rename(columns={
                "LR_Prediction": "Linear Reg ($)",
                "DT_Prediction": "Decision Tree ($)",
                "Labour_Hours": "Labor Hours",
                "FTE_Staff": "FTE Needed",
                "Estimated_Staff": "Staff Recommended",
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOLIDAY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif current == "Holiday Analysis":
    df = get_dataset()

    st.markdown('<div class="page-heading">🎄 Holiday Impact Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subhead">Measure how Walmart holiday weeks affect sales across stores and departments.</div>',
                unsafe_allow_html=True)

    hol_avg  = df.loc[df["IsHoliday"], "Weekly_Sales"].mean()
    nhol_avg = df.loc[~df["IsHoliday"], "Weekly_Sales"].mean()
    lift     = ((hol_avg - nhol_avg) / nhol_avg) * 100

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Holiday Avg Sales",    fmt_c(hol_avg),  "Per store-dept week",   "green")
    with k2: kpi("Non-Holiday Avg",      fmt_c(nhol_avg), "Standard weekly base")
    with k3: kpi("Holiday Sales Lift",   f"{lift:+.1f}%", "Volume expansion",      "amber")
    with k4: kpi("Holiday Weeks (Train)", "29,661",        "7.0% of training data", "purple")

    sec("📦 Top 20 Departments by Holiday Uplift")
    h_lift = df.groupby(["Dept", "IsHoliday"])["Weekly_Sales"].mean().unstack().reset_index()
    h_lift.columns = ["Dept", "Non_Holiday", "Holiday"]
    h_lift = h_lift.dropna()
    h_lift["Uplift_%"] = ((h_lift["Holiday"] - h_lift["Non_Holiday"]) / h_lift["Non_Holiday"] * 100).round(1)
    top20 = h_lift.nlargest(20, "Uplift_%")

    fig_up = px.bar(
        top20, x="Dept", y="Uplift_%",
        color="Uplift_%", color_continuous_scale=["#90CAF9", "#C62828"],
        labels={"Uplift_%": "Sales Uplift (%)", "Dept": "Department"},
        text_auto=True,
    )
    fig_up.update_layout(
        height=380, margin=dict(t=10, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#37474F"), coloraxis_showscale=False,
        yaxis=dict(gridcolor="#F0F4F8"),
    )
    st.plotly_chart(fig_up, use_container_width=True, theme=None)

    sec("🏪 Store-Wise Holiday vs Non-Holiday Performance")
    store_hol = df.groupby(["Store", "IsHoliday"])["Weekly_Sales"].mean().unstack().reset_index()
    store_hol.columns = ["Store", "Non_Holiday", "Holiday"]
    store_hol = store_hol.dropna()

    fig_sh = go.Figure()
    fig_sh.add_trace(go.Bar(x=store_hol["Store"], y=store_hol["Non_Holiday"],
                            name="Non-Holiday", marker_color="#1976D2"))
    fig_sh.add_trace(go.Bar(x=store_hol["Store"], y=store_hol["Holiday"],
                            name="Holiday Weeks", marker_color="#E53935"))
    fig_sh.update_layout(
        barmode="group", height=360, margin=dict(t=10, b=10),
        xaxis_title="Store ID", yaxis_title="Avg Weekly Sales ($)",
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#37474F"),
        legend=dict(orientation="h", y=1.06),
        yaxis=dict(gridcolor="#F0F4F8"),
    )
    st.plotly_chart(fig_sh, use_container_width=True, theme=None)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif current == "Dataset Explorer":
    df     = get_dataset()
    stores = get_stores_df()

    st.markdown('<div class="page-heading">🗃️ Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subhead">Explore the genuine Kaggle Walmart Recruiting dataset.</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Metadata & Schema", "🔍 Missing Values", "📂 Interactive Preview"])

    with tab1:
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi("Total Observations", f"{len(df):,}",         "Store-dept-week rows")
        with k2: kpi("Number of Stores",   "45",                   "Types: A(22) B(17) C(6)")
        with k3: kpi("Departments",        "81",                   "Unique department IDs")
        with k4: kpi("Unique Weeks",       f"{df['Date'].nunique()}", "Sales weeks in dataset")

        schema = pd.DataFrame({
            "Column":          df.columns.tolist(),
            "Data Type":       [str(df[c].dtype) for c in df.columns],
            "Missing Values":  [int(df[c].isnull().sum()) for c in df.columns],
            "Null %":          [f"{df[c].isnull().mean()*100:.1f}%" for c in df.columns],
        })
        st.dataframe(schema, use_container_width=True, hide_index=True)

    with tab2:
        null_c = df.isnull().sum()
        null_df = null_c[null_c > 0].reset_index()
        null_df.columns = ["Feature", "Missing Records"]
        null_df["Missing %"] = (null_df["Missing Records"] / len(df) * 100).round(2)
        st.dataframe(null_df, use_container_width=True, hide_index=True)
        info("MarkDown columns represent promotional events introduced in November 2011. "
             "Prior weeks legitimately contain NaN — not data quality issues.")

    with tab3:
        t1, t2, t3 = st.columns([1, 1, 0.8])
        with t1:
            p_store = st.selectbox("Filter Store",      ["All"] + sorted(df["Store"].unique().tolist()), key="pe_s")
        with t2:
            p_dept  = st.selectbox("Filter Department", ["All"] + sorted(df["Dept"].unique().tolist()),  key="pe_d")
        with t3:
            n_show  = st.slider("Max Rows", 10, 300, 50)

        view = df.copy()
        if p_store != "All": view = view[view["Store"] == int(p_store)]
        if p_dept  != "All": view = view[view["Dept"]  == int(p_dept)]
        st.dataframe(view.head(n_show), use_container_width=True)
        st.caption(f"Showing {min(n_show, len(view)):,} of {len(view):,} filtered records.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INVENTORY INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif current == "Inventory Insights":
    eng = get_engineered()

    st.markdown('<div class="page-heading">📦 Inventory Planning Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subhead">Seasonal and cyclical demand patterns derived from engineered temporal features.</div>',
                unsafe_allow_html=True)

    sec("🌤️ Seasonal Demand Volatility")
    season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
    s_agg = eng.groupby("Season")["Weekly_Sales"].mean().reset_index()
    s_agg["Season_Name"] = s_agg["Season"].map(season_map)

    fig_s = px.bar(
        s_agg, x="Season_Name", y="Weekly_Sales",
        color="Season_Name",
        color_discrete_sequence=["#1E88E5", "#43A047", "#FB8C00", "#E53935"],
        labels={"Weekly_Sales": "Avg Weekly Sales ($)", "Season_Name": "Season"},
        text_auto=True,
    )
    fig_s.update_layout(
        height=340, margin=dict(t=10, b=10), showlegend=False,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#37474F"),
        yaxis=dict(gridcolor="#F0F4F8"),
    )
    st.plotly_chart(fig_s, use_container_width=True, theme=None)

    sec("📅 Week-of-Year Annual Demand Profile")
    woy = eng.groupby("WeekOfYear")["Weekly_Sales"].mean().reset_index()
    fig_woy = px.line(
        woy, x="WeekOfYear", y="Weekly_Sales",
        labels={"Weekly_Sales": "Avg Weekly Sales ($)", "WeekOfYear": "Calendar Week (1–52)"},
        color_discrete_sequence=["#1565C0"],
    )
    for hw in [6, 36, 47, 51, 52]:
        fig_woy.add_vline(x=hw, line_color="#E53935", line_dash="dot", opacity=0.6)
    fig_woy.update_layout(
        height=340, margin=dict(t=10, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#37474F"),
        xaxis=dict(gridcolor="#F0F4F8"), yaxis=dict(gridcolor="#F0F4F8"),
    )
    st.plotly_chart(fig_woy, use_container_width=True, theme=None)
    st.caption("Red dotted lines: Super Bowl (Wk 6), Labor Day (Wk 36), Thanksgiving (Wk 47), Christmas (Wk 51–52).")

    sec("📊 Department-Level Sales Variance")
    dept_stats = eng.groupby("Dept")["Weekly_Sales"].agg(["mean", "std"]).reset_index()
    dept_stats.columns = ["Dept", "Mean Sales", "Std Dev"]
    dept_stats["CV"] = (dept_stats["Std Dev"] / dept_stats["Mean Sales"] * 100).round(1)
    top_volatile = dept_stats.nlargest(20, "CV")

    fig_cv = px.bar(
        top_volatile, x="Dept", y="CV",
        color="CV", color_continuous_scale=["#90CAF9", "#1565C0"],
        labels={"CV": "Coefficient of Variation (%)", "Dept": "Department"},
        text_auto=True,
    )
    fig_cv.update_layout(
        height=340, margin=dict(t=10, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#37474F"), coloraxis_showscale=False,
        yaxis=dict(gridcolor="#F0F4F8"),
    )
    st.plotly_chart(fig_cv, use_container_width=True, theme=None)
    st.caption("Coefficient of Variation (CV) = Std Dev / Mean × 100. Higher CV = more volatile demand requiring safety stock buffers.")
