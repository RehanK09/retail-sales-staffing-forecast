"""
app.py — Retail Store Weekly Sales Forecasting & Staffing Demand Analysis System
Redesigned Streamlit Dashboard (Phase 1 UI Refresh)

Author : Rehan Khan | Reg No: 12526856 | Roll No: 29 | Batch: K3P25RA
Course : CSE252 – Artificial Intelligence and Machine Learning
Dataset: Kaggle Walmart Recruiting – Store Sales Forecasting (2010-02-05 to 2012-10-26)

NOTE ON DATA PERIOD:
  All predictions in this application are based on the historical Walmart Kaggle
  dataset (training period: Feb 2010 – Oct 2012). Forecasts are model outputs
  for weeks within or just beyond the dataset range. This is NOT a live
  real-time production system and does not reflect actual 2026 Walmart operations.
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

from src.data_loader  import load_merged_train, load_stores, load_features
from src.preprocessing import engineer_features
from src.models        import models_exist, load_all
from src.evaluation    import load_results
from src.staffing      import (estimate_staffing, staffing_from_forecast_df,
                               staffing_recommendation,
                               DEFAULT_SALES_PER_HOUR, DEFAULT_HOURS_PER_WEEK)
from src.forecasting   import predict, predict_range

# ─────────────────────────── page config ────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── constants ──────────────────────────────────────
C_BLUE   = "#1565C0"
C_LBLUE  = "#1976D2"
C_ACCENT = "#1E88E5"
C_GREEN  = "#2E7D32"
C_RED    = "#C62828"
C_AMBER  = "#F57F17"
C_GREY   = "#546E7A"
C_BG     = "#F5F7FA"
C_WHITE  = "#FFFFFF"
C_BORDER = "#E0E6EF"

# dataset range (true dates from the Kaggle training file)
DATA_START = pd.Timestamp("2010-02-05")
DATA_END   = pd.Timestamp("2012-10-26")

# ─────────────────────────── global CSS ─────────────────────────────────────
st.markdown("""
<style>
/* ---- body / app background ---- */
.stApp { background-color: #F5F7FA; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: #0D47A1; }
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p  { color: #E3F2FD !important; }
[data-testid="stSidebar"] .stRadio > label { color: #BBDEFB !important; font-size: 0.78rem; }

/* ---- KPI card ---- */
.kpi {
    background: #FFFFFF;
    border: 1px solid #E0E6EF;
    border-radius: 10px;
    padding: 16px 20px 14px 20px;
    margin-bottom: 10px;
    border-top: 3px solid #1565C0;
}
.kpi-label { font-size: 0.72rem; font-weight: 700; color: #546E7A;
             text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.kpi-value { font-size: 1.75rem; font-weight: 800; color: #0D1F3C; line-height: 1.1; }
.kpi-sub   { font-size: 0.72rem; color: #90A4AE; margin-top: 3px; }
.kpi-green { border-top-color: #2E7D32; }
.kpi-amber { border-top-color: #F57F17; }
.kpi-red   { border-top-color: #C62828; }

/* ---- section heading ---- */
.sec-head {
    font-size: 1.0rem; font-weight: 700; color: #0D47A1;
    border-bottom: 2px solid #BBDEFB; padding-bottom: 5px;
    margin: 18px 0 12px 0;
}

/* ---- info / warn banners ---- */
.info-box {
    background: #E3F2FD; border-left: 4px solid #1565C0;
    border-radius: 0 6px 6px 0; padding: 10px 14px;
    font-size: 0.82rem; color: #0D47A1; margin: 8px 0;
}
.warn-box {
    background: #FFF8E1; border-left: 4px solid #F57F17;
    border-radius: 0 6px 6px 0; padding: 10px 14px;
    font-size: 0.82rem; color: #5D3A00; margin: 8px 0;
}
.data-note {
    background: #E8F5E9; border-left: 4px solid #2E7D32;
    border-radius: 0 6px 6px 0; padding: 10px 14px;
    font-size: 0.80rem; color: #1B5E20; margin: 6px 0;
}

/* ---- tab style ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #E8EEF7; border-radius: 8px; padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px; font-size: 0.82rem; font-weight: 600; color: #546E7A;
}
.stTabs [aria-selected="true"] {
    background: #1565C0 !important; color: white !important;
}

/* ---- divider tweak ---- */
hr { border-top: 1px solid #E0E6EF; margin: 14px 0; }

/* ---- dataframe table ---- */
[data-testid="stDataFrame"] { border: 1px solid #E0E6EF; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _get_merged():
    return load_merged_train()

@st.cache_data(show_spinner=False)
def _get_eng():
    return engineer_features(_get_merged())

@st.cache_data(show_spinner=False)
def _get_stores():
    return load_stores()

@st.cache_data(show_spinner=False)
def _get_features_df():
    df = load_features()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data(show_spinner=False)
def _get_results():
    return load_results()

@st.cache_resource(show_spinner=False)
def _get_models():
    if models_exist():
        return load_all()
    return None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════════

def kpi(label, value, sub="", color=""):
    cls = f"kpi {color}".strip()
    st.markdown(f"""
    <div class="{cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def sec(title):
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)


def info(text):
    st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)


def warn(text):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)


def data_note(text):
    st.markdown(f'<div class="data-note">📅 {text}</div>', unsafe_allow_html=True)


def _fmt(val):
    return f"${val:,.2f}"


def _safe_predict(store, dept, date, is_hol):
    """Call predict() and return (lr_pred, dt_pred) or raise a user-friendly error."""
    try:
        r = predict(store, dept, pd.Timestamp(date), is_hol)
        return r["lr_pred"], r["dt_pred"]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


def _next_friday(ref: pd.Timestamp) -> pd.Timestamp:
    """Return the next Friday on or after ref (Walmart weeks end Friday)."""
    days_ahead = (4 - ref.weekday()) % 7   # Friday = weekday 4
    return ref + pd.Timedelta(days=days_ahead if days_ahead else 7)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def _sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:14px 0 8px 0;'>
          <span style='font-size:2.2rem;'>🛒</span><br>
          <span style='font-size:1.05rem;font-weight:800;line-height:1.3;'>
              Walmart<br>Sales Forecasting
          </span><br>
          <span style='font-size:0.68rem;opacity:0.75;'>
              CSE252 · Rehan Khan · 12526856
          </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1565C0;margin:8px 0;'>", unsafe_allow_html=True)

        # --- model status pill ---
        if models_exist():
            r = _get_results()
            if r:
                dt_r2 = r["Decision Tree"]["overall"]["R2"]
                st.markdown(f"""
                <div style='background:#0A3880;border-radius:8px;padding:10px 12px;
                            margin-bottom:8px;font-size:0.78rem;'>
                  ✅ <b>Models ready</b><br>
                  <span style='opacity:0.8;'>DT R² = {dt_r2:.4f} &nbsp;|&nbsp;
                  LR R² = {r["Linear Regression"]["overall"]["R2"]:.4f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.error("Models not found. Run train.py.")

        st.markdown("**MAIN**", unsafe_allow_html=False)
        page = st.radio(
            "nav", ["🏠 Dashboard", "🔮 Sales Forecasting"],
            label_visibility="collapsed",
        )

        with st.expander("📊 Advanced Analysis"):
            adv = st.radio(
                "adv", [
                    "📈 Model Comparison",
                    "🎄 Holiday Analysis",
                    "🗃️ Dataset Explorer",
                    "📋 Detailed Metrics",
                ],
                label_visibility="collapsed",
            )
            if st.button("Open →", key="adv_go", use_container_width=True):
                page = adv

        with st.expander("🛠️ Planning Tools"):
            plan = st.radio(
                "plan", ["👥 Staffing Estimator", "📦 Inventory Insights"],
                label_visibility="collapsed",
            )
            if st.button("Open →", key="plan_go", use_container_width=True):
                page = plan

        st.markdown("<hr style='border-color:#1565C0;margin:8px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.68rem;opacity:0.6;text-align:center;padding-bottom:6px;'>
        Dataset: Walmart Kaggle 2010–2012<br>
        CSE252 Project · Nov 2026
        </div>""", unsafe_allow_html=True)

    return page


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    df  = _get_merged()
    eng = _get_eng()
    r   = _get_results()

    # ── header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:4px 0 18px 0;'>
      <span style='font-size:1.5rem;font-weight:800;color:#0D47A1;'>
          🛒 Retail Sales Forecasting Dashboard
      </span><br>
      <span style='font-size:0.82rem;color:#546E7A;'>
          Walmart Recruiting Dataset · 45 Stores · 81 Departments
      </span>
    </div>""", unsafe_allow_html=True)

    data_note(
        f"Historical dataset period: {DATA_START.date()} → {DATA_END.date()}. "
        "Forecasts shown below are model outputs based on this dataset. "
        "This is not a live production system."
    )

    # ── quick-forecast controls (always visible) ─────────────────────────────
    sec("⚡ Quick Forecast")
    qc1, qc2, qc3, qc4 = st.columns([1.2, 1.2, 1.2, 0.8])
    all_stores = sorted(df["Store"].unique().tolist())
    all_depts  = sorted(df["Dept"].unique().tolist())

    with qc1:
        q_store = st.selectbox("Store", all_stores, index=0, key="dash_store")
    with qc2:
        q_dept  = st.selectbox("Department", all_depts, index=0, key="dash_dept")
    with qc3:
        # default: one week after dataset end
        q_date  = st.date_input(
            "Forecast week",
            value=pd.Timestamp("2012-11-02").date(),
            min_value=pd.Timestamp("2010-01-01").date(),
            max_value=pd.Timestamp("2014-12-31").date(),
            key="dash_date",
        )
    with qc4:
        q_hol  = st.checkbox("Holiday week", value=False, key="dash_hol")
        q_model = st.radio("Model", ["DT", "LR"], horizontal=True, key="dash_model",
                           help="DT = Decision Tree (R²=0.92) · LR = Linear Regression (R²=0.10)")

    if not models_exist():
        st.error("Models not found. Run `train.py` first.")
        return

    # ── compute forecasts for next-week and next-month ───────────────────────
    next_week_date  = pd.Timestamp(q_date)
    next_month_end  = next_week_date + pd.Timedelta(weeks=4)

    with st.spinner("Computing forecast..."):
        lr_w, dt_w = _safe_predict(q_store, q_dept, next_week_date, q_hol)
        if lr_w is None:
            return

    # next-month: sum 4 weekly predictions
    month_dates = pd.date_range(next_week_date, periods=4, freq="W-FRI")
    month_total_lr, month_total_dt = 0.0, 0.0
    for d in month_dates:
        lr_x, dt_x = _safe_predict(q_store, q_dept, d, q_hol)
        if lr_x is not None:
            month_total_lr += lr_x
            month_total_dt += dt_x

    fc_week  = dt_w  if q_model == "DT" else lr_w
    fc_month = month_total_dt if q_model == "DT" else month_total_lr

    # ── KPI row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi("Next Week Forecast",
            _fmt(fc_week),
            f"Store {q_store} · Dept {q_dept} · {next_week_date.date()}")
    with k2:
        kpi("Next Month Est. Demand",
            _fmt(fc_month),
            f"4-week sum from {next_week_date.date()}", color="kpi-green")
    with k3:
        hist_avg = df[(df["Store"]==q_store) & (df["Dept"]==q_dept)]["Weekly_Sales"].mean()
        diff_pct = ((fc_week - hist_avg) / hist_avg * 100) if hist_avg else 0
        col = "kpi-green" if diff_pct >= 0 else "kpi-red"
        kpi("vs Historical Avg",
            f"{diff_pct:+.1f}%",
            f"Hist avg ${hist_avg:,.0f}/wk", color=col)
    with k4:
        kpi("Best Model",
            "Decision Tree",
            "R² = 0.9186 (validation set)", color="kpi-green")
    with k5:
        hol_txt = "🎄 YES" if q_hol else "Regular"
        kpi("Holiday Week", hol_txt, "Super Bowl / Thanksgiving / Christmas / Labour Day")

    st.divider()

    # ── historical vs forecast chart ─────────────────────────────────────────
    sec("📈 Historical Sales vs Forecast")
    hist = df[(df["Store"]==q_store) & (df["Dept"]==q_dept)].sort_values("Date")

    if hist.empty:
        st.info("No historical data for this store/department combination.")
    else:
        # Build a forecast range ±4 weeks
        fc_range = predict_range(q_store, q_dept,
                                 next_week_date,
                                 next_week_date + pd.Timedelta(weeks=7))
        fig = go.Figure()

        # historical area
        fig.add_trace(go.Scatter(
            x=hist["Date"], y=hist["Weekly_Sales"],
            mode="lines", name="Historical Sales",
            line=dict(color=C_GREY, width=1.5),
            fill="tozeroy", fillcolor="rgba(84,110,122,0.08)",
        ))

        # forecast line
        fig.add_trace(go.Scatter(
            x=fc_range["Date"],
            y=fc_range["DT_Prediction"] if q_model == "DT" else fc_range["LR_Prediction"],
            mode="lines+markers", name=f"{q_model} Forecast",
            line=dict(color=C_BLUE, width=2.5, dash="dot"),
            marker=dict(size=7, color=C_BLUE),
        ))

        # highlight the primary forecast point
        fig.add_trace(go.Scatter(
            x=[next_week_date], y=[fc_week],
            mode="markers+text",
            name="Next Week",
            text=[f"  {_fmt(fc_week)}"],
            textposition="top right",
            textfont=dict(color=C_BLUE, size=11, family="Arial Black"),
            marker=dict(size=14, color=C_BLUE, symbol="star"),
            showlegend=True,
        ))

        # holiday band markers on historical data
        hol_hist = hist[hist["IsHoliday"]]
        if not hol_hist.empty:
            fig.add_trace(go.Scatter(
                x=hol_hist["Date"], y=hol_hist["Weekly_Sales"],
                mode="markers", name="Holiday Weeks",
                marker=dict(color=C_RED, size=5, symbol="diamond"),
                opacity=0.7,
            ))

        # dividing line between history and forecast
        fig.add_vline(x=DATA_END, line_color=C_AMBER, line_dash="dash",
                      line_width=1.5, opacity=0.7)
        fig.add_annotation(
            x=DATA_END, y=0, yref="paper", yanchor="bottom",
            text="  Dataset end", font=dict(color=C_AMBER, size=10),
            showarrow=False,
        )

        fig.update_layout(
            height=360, template="plotly_white",
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(t=10, b=10, l=0, r=0),
            xaxis_title="", yaxis_title="Weekly Sales ($)",
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#EEF0F5"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Historical data: {DATA_START.date()} – {DATA_END.date()} (genuine Kaggle dataset). "
            "Dashed line = model forecast. "
            "⚠️ Forecast points are model outputs, not verified future sales."
        )

    # ── model summary strip ──────────────────────────────────────────────────
    if r:
        sec("🏅 Model Performance Summary (Validation Set)")
        m1, m2, m3, m4 = st.columns(4)
        dt = r["Decision Tree"]["overall"]
        lr = r["Linear Regression"]["overall"]
        with m1: kpi("DT — R² Score",   f"{dt['R2']:.4f}", "Decision Tree", "kpi-green")
        with m2: kpi("DT — MAE",  f"${dt['MAE']:,.0f}", "Mean Absolute Error")
        with m3: kpi("LR — R² Score",   f"{lr['R2']:.4f}", "Linear Regression", "kpi-red")
        with m4: kpi("LR — MAE",  f"${lr['MAE']:,.0f}", "Mean Absolute Error")
        info("Validation on chronological hold-out (last 20% of weeks). No random shuffling. Decision Tree significantly outperforms Linear Regression on this dataset.")

    # ── 4-week forecast table ────────────────────────────────────────────────
    sec("📅 4-Week Forecast Breakdown")
    rows = []
    for d in month_dates:
        lr_v, dt_v = _safe_predict(q_store, q_dept, d, q_hol)
        if lr_v is not None:
            rows.append({
                "Week":        str(d.date()),
                "LR Forecast": f"${lr_v:,.2f}",
                "DT Forecast": f"${dt_v:,.2f}",
                "Selected":    f"${(dt_v if q_model=='DT' else lr_v):,.2f}",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SALES FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

def page_forecasting():
    st.markdown("## 🔮 Sales Forecasting")
    data_note(
        f"Dataset range: {DATA_START.date()} to {DATA_END.date()}. "
        "Predictions use the trained Decision Tree / Linear Regression models. "
        "Values outside the training period are model extrapolations."
    )

    if not models_exist():
        st.error("Models not found. Run `train.py` first.")
        return

    df = _get_merged()

    # ── controls ─────────────────────────────────────────────────────────────
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([1,1,1.2,1,1])
        with c1: f_store = st.selectbox("Store", sorted(df["Store"].unique()), key="fc2_s")
        with c2: f_dept  = st.selectbox("Department", sorted(df["Dept"].unique()), key="fc2_d")
        with c3:
            f_date  = st.date_input("Forecast week",
                value=pd.Timestamp("2012-11-02").date(),
                min_value=pd.Timestamp("2010-01-01").date(),
                max_value=pd.Timestamp("2014-12-31").date(), key="fc2_dt")
        with c4: f_hol   = st.checkbox("Holiday week", key="fc2_hol")
        with c5: f_model = st.radio("Model", ["Decision Tree", "Linear Regression"],
                                     key="fc2_model")

    go_btn = st.button("Generate Forecast", type="primary", use_container_width=False)

    if go_btn:
        lr_w, dt_w = _safe_predict(f_store, f_dept, pd.Timestamp(f_date), f_hol)
        if lr_w is None:
            return

        fc = dt_w if "Decision" in f_model else lr_w

        # 4-week range from selected date
        range_end = pd.Timestamp(f_date) + pd.Timedelta(weeks=4)
        with st.spinner("Computing range forecast..."):
            fdf = predict_range(f_store, f_dept, pd.Timestamp(f_date), range_end)

        month_fc = fdf["DT_Prediction"].sum() if "Decision" in f_model else fdf["LR_Prediction"].sum()

        # KPIs
        k1, k2, k3 = st.columns(3)
        with k1: kpi("Next Week Forecast",   _fmt(fc),        f"{f_model} · Store {f_store} · Dept {f_dept}")
        with k2: kpi("4-Week Total Forecast", _fmt(month_fc), "Sum of next 4 weekly predictions", "kpi-green")
        with k3:
            hist_avg = df[(df["Store"]==f_store)&(df["Dept"]==f_dept)]["Weekly_Sales"].mean()
            kpi("Historical Weekly Average", f"${hist_avg:,.0f}" if not np.isnan(hist_avg) else "N/A",
                f"All weeks · Store {f_store} · Dept {f_dept}")

        # chart: historical + range forecast
        hist = df[(df["Store"]==f_store)&(df["Dept"]==f_dept)].sort_values("Date")
        col_key = "DT_Prediction" if "Decision" in f_model else "LR_Prediction"

        fig = go.Figure()
        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=hist["Date"], y=hist["Weekly_Sales"],
                name="Historical", mode="lines",
                line=dict(color=C_GREY, width=1.5),
            ))
        fig.add_trace(go.Scatter(
            x=fdf["Date"], y=fdf[col_key],
            name=f"{f_model} Forecast",
            mode="lines+markers",
            line=dict(color=C_BLUE, width=2.5, dash="dot"),
            marker=dict(size=8, color=C_BLUE),
        ))
        fig.add_vline(x=DATA_END, line_color=C_AMBER, line_dash="dash",
                      line_width=1.5, opacity=0.7)
        fig.update_layout(
            height=380, template="plotly_white",
            title=f"Store {f_store} · Dept {f_dept} — Historical vs Forecast",
            xaxis_title="", yaxis_title="Weekly Sales ($)",
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Range table
        sec("📅 Weekly Forecast Details")
        fdf_display = fdf.copy()
        fdf_display["Date"] = fdf_display["Date"].dt.date.astype(str)
        fdf_display["LR_Prediction"] = fdf_display["LR_Prediction"].apply(_fmt)
        fdf_display["DT_Prediction"] = fdf_display["DT_Prediction"].apply(_fmt)
        fdf_display.columns = ["Week", "LR Forecast", "DT Forecast"]
        st.dataframe(fdf_display, use_container_width=True, hide_index=True)
        st.caption("⚠️ All values are model estimates. This dataset covers 2010–2012. Forecasts beyond Oct 2012 are model extrapolations.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def page_model_comparison():
    st.markdown("## 📈 Model Comparison")
    r = _get_results()
    if r is None:
        st.error("Run `train.py` first.")
        return

    info("Validation on a chronological hold-out set (last 20% of weeks). No random shuffling was used. Scaler was fitted on training data only.")

    lr, dt = r["Linear Regression"], r["Decision Tree"]

    # Overall table
    sec("Overall Validation Metrics")
    tbl = pd.DataFrame({
        "Metric":            ["MAE ($)", "RMSE ($)", "R² Score", "Validation Samples"],
        "Linear Regression": [f"${lr['overall']['MAE']:,.0f}",
                               f"${lr['overall']['RMSE']:,.0f}",
                               f"{lr['overall']['R2']:.4f}",
                               f"{lr['overall']['n']:,}"],
        "Decision Tree":     [f"${dt['overall']['MAE']:,.0f}",
                               f"${dt['overall']['RMSE']:,.0f}",
                               f"{dt['overall']['R2']:.4f}",
                               f"{dt['overall']['n']:,}"],
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Bar chart
    chart_df = pd.DataFrame({
        "Model":  ["Linear Regression","Decision Tree"]*2,
        "Metric": ["MAE","MAE","RMSE","RMSE"],
        "Value":  [lr["overall"]["MAE"], dt["overall"]["MAE"],
                   lr["overall"]["RMSE"], dt["overall"]["RMSE"]],
    })
    fig = px.bar(chart_df, x="Metric", y="Value", color="Model", barmode="group",
                 color_discrete_map={"Linear Regression": C_RED, "Decision Tree": C_BLUE},
                 labels={"Value":"Error ($)"}, template="plotly_white", text_auto=True)
    fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
    fig.update_layout(height=360, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    # Holiday breakdown
    sec("Holiday vs Non-Holiday Breakdown")
    rows = []
    for mname, res in [("Linear Regression", lr), ("Decision Tree", dt)]:
        for period, pdata in [("Holiday", res["holiday"]), ("Non-Holiday", res["non_holiday"])]:
            rows.append({"Model": mname, "Period": period,
                         "MAE": pdata["MAE"], "RMSE": pdata["RMSE"],
                         "R²": pdata["R2"], "n": pdata["n"]})
    bd = pd.DataFrame(rows)

    fig2 = px.bar(bd, x="Model", y="MAE", color="Period", barmode="group",
                  color_discrete_map={"Holiday": C_RED, "Non-Holiday": C_BLUE},
                  template="plotly_white", text="MAE",
                  labels={"MAE":"Mean Absolute Error ($)"})
    fig2.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig2.update_layout(height=360, margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)

    sec("Holiday Performance Gap")
    for mn, res in [("Linear Regression", lr), ("Decision Tree", dt)]:
        c1, c2 = st.columns(2)
        with c1:
            kpi(f"{mn} · MAE Gap", f"+${res['holiday_gap']['MAE_gap']:,.0f}",
                "Holiday − Non-Holiday MAE")
        with c2:
            kpi(f"{mn} · RMSE Gap", f"+${res['holiday_gap']['RMSE_gap']:,.0f}",
                "Holiday − Non-Holiday RMSE")

    info("Holiday weeks have higher errors due to unusual demand spikes (Thanksgiving, Christmas). The Decision Tree handles these better than Linear Regression.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOLIDAY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def page_holiday_analysis():
    st.markdown("## 🎄 Holiday Analysis")
    df = _get_merged()
    eng = _get_eng()

    hol_avg  = df.loc[df["IsHoliday"],  "Weekly_Sales"].mean()
    nhol_avg = df.loc[~df["IsHoliday"], "Weekly_Sales"].mean()
    lift     = (hol_avg - nhol_avg) / nhol_avg * 100

    c1, c2, c3 = st.columns(3)
    with c1: kpi("Holiday Avg Sales",     f"${hol_avg:,.0f}", "Per store-dept week", "kpi-green")
    with c2: kpi("Non-Holiday Avg Sales", f"${nhol_avg:,.0f}", "Per store-dept week")
    with c3: kpi("Holiday Sales Lift",    f"{lift:+.1f}%",    "vs non-holiday average", "kpi-amber")

    fig = px.box(df, x="IsHoliday", y="Weekly_Sales", color="IsHoliday",
                 color_discrete_map={True: C_RED, False: C_BLUE},
                 labels={"IsHoliday":"Holiday Week","Weekly_Sales":"Weekly Sales ($)"},
                 template="plotly_white")
    fig.update_layout(showlegend=False, height=360)
    st.plotly_chart(fig, use_container_width=True)

    sec("Holiday Sales Uplift by Department (Top 20)")
    hol_lift = df.groupby(["Dept","IsHoliday"])["Weekly_Sales"].mean().unstack().reset_index()
    hol_lift.columns = ["Dept","Non_Holiday","Holiday"]
    hol_lift = hol_lift.dropna()
    hol_lift["Uplift_%"] = ((hol_lift["Holiday"]-hol_lift["Non_Holiday"])/hol_lift["Non_Holiday"]*100).round(1)
    top20 = hol_lift.nlargest(20,"Uplift_%")
    fig2 = px.bar(top20, x="Dept", y="Uplift_%",
                  color="Uplift_%", color_continuous_scale=["#BBDEFB", C_RED],
                  template="plotly_white",
                  labels={"Uplift_%":"Holiday Uplift (%)","Dept":"Department"})
    fig2.update_layout(height=360, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Departments with the highest holiday uplift require additional inventory and staffing buffers.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

def page_dataset_explorer():
    st.markdown("## 🗃️ Dataset Explorer")
    info(f"Genuine Walmart Kaggle dataset. Training period: {DATA_START.date()} to {DATA_END.date()}.")
    df = _get_merged()
    stores = _get_stores()

    tab1, tab2, tab3 = st.tabs(["Overview", "Missing Values", "Data Preview"])
    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Records",    f"{len(df):,}")
        with c2: kpi("Stores",     "45")
        with c3: kpi("Departments","81")
        with c4: kpi("Date Range", f"{DATA_START.date()} → {DATA_END.date()}")
        st.divider()
        col_df = pd.DataFrame({
            "Column": df.columns.tolist(),
            "Type":   [str(df[c].dtype) for c in df.columns],
            "Nulls":  [int(df[c].isnull().sum()) for c in df.columns],
            "Null%":  [f"{df[c].isnull().mean()*100:.1f}%" for c in df.columns],
            "Unique": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(col_df, use_container_width=True, hide_index=True)
    with tab2:
        null_s = df.isnull().sum()
        null_df = null_s[null_s>0].reset_index()
        null_df.columns = ["Feature","Null Count"]
        null_df["Null %"] = (null_df["Null Count"]/len(df)*100).round(2)
        if null_df.empty:
            st.success("No missing values in merged dataset.")
        else:
            fig = px.bar(null_df, x="Feature", y="Null Count", color="Null %",
                         color_continuous_scale=["#BBDEFB", C_RED],
                         template="plotly_white")
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(null_df, use_container_width=True, hide_index=True)
        warn("MarkDown columns are missing for weeks prior to November 2011 — this is standard for the Kaggle dataset. Missing values are filled with 0 during preprocessing.")
    with tab3:
        s_sel = st.selectbox("Store", ["All"]+sorted(df["Store"].unique().tolist()), key="de_s")
        d_sel = st.selectbox("Dept",  ["All"]+sorted(df["Dept"].unique().tolist()),  key="de_d")
        n_rows = st.slider("Rows", 10, 300, 50, key="de_n")
        prv = df.copy()
        if s_sel!="All": prv = prv[prv["Store"]==int(s_sel)]
        if d_sel!="All": prv = prv[prv["Dept"]==int(d_sel)]
        st.dataframe(prv.head(n_rows), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DETAILED METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def page_detailed_metrics():
    st.markdown("## 📋 Detailed Evaluation Metrics")
    r = _get_results()
    if r is None:
        st.error("Run `train.py` first.")
        return

    for mname, res in r.items():
        with st.expander(f"**{mname}** — R² {res['overall']['R2']:.4f}", expanded=(mname=="Decision Tree")):
            c1, c2, c3 = st.columns(3)
            for label, grp in [("Overall", "overall"), ("Holiday", "holiday"), ("Non-Holiday", "non_holiday")]:
                g = res[grp]
                col = [c1,c2,c3][["Overall","Holiday","Non-Holiday"].index(label)]
                with col:
                    st.markdown(f"**{label}** (n={g['n']:,})")
                    kpi("MAE",  f"${g['MAE']:,.0f}")
                    kpi("RMSE", f"${g['RMSE']:,.0f}")
                    kpi("R²",   f"{g['R2']:.4f}")

    sec("Validation Methodology")
    st.markdown("""
    | Parameter | Value |
    |:--|:--|
    | Split strategy | Chronological hold-out — no random shuffling |
    | Train window | First 80% of weeks (by date) |
    | Validation window | Last 20% of weeks (~2012-04-20 onward) |
    | Train samples | ~338,738 store-dept-week records |
    | Validation samples | ~82,832 store-dept-week records |
    | Feature scaling | StandardScaler fitted on training data only |
    | Features used | 29 (temporal, store, MarkDown indicators, macroeconomic) |
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STAFFING ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

def page_staffing():
    st.markdown("## 👥 Staffing Estimator")
    warn(
        "ESTIMATES ONLY. This tool uses configurable operational assumptions to translate "
        "forecasted sales into staffing estimates. The Walmart dataset does NOT contain "
        "employee headcount, labour-hours, or attendance records. These outputs are "
        "decision-support planning aids — not verified staffing requirements."
    )

    if not models_exist():
        st.error("Models not found. Run `train.py` first.")
        return

    df = _get_merged()

    c1, c2 = st.columns([1, 2])
    with c1:
        sec("Parameters")
        s_store = st.selectbox("Store", sorted(df["Store"].unique()), key="st_s")
        s_dept  = st.selectbox("Department", sorted(df["Dept"].unique()), key="st_d")
        s_date  = st.date_input("Week", value=pd.Timestamp("2012-11-02").date(), key="st_dt")
        s_hol   = st.checkbox("Holiday week", key="st_hol")
        st.markdown("**Operational Assumptions**")
        sales_hr = st.number_input("Sales per Associate-Hour ($)", 50.0, 1000.0,
                                    float(DEFAULT_SALES_PER_HOUR), 25.0, key="st_shr",
                                    help="Typical US retail: $200–$400/hr")
        hrs_wk   = st.number_input("Working Hours / FTE / Week", 10.0, 60.0,
                                    float(DEFAULT_HOURS_PER_WEEK), 1.0, key="st_hrs")
        min_staff = st.number_input("Minimum Staff", 1, 20, 2, key="st_min")
        s_model  = st.radio("Forecast Model", ["Decision Tree","Linear Regression"], key="st_mdl")
        calc_btn = st.button("Calculate", type="primary", use_container_width=True)

    with c2:
        if calc_btn:
            lr_w, dt_w = _safe_predict(s_store, s_dept, pd.Timestamp(s_date), s_hol)
            if lr_w is None:
                return
            fc = dt_w if "Decision" in s_model else lr_w
            est = estimate_staffing(fc, sales_hr, hrs_wk, int(min_staff))
            rec = staffing_recommendation(est["estimated_staff"], s_hol)

            sec("Results")
            r1, r2 = st.columns(2)
            with r1:
                kpi("Forecasted Weekly Sales", _fmt(est["forecasted_sales"]),
                    f"Model: {s_model}", "kpi-green")
                kpi("Estimated Labour Hours",  f"{est['labour_hours']:,.1f} hrs",
                    f"Sales ÷ ${sales_hr:.0f}/hr")
            with r2:
                kpi("FTE Equivalent",          f"{est['fte_staff']:.1f} FTE",
                    f"Labour hrs ÷ {hrs_wk:.0f} hrs/wk")
                kpi("Estimated Staff Required", str(est["estimated_staff"]),
                    f"Min: {min_staff}", "kpi-amber")

            st.markdown(f'<div class="info-box"><b>Recommendation:</b> {rec}</div>',
                        unsafe_allow_html=True)

            # gauge-style donut
            used  = est["labour_hours"]
            avail = max(0, est["estimated_staff"] * hrs_wk - used)
            fig = go.Figure(go.Pie(
                labels=["Labour Hours Used","Remaining Capacity"],
                values=[used, avail],
                hole=0.60,
                marker_colors=[C_BLUE, "#E8EEF7"],
                textinfo="label+percent",
            ))
            fig.add_annotation(text=f"{est['estimated_staff']}<br><span style='font-size:0.7em'>Staff</span>",
                               x=0.5, y=0.5, font_size=20, showarrow=False,
                               font=dict(color=C_BLUE, family="Arial Black"))
            fig.update_layout(showlegend=True, height=320, title="Labour Hour Utilisation",
                               margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(est["assumption_note"])

        # -- multi-week staffing ---
        st.divider()
        sec("Multi-Week Staffing Forecast")
        sc1, sc2, sc3 = st.columns(3)
        with sc1: rng_s2 = st.selectbox("Store", sorted(df["Store"].unique()), key="mw_s")
        with sc2: rng_d2 = st.selectbox("Dept",  sorted(df["Dept"].unique()),  key="mw_d")
        with sc3:
            mw_start = st.date_input("From", pd.Timestamp("2012-11-02").date(), key="mw_start")
            mw_end   = st.date_input("To",   pd.Timestamp("2013-01-31").date(), key="mw_end")

        if st.button("Generate Multi-Week Plan", use_container_width=False):
            with st.spinner("Computing..."):
                fdf = predict_range(rng_s2, rng_d2, pd.Timestamp(mw_start), pd.Timestamp(mw_end))
                fdf = staffing_from_forecast_df(fdf, "DT_Prediction", sales_hr, hrs_wk, int(min_staff))
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=["DT Forecast ($)", "Est. Staff"])
            fig2.add_trace(go.Bar(x=fdf["Date"], y=fdf["DT_Prediction"],
                                   marker_color=C_BLUE, name="Forecast"), row=1, col=1)
            fig2.add_trace(go.Bar(x=fdf["Date"], y=fdf["Estimated_Staff"],
                                   marker_color=C_GREEN, name="Staff"), row=2, col=1)
            fig2.update_layout(height=460, template="plotly_white", showlegend=False,
                                margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
            fdf_show = fdf.copy()
            fdf_show["Date"] = fdf_show["Date"].dt.date.astype(str)
            fdf_show = fdf_show.rename(columns={
                "LR_Prediction":"LR ($)","DT_Prediction":"DT ($)",
                "Labour_Hours":"Labour Hrs","FTE_Staff":"FTE","Estimated_Staff":"Est. Staff"
            })
            st.dataframe(fdf_show, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INVENTORY INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def page_inventory():
    st.markdown("## 📦 Inventory Planning Insights")
    info("Insights are derived from historical sales patterns. Use as supplementary planning signals, not procurement targets.")
    eng = _get_eng()
    df  = _get_merged()

    sec("Seasonal Demand Pattern")
    season_map = {1:"Winter",2:"Spring",3:"Summer",4:"Autumn"}
    seas = eng.groupby("Season")["Weekly_Sales"].mean().reset_index()
    seas["Season"] = seas["Season"].map(season_map)
    fig = px.bar(seas, x="Season", y="Weekly_Sales",
                 color="Season",
                 color_discrete_sequence=[C_BLUE, C_GREEN, C_AMBER, C_RED],
                 template="plotly_white",
                 labels={"Weekly_Sales":"Avg Weekly Sales ($)"})
    fig.update_layout(showlegend=False, height=320, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    sec("Week-of-Year Demand Profile")
    woy = eng.groupby("WeekOfYear")["Weekly_Sales"].mean().reset_index()
    fig2 = px.line(woy, x="WeekOfYear", y="Weekly_Sales", template="plotly_white",
                   labels={"Weekly_Sales":"Avg Weekly Sales ($)","WeekOfYear":"Week of Year"},
                   color_discrete_sequence=[C_BLUE])
    for hw in [6, 36, 47, 51, 52]:
        fig2.add_vline(x=hw, line_color=C_RED, line_dash="dot", opacity=0.5)
    fig2.update_layout(height=320, margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Dotted red lines = typical holiday weeks (Super Bowl≈Wk6, Labour Day≈Wk36, Thanksgiving≈Wk47, Christmas≈Wk51-52).")

    sec("Store Demand Variability")
    stores = _get_stores()
    store_stats = (df.groupby("Store")["Weekly_Sales"]
                    .agg(["mean","std","max"]).reset_index()
                    .merge(stores, on="Store"))
    store_stats["CoV%"] = (store_stats["std"]/store_stats["mean"]*100).round(1)
    fig3 = px.scatter(store_stats, x="mean", y="std", size="max", color="Type",
                      hover_name="Store",
                      color_discrete_map={"A":C_BLUE,"B":C_AMBER,"C":C_GREEN},
                      template="plotly_white",
                      labels={"mean":"Avg Weekly Sales ($)","std":"Std Dev ($)","Type":"Store Type"},
                      title="Store Sales: Mean vs Variability (bubble = peak weekly sales)")
    fig3.update_layout(height=400, margin=dict(t=30))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("High CoV% stores need larger safety-stock buffers.")


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    page = _sidebar()

    # Buttons inside expanders set page via re-run; fall back to radio selection
    if   "Dashboard"          in page: page_dashboard()
    elif "Forecasting"        in page: page_forecasting()
    elif "Model Comparison"   in page: page_model_comparison()
    elif "Holiday Analysis"   in page: page_holiday_analysis()
    elif "Dataset Explorer"   in page: page_dataset_explorer()
    elif "Detailed Metrics"   in page: page_detailed_metrics()
    elif "Staffing"           in page: page_staffing()
    elif "Inventory"          in page: page_inventory()
    else:                              page_dashboard()


if __name__ == "__main__":
    main()
