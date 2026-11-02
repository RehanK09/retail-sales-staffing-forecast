"""
app.py — Retail Store Weekly Sales Forecasting & Staffing Demand Analysis System
Interactive Streamlit Dashboard

Author : Rehan Khan
Reg No : 12526856  |  Roll No: 29  |  Batch: K3P25RA
Course : CSE252 – Artificial Intelligence and Machine Learning
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
import json

# ── project modules ─────────────────────────────────────────────────────────
from src.data_loader   import load_merged_train, load_train, load_stores, load_features
from src.preprocessing import engineer_features, temporal_split
from src.models        import models_exist, load_all
from src.evaluation    import load_results
from src.staffing      import estimate_staffing, staffing_recommendation, DEFAULT_SALES_PER_HOUR, DEFAULT_HOURS_PER_WEEK
from src.forecasting   import predict, predict_range

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Walmart Sales Forecasting",
    page_icon   = "🛒",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Color palette (Walmart-inspired dark-blue + accent) ──────────────────────
COLORS = {
    "primary"   : "#0071CE",   # Walmart blue
    "secondary" : "#FFC220",   # Walmart yellow
    "success"   : "#2ecc71",
    "danger"    : "#e74c3c",
    "neutral"   : "#8e9eb5",
    "bg_card"   : "#F0F4FA",
    "lr"        : "#0071CE",
    "dt"        : "#FFC220",
}

HOLIDAY_COLOR     = "#e74c3c"
NON_HOLIDAY_COLOR = "#0071CE"

# ── Custom CSS ────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* Main background */
    .stApp { background-color: #F7F9FC; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003087 0%, #0071CE 100%);
    }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stRadio label { color: #FFFFFF !important; }

    /* KPI card */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #0071CE;
        margin-bottom: 10px;
    }
    .kpi-label  { font-size: 0.82rem; color: #6b7c93; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value  { font-size: 1.8rem; font-weight: 800; color: #0d1f3c; line-height: 1.2; }
    .kpi-delta  { font-size: 0.78rem; color: #6b7c93; margin-top: 2px; }

    /* Section header */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #003087;
        border-bottom: 3px solid #0071CE;
        padding-bottom: 6px;
        margin-bottom: 18px;
    }

    /* Info banner */
    .info-banner {
        background: #E8F1FB;
        border-left: 4px solid #0071CE;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #003087;
        margin: 10px 0;
    }

    /* Warning banner */
    .warn-banner {
        background: #FFF8E1;
        border-left: 4px solid #FFC220;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #5a3e00;
        margin: 10px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #e6ecf5;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        font-weight: 600;
        color: #556080;
    }
    .stTabs [aria-selected="true"] {
        background: #0071CE !important;
        color: white !important;
    }

    /* Footer */
    .footer-note {
        font-size: 0.78rem;
        color: #aab4c8;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_merged_data():
    return load_merged_train()

@st.cache_data(show_spinner=False)
def get_engineered_data():
    raw = get_merged_data()
    return engineer_features(raw)

@st.cache_data(show_spinner=False)
def get_stores():
    return load_stores()

@st.cache_data(show_spinner=False)
def get_features_df():
    df = load_features()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data(show_spinner=False)
def get_eval_results():
    return load_results()

@st.cache_resource(show_spinner=False)
def get_models():
    if models_exist():
        return load_all()
    return None, None, None, None


# ── KPI card helper ──────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, delta: str = "", border_color: str = "#0071CE"):
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color:{border_color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta">{delta}</div>
    </div>""", unsafe_allow_html=True)


def section_header(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def info_box(text: str):
    st.markdown(f'<div class="info-banner">ℹ️  {text}</div>', unsafe_allow_html=True)


def warn_box(text: str):
    st.markdown(f'<div class="warn-banner">⚠️  {text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Executive Overview ────────────────────────────────────────────────────
def page_executive_overview():
    df = get_merged_data()

    st.markdown("## 🏪 Executive Overview")
    st.markdown("High-level business intelligence across all Walmart stores in the dataset.")

    # KPI row
    total_sales  = df["Weekly_Sales"].sum()
    mean_weekly  = df["Weekly_Sales"].mean()
    holiday_pct  = df["IsHoliday"].mean() * 100
    holiday_avg  = df.loc[df["IsHoliday"],  "Weekly_Sales"].mean()
    nonhol_avg   = df.loc[~df["IsHoliday"], "Weekly_Sales"].mean()
    holiday_lift = ((holiday_avg - nonhol_avg) / nonhol_avg * 100)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Total Sales Revenue", f"${total_sales/1e9:.2f}B", "All stores · all weeks")
    with c2: kpi_card("Avg Weekly Sales", f"${mean_weekly:,.0f}", "Per store-dept observation")
    with c3: kpi_card("Stores", "45", "Types A, B and C")
    with c4: kpi_card("Departments", "81", "Across all stores")
    with c5: kpi_card("Holiday Weeks", f"{df['IsHoliday'].sum():,}",
                       f"{holiday_pct:.1f}% of all weeks", border_color="#FFC220")

    st.divider()

    # -- Sales trend over time --
    section_header("📈 Weekly Sales Trend — All Stores")
    trend = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    fig = px.area(
        trend, x="Date", y="Weekly_Sales",
        labels={"Weekly_Sales": "Total Weekly Sales ($)", "Date": "Week"},
        color_discrete_sequence=[COLORS["primary"]],
        template="plotly_white",
    )
    # Overlay holiday markers
    holiday_dates = df[df["IsHoliday"]]["Date"].unique()
    for hd in holiday_dates:
        fig.add_vline(x=hd, line_width=1, line_color=HOLIDAY_COLOR, opacity=0.3)
    fig.update_traces(fillcolor="rgba(0,113,206,0.15)")
    fig.update_layout(height=360, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    info_box("Red vertical lines indicate holiday weeks (Super Bowl, Labour Day, Thanksgiving, Christmas).")

    # -- Holiday vs Non-Holiday & Store Type --
    c1, c2 = st.columns(2)
    with c1:
        section_header("🎄 Holiday vs Non-Holiday Sales")
        hol_df = df.groupby("IsHoliday")["Weekly_Sales"].mean().reset_index()
        hol_df["Category"] = hol_df["IsHoliday"].map({True: "Holiday", False: "Non-Holiday"})
        fig2 = px.bar(
            hol_df, x="Category", y="Weekly_Sales",
            color="Category",
            color_discrete_map={"Holiday": HOLIDAY_COLOR, "Non-Holiday": NON_HOLIDAY_COLOR},
            labels={"Weekly_Sales": "Avg Weekly Sales ($)"},
            template="plotly_white",
        )
        fig2.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Holiday weeks average **${holiday_avg:,.0f}** vs non-holiday **${nonhol_avg:,.0f}** — a **{holiday_lift:+.1f}%** lift.")

    with c2:
        section_header("🏬 Sales by Store Type")
        type_df = df.groupby("Type")["Weekly_Sales"].mean().reset_index()
        fig3 = px.bar(
            type_df, x="Type", y="Weekly_Sales",
            color="Type",
            color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["success"]],
            labels={"Weekly_Sales": "Avg Weekly Sales ($)", "Type": "Store Type"},
            template="plotly_white",
        )
        fig3.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Type A are the largest stores (highest throughput). Type C are smallest.")

    # -- Monthly seasonality --
    section_header("📅 Monthly Sales Seasonality")
    df2 = get_engineered_data()
    month_df = df2.groupby("Month")["Weekly_Sales"].mean().reset_index()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_df["Month_Name"] = month_df["Month"].map(month_names)
    fig4 = px.bar(
        month_df, x="Month_Name", y="Weekly_Sales",
        color="Weekly_Sales",
        color_continuous_scale=["#B0D0F0", "#0071CE"],
        labels={"Weekly_Sales": "Avg Weekly Sales ($)", "Month_Name": "Month"},
        template="plotly_white",
    )
    fig4.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)


# ── 2. Dataset Explorer ──────────────────────────────────────────────────────
def page_dataset_explorer():
    df = get_merged_data()
    stores = get_stores()

    st.markdown("## 🗃️ Dataset Explorer")
    info_box("Explore the raw Walmart Kaggle dataset. All figures are from the genuine CSV files.")

    # Schema tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Missing Values", "📂 Data Preview", "🏪 Stores Info"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Total Records",   f"{len(df):,}")
        with c2: kpi_card("Features",        f"{len(df.columns)}")
        with c3: kpi_card("Date Range",      f"{df['Date'].min().date()} → {df['Date'].max().date()}")
        with c4: kpi_card("Unique Weeks",    f"{df['Date'].nunique()}")

        st.divider()
        col_info = pd.DataFrame({
            "Column":    df.columns.tolist(),
            "Dtype":     [str(df[c].dtype) for c in df.columns],
            "Nulls":     [int(df[c].isnull().sum()) for c in df.columns],
            "Null%":     [f"{df[c].isnull().mean()*100:.1f}%" for c in df.columns],
            "Unique":    [df[c].nunique() for c in df.columns],
            "Sample":    [str(df[c].iloc[0]) for c in df.columns],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    with tab2:
        null_counts = df.isnull().sum()
        null_df = null_counts[null_counts > 0].reset_index()
        null_df.columns = ["Feature", "Null Count"]
        null_df["Null %"] = (null_df["Null Count"] / len(df) * 100).round(2)
        null_df["Note"] = null_df["Feature"].map({
            "MarkDown1": "Promotional markdown — introduced Nov 2011",
            "MarkDown2": "Promotional markdown — introduced Nov 2011",
            "MarkDown3": "Promotional markdown — introduced Nov 2011",
            "MarkDown4": "Promotional markdown — introduced Nov 2011",
            "MarkDown5": "Promotional markdown — introduced Nov 2011",
            "CPI":       "Consumer Price Index — missing for some stores",
            "Unemployment": "Unemployment rate — missing for some stores",
        })

        if null_df.empty:
            st.success("No missing values in the merged dataset!")
        else:
            fig = px.bar(
                null_df, x="Feature", y="Null Count",
                color="Null %",
                color_continuous_scale=["#B0D0F0", COLORS["danger"]],
                text="Null %",
                template="plotly_white",
                labels={"Null Count": "Missing Rows"},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=380, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(null_df, use_container_width=True, hide_index=True)
        warn_box("MarkDown missing values are filled with 0 (with indicator flags) during preprocessing. CPI/Unemployment are imputed using per-store medians.")

    with tab3:
        store_filter = st.selectbox("Filter by Store", ["All"] + sorted(df["Store"].unique().tolist()))
        dept_filter  = st.selectbox("Filter by Dept",  ["All"] + sorted(df["Dept"].unique().tolist()))
        n_rows = st.slider("Rows to display", 10, 500, 50)
        preview_df = df.copy()
        if store_filter != "All":  preview_df = preview_df[preview_df["Store"] == int(store_filter)]
        if dept_filter  != "All":  preview_df = preview_df[preview_df["Dept"]  == int(dept_filter)]
        st.dataframe(preview_df.head(n_rows), use_container_width=True)
        st.caption(f"Showing {min(n_rows, len(preview_df)):,} of {len(preview_df):,} filtered rows.")

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Store Directory")
            st.dataframe(stores, use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Size Distribution by Type")
            fig_box = px.box(
                stores, x="Type", y="Size",
                color="Type",
                color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["success"]],
                labels={"Size": "Store Size (sq ft)", "Type": "Store Type"},
                template="plotly_white",
            )
            fig_box.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig_box, use_container_width=True)


# ── 3. Sales Analytics ───────────────────────────────────────────────────────
def page_sales_analytics():
    df = get_engineered_data()

    st.markdown("## 📊 Sales Analytics")

    # Filters
    with st.expander("🔧 Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            all_stores = sorted(df["Store"].unique().tolist())
            sel_stores = st.multiselect("Stores", all_stores, default=all_stores[:5], key="sa_stores")
        with c2:
            all_depts = sorted(df["Dept"].unique().tolist())
            sel_depts = st.multiselect("Departments", all_depts, default=all_depts[:5], key="sa_depts")
        with c3:
            date_min = df["Date"].min().date()
            date_max = df["Date"].max().date()
            date_range = st.date_input("Date Range",
                value=(date_min, date_max),
                min_value=date_min, max_value=date_max, key="sa_dates")

    sel_df = df[df["Store"].isin(sel_stores) & df["Dept"].isin(sel_depts)]
    if len(date_range) == 2:
        sel_df = sel_df[(sel_df["Date"].dt.date >= date_range[0]) & (sel_df["Date"].dt.date <= date_range[1])]

    if sel_df.empty:
        st.warning("No data matching selected filters.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🏪 By Store", "🗂️ By Department", "🎄 Holiday Analysis"])

    with tab1:
        section_header("Weekly Sales Trend")
        weekly = sel_df.groupby(["Date", "IsHoliday"])["Weekly_Sales"].sum().reset_index()
        fig = px.line(weekly, x="Date", y="Weekly_Sales",
                      color="IsHoliday",
                      color_discrete_map={True: HOLIDAY_COLOR, False: NON_HOLIDAY_COLOR},
                      labels={"Weekly_Sales": "Total Sales ($)", "IsHoliday": "Holiday"},
                      template="plotly_white")
        fig.update_layout(height=360, legend_title="Holiday Week",
                          legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)

        # Monthly avg
        section_header("Monthly Average Sales")
        months = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                  7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        mon = sel_df.groupby("Month")["Weekly_Sales"].mean().reset_index()
        mon["Month_Name"] = mon["Month"].map(months)
        fig2 = px.bar(mon, x="Month_Name", y="Weekly_Sales",
                      color="Weekly_Sales",
                      color_continuous_scale=["#B0D0F0","#0071CE"],
                      template="plotly_white",
                      labels={"Weekly_Sales":"Avg Weekly Sales ($)","Month_Name":"Month"})
        fig2.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section_header("Sales by Store")
        store_agg = sel_df.groupby("Store")["Weekly_Sales"].agg(["mean","sum"]).reset_index()
        store_agg.columns = ["Store","Avg Weekly Sales","Total Sales"]
        fig = px.bar(store_agg.sort_values("Avg Weekly Sales", ascending=False),
                     x="Store", y="Avg Weekly Sales",
                     color="Total Sales",
                     color_continuous_scale=["#B0D0F0","#0071CE"],
                     template="plotly_white",
                     labels={"Avg Weekly Sales":"Avg Weekly Sales ($)","Store":"Store ID"})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        # Trend by store (top 5 by total)
        top5 = store_agg.nlargest(5,"Total Sales")["Store"].tolist()
        top5_df = sel_df[sel_df["Store"].isin(top5)]
        top5_trend = top5_df.groupby(["Date","Store"])["Weekly_Sales"].sum().reset_index()
        fig2 = px.line(top5_trend, x="Date", y="Weekly_Sales", color="Store",
                       template="plotly_white",
                       labels={"Weekly_Sales":"Weekly Sales ($)"},
                       title="Weekly Sales Trend — Top 5 Selected Stores")
        fig2.update_layout(height=340)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        section_header("Top Departments by Sales")
        dept_agg = sel_df.groupby("Dept")["Weekly_Sales"].mean().reset_index()
        dept_agg = dept_agg.sort_values("Weekly_Sales", ascending=False)
        n_depts = st.slider("Show top N departments", 5, 40, 20, key="dept_slider")
        fig = px.bar(dept_agg.head(n_depts),
                     x="Dept", y="Weekly_Sales",
                     color="Weekly_Sales",
                     color_continuous_scale=["#B0D0F0","#003087"],
                     template="plotly_white",
                     labels={"Weekly_Sales":"Avg Weekly Sales ($)","Dept":"Department ID"})
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Sales distribution
        section_header("Weekly Sales Distribution")
        fig_hist = px.histogram(sel_df, x="Weekly_Sales", nbins=60,
                                 template="plotly_white",
                                 color_discrete_sequence=[COLORS["primary"]],
                                 labels={"Weekly_Sales":"Weekly Sales ($)"})
        fig_hist.update_layout(height=320)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(f"Distribution is right-skewed. Mean: ${sel_df['Weekly_Sales'].mean():,.0f} | Median: ${sel_df['Weekly_Sales'].median():,.0f}")

    with tab4:
        section_header("Holiday vs Non-Holiday Comparison")
        c1, c2 = st.columns(2)
        with c1:
            hol_avg  = sel_df.loc[sel_df["IsHoliday"],  "Weekly_Sales"].mean()
            nhol_avg = sel_df.loc[~sel_df["IsHoliday"], "Weekly_Sales"].mean()
            lift = (hol_avg - nhol_avg) / nhol_avg * 100 if nhol_avg > 0 else 0
            kpi_card("Holiday Avg Sales",     f"${hol_avg:,.0f}",  border_color=HOLIDAY_COLOR)
            kpi_card("Non-Holiday Avg Sales", f"${nhol_avg:,.0f}", border_color=NON_HOLIDAY_COLOR)
            kpi_card("Holiday Sales Lift",    f"{lift:+.1f}%", "vs non-holiday average",
                     border_color=COLORS["secondary"])

        with c2:
            fig_box = px.box(sel_df, x="IsHoliday", y="Weekly_Sales",
                             color="IsHoliday",
                             color_discrete_map={True: HOLIDAY_COLOR, False: NON_HOLIDAY_COLOR},
                             labels={"Weekly_Sales":"Weekly Sales ($)","IsHoliday":"Holiday Week"},
                             template="plotly_white")
            fig_box.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig_box, use_container_width=True)

        # Year-over-year holiday view
        hol_only = sel_df[sel_df["IsHoliday"]].groupby(["Year","Month"])["Weekly_Sales"].mean().reset_index()
        fig_hol = px.bar(hol_only, x="Month", y="Weekly_Sales", color="Year",
                         barmode="group",
                         labels={"Weekly_Sales":"Avg Sales ($)","Month":"Month","Year":"Year"},
                         template="plotly_white",
                         title="Holiday Week Average Sales by Month and Year")
        fig_hol.update_layout(height=340)
        st.plotly_chart(fig_hol, use_container_width=True)


# ── 4. Sales Forecasting ─────────────────────────────────────────────────────
def page_sales_forecasting():
    st.markdown("## 🔮 Sales Forecasting")
    info_box("Select a store, department, and date to generate a point sales forecast using the trained models.")

    if not models_exist():
        st.error("⚠️ Trained models not found. Please run `train.py` first.")
        return

    df = get_merged_data()
    stores = get_stores()
    feats  = get_features_df()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### ⚙️ Prediction Parameters")
        sel_store = st.selectbox("Store ID", sorted(df["Store"].unique().tolist()), key="fc_store")
        sel_dept  = st.selectbox("Department", sorted(df["Dept"].unique().tolist()), key="fc_dept")
        sel_date  = st.date_input("Prediction Week",
                                   value=pd.Timestamp("2012-11-09").date(),
                                   min_value=pd.Timestamp("2010-01-01").date(),
                                   max_value=pd.Timestamp("2014-12-31").date(),
                                   key="fc_date")
        sel_hol   = st.checkbox("Mark as Holiday Week", value=False, key="fc_hol")
        sel_model = st.radio("Model", ["Linear Regression", "Decision Tree", "Both"], key="fc_model")
        predict_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

    with c2:
        if predict_btn:
            with st.spinner("Generating forecast..."):
                try:
                    result = predict(sel_store, sel_dept,
                                     pd.Timestamp(sel_date), sel_hol)
                    lr_p = result["lr_pred"]
                    dt_p = result["dt_pred"]

                    st.markdown("### 📊 Forecast Results")
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        kpi_card("Linear Regression", f"${lr_p:,.2f}", "Weekly Sales Forecast",
                                 border_color=COLORS["lr"])
                    with rc2:
                        kpi_card("Decision Tree",     f"${dt_p:,.2f}", "Weekly Sales Forecast",
                                 border_color=COLORS["dt"])

                    # Historical comparison for this store/dept
                    hist = df[(df["Store"] == sel_store) & (df["Dept"] == sel_dept)].copy()
                    hist = hist.sort_values("Date")

                    if not hist.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hist["Date"], y=hist["Weekly_Sales"],
                            mode="lines", name="Historical Sales",
                            line=dict(color=COLORS["neutral"], width=1.5),
                        ))
                        # Forecast markers
                        for model_name, pred_val, col in [
                            ("LR Forecast", lr_p, COLORS["lr"]),
                            ("DT Forecast", dt_p, COLORS["dt"]),
                        ]:
                            if sel_model == "Linear Regression" and "LR" not in model_name:
                                continue
                            if sel_model == "Decision Tree" and "DT" not in model_name:
                                continue
                            fig.add_trace(go.Scatter(
                                x=[pd.Timestamp(sel_date)], y=[pred_val],
                                mode="markers+text",
                                name=model_name,
                                text=[f"${pred_val:,.0f}"],
                                textposition="top center",
                                marker=dict(size=16, color=col, symbol="star"),
                            ))
                        fig.update_layout(
                            title=f"Store {sel_store} · Dept {sel_dept} — Historical Sales & Forecast",
                            xaxis_title="Date", yaxis_title="Weekly Sales ($)",
                            template="plotly_white", height=380,
                            legend=dict(x=0.01, y=0.99),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No historical data available for this store/department combination.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        # -- Multi-week forecast range ---
        st.markdown("---")
        st.markdown("### 📅 Forecast Range (Multiple Weeks)")
        rc1, rc2 = st.columns(2)
        with rc1:
            rng_start = st.date_input("Range Start", value=pd.Timestamp("2012-11-01").date(), key="rng_s")
            rng_store = st.selectbox("Store", sorted(df["Store"].unique()), key="rng_store")
        with rc2:
            rng_end   = st.date_input("Range End",   value=pd.Timestamp("2013-04-01").date(), key="rng_e")
            rng_dept  = st.selectbox("Dept",  sorted(df["Dept"].unique()),  key="rng_dept")

        if st.button("Generate Range Forecast", use_container_width=True):
            with st.spinner("Computing range forecast..."):
                try:
                    fdf = predict_range(
                        rng_store, rng_dept,
                        pd.Timestamp(rng_start), pd.Timestamp(rng_end),
                    )
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=fdf["Date"], y=fdf["LR_Prediction"],
                        name="Linear Regression", mode="lines+markers",
                        line=dict(color=COLORS["lr"], dash="dash"),
                    ))
                    fig2.add_trace(go.Scatter(
                        x=fdf["Date"], y=fdf["DT_Prediction"],
                        name="Decision Tree", mode="lines+markers",
                        line=dict(color=COLORS["dt"]),
                    ))
                    fig2.update_layout(
                        title=f"Forecast Range: Store {rng_store} · Dept {rng_dept}",
                        xaxis_title="Week", yaxis_title="Forecasted Sales ($)",
                        template="plotly_white", height=380,
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    st.dataframe(fdf.style.format({"LR_Prediction": "${:,.2f}",
                                                    "DT_Prediction": "${:,.2f}"}),
                                 use_container_width=True)
                except Exception as e:
                    st.error(f"Range forecast failed: {e}")


# ── 5. Model Performance ─────────────────────────────────────────────────────
def page_model_performance():
    st.markdown("## 🏆 Model Performance")
    results = get_eval_results()

    if results is None:
        st.error("⚠️ Evaluation results not found. Please run `train.py` first.")
        return

    lr = results["Linear Regression"]
    dt = results["Decision Tree"]

    info_box("Models were evaluated on a chronological hold-out validation set covering the last 20% of weeks in the training data. No random shuffling was applied.")

    # --- Overall comparison table ---
    section_header("📋 Overall Performance Comparison")
    comp_data = {
        "Metric":              ["MAE ($)", "RMSE ($)", "R² Score"],
        "Linear Regression":   [f"${lr['overall']['MAE']:,.0f}",
                                 f"${lr['overall']['RMSE']:,.0f}",
                                 f"{lr['overall']['R2']:.4f}"],
        "Decision Tree":       [f"${dt['overall']['MAE']:,.0f}",
                                 f"${dt['overall']['RMSE']:,.0f}",
                                 f"{dt['overall']['R2']:.4f}"],
    }
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df.style.set_properties(**{"text-align": "center"}),
                 use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    lr_r2 = lr["overall"]["R2"]; dt_r2 = dt["overall"]["R2"]
    with c1:
        kpi_card("LR — MAE",  f"${lr['overall']['MAE']:,.0f}", border_color=COLORS["lr"])
        kpi_card("DT — MAE",  f"${dt['overall']['MAE']:,.0f}", border_color=COLORS["dt"])
    with c2:
        kpi_card("LR — RMSE", f"${lr['overall']['RMSE']:,.0f}", border_color=COLORS["lr"])
        kpi_card("DT — RMSE", f"${dt['overall']['RMSE']:,.0f}", border_color=COLORS["dt"])
    with c3:
        kpi_card("LR — R²",   f"{lr_r2:.4f}", border_color=COLORS["lr"])
        kpi_card("DT — R²",   f"{dt_r2:.4f}", border_color=COLORS["dt"])

    # --- Bar chart comparison ---
    section_header("📊 Metric Comparison Chart")
    chart_data = pd.DataFrame({
        "Model":  ["Linear Regression", "Decision Tree"] * 2,
        "Metric": ["MAE", "MAE", "RMSE", "RMSE"],
        "Value":  [lr["overall"]["MAE"], dt["overall"]["MAE"],
                   lr["overall"]["RMSE"], dt["overall"]["RMSE"]],
    })
    fig = px.bar(chart_data, x="Metric", y="Value", color="Model",
                 barmode="group",
                 color_discrete_map={"Linear Regression": COLORS["lr"],
                                     "Decision Tree":     COLORS["dt"]},
                 template="plotly_white",
                 labels={"Value": "Error ($)", "Metric": "Error Metric"},
                 text_auto=True)
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    # --- Holiday vs Non-holiday ---
    section_header("🎄 Holiday vs Non-Holiday Breakdown")

    rows = []
    for model_key, res in [("Linear Regression", lr), ("Decision Tree", dt)]:
        for period, pdata in [("Holiday", res["holiday"]), ("Non-Holiday", res["non_holiday"])]:
            rows.append({
                "Model":  model_key, "Period": period,
                "MAE":    pdata["MAE"], "RMSE": pdata["RMSE"],
                "R2":     pdata["R2"], "n":    pdata["n"],
            })
    breakdown_df = pd.DataFrame(rows)

    tab1, tab2 = st.tabs(["MAE Comparison", "RMSE Comparison"])
    with tab1:
        fig = px.bar(breakdown_df, x="Model", y="MAE", color="Period",
                     barmode="group",
                     color_discrete_map={"Holiday": HOLIDAY_COLOR, "Non-Holiday": NON_HOLIDAY_COLOR},
                     template="plotly_white",
                     labels={"MAE":"Mean Absolute Error ($)"},
                     text="MAE")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.bar(breakdown_df, x="Model", y="RMSE", color="Period",
                     barmode="group",
                     color_discrete_map={"Holiday": HOLIDAY_COLOR, "Non-Holiday": NON_HOLIDAY_COLOR},
                     template="plotly_white",
                     labels={"RMSE":"Root Mean Squared Error ($)"},
                     text="RMSE")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Gap analysis
    section_header("📐 Holiday Performance Gap Analysis")
    for model_name, res in [("Linear Regression", lr), ("Decision Tree", dt)]:
        gap = res["holiday_gap"]
        c1, c2 = st.columns(2)
        with c1:
            kpi_card(f"{model_name} — MAE Gap (Holiday − Non-Holiday)",
                     f"+${gap['MAE_gap']:,.0f}" if gap["MAE_gap"] else "N/A",
                     "Higher MAE = harder to predict holiday weeks",
                     border_color=HOLIDAY_COLOR)
        with c2:
            kpi_card(f"{model_name} — RMSE Gap",
                     f"+${gap['RMSE_gap']:,.0f}" if gap["RMSE_gap"] else "N/A",
                     "Higher RMSE = more variance in holiday week predictions",
                     border_color=HOLIDAY_COLOR)

    st.divider()
    info_box(
        "Holiday weeks have higher prediction errors because they exhibit unusual sales spikes "
        "(e.g. Thanksgiving, Christmas) that deviate significantly from typical weekly patterns. "
        "The Decision Tree handles this better than Linear Regression due to its ability to "
        "model non-linear, threshold-based interactions."
    )

    # -- Validation split info --
    section_header("📌 Validation Methodology")
    st.markdown("""
    | Parameter | Value |
    |:----------|:------|
    | Validation strategy | Chronological hold-out (no random shuffle) |
    | Validation window | Last **20%** of weeks (by date) |
    | Validation period | ~2012-04-20 onwards |
    | Train samples | ~338,738 store-dept-week records |
    | Validation samples | ~82,832 store-dept-week records |
    | Feature scaling | StandardScaler fitted on train set only; applied to validation |
    """)


# ── 6. Staffing Planner ──────────────────────────────────────────────────────
def page_staffing_planner():
    st.markdown("## 👥 Staffing Demand Planner")
    warn_box(
        "IMPORTANT: Staffing figures on this page are ESTIMATES based on configurable operational "
        "assumptions. This tool is a planning aid only. Actual staffing requirements depend on "
        "store policy, local labour agreements, and operational factors not captured in this dataset."
    )

    if not models_exist():
        st.error("⚠️ Trained models not found. Please run `train.py` first.")
        return

    df = get_merged_data()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("### ⚙️ Planning Parameters")
        plan_store = st.selectbox("Store ID", sorted(df["Store"].unique()), key="sp_store")
        plan_dept  = st.selectbox("Department", sorted(df["Dept"].unique()), key="sp_dept")
        plan_date  = st.date_input("Week Date",
                                    value=pd.Timestamp("2012-11-16").date(),
                                    key="sp_date")
        plan_hol   = st.checkbox("Holiday Week", value=False, key="sp_hol")

        st.markdown("### 🔢 Operational Assumptions")
        sales_per_hr = st.number_input(
            "Sales per Associate-Hour ($)",
            min_value=50.0, max_value=1000.0,
            value=float(DEFAULT_SALES_PER_HOUR), step=25.0, key="sp_sph",
            help="Expected USD revenue generated per associate-hour worked. Typical US retail: $200–$400/hr."
        )
        hrs_per_wk = st.number_input(
            "Working Hours per FTE/Week",
            min_value=10.0, max_value=60.0,
            value=float(DEFAULT_HOURS_PER_WEEK), step=1.0, key="sp_hrs",
            help="Standard weekly hours for a full-time-equivalent associate."
        )
        min_staff = st.number_input("Minimum Staff (any week)", min_value=1, max_value=20,
                                     value=2, step=1, key="sp_min")
        plan_model = st.radio("Forecast Model", ["Decision Tree", "Linear Regression"], key="sp_model")
        plan_btn = st.button("Calculate Staffing Estimate", type="primary", use_container_width=True)

    with c2:
        if plan_btn:
            with st.spinner("Computing staffing estimate..."):
                try:
                    preds = predict(plan_store, plan_dept, pd.Timestamp(plan_date), plan_hol)
                    fc_sales = preds["dt_pred"] if plan_model == "Decision Tree" else preds["lr_pred"]

                    est = estimate_staffing(fc_sales, sales_per_hr, hrs_per_wk, int(min_staff))
                    rec = staffing_recommendation(est["estimated_staff"], plan_hol)

                    st.markdown("### 📊 Staffing Estimate Results")
                    r1, r2 = st.columns(2)
                    with r1:
                        kpi_card("Forecasted Weekly Sales", f"${est['forecasted_sales']:,.2f}",
                                 f"Model: {plan_model}", border_color=COLORS["primary"])
                        kpi_card("Estimated Labour Hours",  f"{est['labour_hours']:,.1f} hrs",
                                 f"Sales ÷ ${sales_per_hr:.0f}/hr", border_color=COLORS["secondary"])
                    with r2:
                        kpi_card("FTE Equivalent",          f"{est['fte_staff']:.1f} FTE",
                                 f"Labour hrs ÷ {hrs_per_wk:.0f} hrs/wk")
                        kpi_card("Estimated Staff Required", f"{est['estimated_staff']} staff",
                                 f"Min: {min_staff}",
                                 border_color=COLORS["success"] if not plan_hol else HOLIDAY_COLOR)

                    st.markdown(f"""
                    <div class="info-banner">
                    <b>📋 Planning Recommendation:</b> {rec}
                    </div>
                    """, unsafe_allow_html=True)

                    # Donut chart
                    fig = go.Figure(go.Pie(
                        labels=["Labour Hours Used", "Available Hours (40 hrs × staff)"],
                        values=[est["labour_hours"],
                                max(0, est["estimated_staff"] * hrs_per_wk - est["labour_hours"])],
                        hole=0.55,
                        marker_colors=[COLORS["primary"], "#E6ECF5"],
                        textinfo="label+percent",
                    ))
                    fig.add_annotation(text=f"{est['estimated_staff']}<br>Staff",
                                       x=0.5, y=0.5, font_size=20, showarrow=False,
                                       font=dict(color="#003087", family="Arial Black"))
                    fig.update_layout(showlegend=True, height=350,
                                      title="Labour Hour Utilisation")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.caption(est["assumption_note"])

                except Exception as e:
                    st.error(f"Staffing estimation failed: {e}")

        # -- Multi-week staffing range --
        st.markdown("---")
        st.markdown("### 📅 Multi-Week Staffing Forecast")
        rc1, rc2, rc3 = st.columns(3)
        with rc1: rng_store2 = st.selectbox("Store", sorted(df["Store"].unique()), key="sp_rng_store")
        with rc2: rng_dept2  = st.selectbox("Dept",  sorted(df["Dept"].unique()),  key="sp_rng_dept")
        with rc3:
            rng_s2 = st.date_input("Start", value=pd.Timestamp("2012-11-01").date(), key="sp_rs")
            rng_e2 = st.date_input("End",   value=pd.Timestamp("2013-02-01").date(), key="sp_re")

        if st.button("Generate Multi-Week Staffing Forecast", use_container_width=True):
            with st.spinner("Computing..."):
                try:
                    fdf = predict_range(rng_store2, rng_dept2,
                                        pd.Timestamp(rng_s2), pd.Timestamp(rng_e2))
                    from src.staffing import staffing_from_forecast_df
                    fdf = staffing_from_forecast_df(fdf, "DT_Prediction", sales_per_hr, hrs_per_wk, int(min_staff))

                    fig_multi = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                              subplot_titles=["Forecasted Weekly Sales ($)",
                                                              "Estimated Staff Required"])
                    fig_multi.add_trace(go.Bar(x=fdf["Date"], y=fdf["DT_Prediction"],
                                               name="DT Forecast", marker_color=COLORS["dt"]), row=1, col=1)
                    fig_multi.add_trace(go.Bar(x=fdf["Date"], y=fdf["Estimated_Staff"],
                                               name="Est. Staff", marker_color=COLORS["primary"]), row=2, col=1)
                    fig_multi.update_layout(height=520, template="plotly_white", showlegend=False)
                    st.plotly_chart(fig_multi, use_container_width=True)
                    st.dataframe(
                        fdf.rename(columns={"LR_Prediction":"LR Forecast ($)",
                                             "DT_Prediction":"DT Forecast ($)",
                                             "Labour_Hours":"Labour Hours",
                                             "FTE_Staff":"FTE Staff",
                                             "Estimated_Staff":"Est. Staff"})
                           .style.format({"LR Forecast ($)":"${:,.0f}",
                                          "DT Forecast ($)":"${:,.0f}",
                                          "Labour Hours":"{:.1f}",
                                          "FTE Staff":"{:.2f}"}),
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Multi-week staffing forecast failed: {e}")


# ── 7. Inventory Planning Insights ──────────────────────────────────────────
def page_inventory_planning():
    st.markdown("## 📦 Inventory Planning Insights")
    info_box("Inventory planning insights are derived from historical sales patterns and forecasts. Use these as supplementary decision-support signals, not definitive procurement targets.")

    df = get_engineered_data()

    tab1, tab2, tab3 = st.tabs(["📈 Demand Patterns", "🎄 Holiday Preparation", "📊 Store Insights"])

    with tab1:
        section_header("Seasonal Demand Patterns")
        season_map = {1:"Winter", 2:"Spring", 3:"Summer", 4:"Autumn"}
        seas = df.groupby("Season")["Weekly_Sales"].mean().reset_index()
        seas["Season_Name"] = seas["Season"].map(season_map)
        fig = px.bar(seas, x="Season_Name", y="Weekly_Sales",
                     color="Season_Name",
                     color_discrete_sequence=[COLORS["primary"], COLORS["success"],
                                              COLORS["secondary"], "#9b59b6"],
                     template="plotly_white",
                     labels={"Weekly_Sales":"Avg Weekly Sales ($)","Season_Name":"Season"})
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, use_container_width=True)

        # Week of Year pattern
        woy = df.groupby("WeekOfYear")["Weekly_Sales"].mean().reset_index()
        fig2 = px.line(woy, x="WeekOfYear", y="Weekly_Sales",
                       template="plotly_white",
                       labels={"Weekly_Sales":"Avg Weekly Sales ($)","WeekOfYear":"Week of Year"},
                       color_discrete_sequence=[COLORS["primary"]])
        # Mark typical holiday weeks
        for hw in [6, 36, 47, 51, 52]:
            fig2.add_vline(x=hw, line_color=HOLIDAY_COLOR, line_dash="dot", opacity=0.5)
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Dotted red lines indicate typical holiday weeks (Super Bowl≈Wk6, Labor Day≈Wk36, Thanksgiving≈Wk47, Christmas≈Wk51-52).")

    with tab2:
        section_header("Holiday Demand Uplift by Department")
        hol_lift = df.groupby(["Dept","IsHoliday"])["Weekly_Sales"].mean().unstack().reset_index()
        hol_lift.columns = ["Dept", "Non_Holiday", "Holiday"]
        hol_lift = hol_lift.dropna()
        hol_lift["Uplift_%"] = ((hol_lift["Holiday"] - hol_lift["Non_Holiday"])
                                 / hol_lift["Non_Holiday"] * 100).round(1)
        top_uplift = hol_lift.nlargest(20, "Uplift_%")

        fig = px.bar(top_uplift, x="Dept", y="Uplift_%",
                     color="Uplift_%",
                     color_continuous_scale=["#B0D0F0", HOLIDAY_COLOR],
                     template="plotly_white",
                     labels={"Uplift_%":"Holiday Sales Uplift (%)","Dept":"Department ID"},
                     title="Top 20 Departments — Holiday Sales Uplift %")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Departments with the highest holiday uplift require the greatest inventory and staffing buffers around major holidays.")
        st.dataframe(top_uplift.sort_values("Uplift_%", ascending=False).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    with tab3:
        section_header("Store-Level Inventory Demand Profile")
        stores = get_stores()
        merged_store = (df.groupby("Store")["Weekly_Sales"].agg(["mean","std","max"])
                         .rename(columns={"mean":"Avg Weekly Sales",
                                          "std":"Sales Std Dev",
                                          "max":"Peak Weekly Sales"})
                         .reset_index()
                         .merge(stores, on="Store"))
        merged_store["CoV%"] = (merged_store["Sales Std Dev"] / merged_store["Avg Weekly Sales"] * 100).round(1)

        fig = px.scatter(
            merged_store, x="Avg Weekly Sales", y="Sales Std Dev",
            size="Peak Weekly Sales", color="Type",
            hover_name="Store",
            hover_data={"CoV%": True, "Size": True},
            color_discrete_map={"A": COLORS["primary"], "B": COLORS["secondary"], "C": COLORS["success"]},
            template="plotly_white",
            labels={"Avg Weekly Sales":"Avg Weekly Sales ($)",
                    "Sales Std Dev":"Sales Std Dev ($)",
                    "Type":"Store Type"},
            title="Store Sales Profile: Mean vs Variability (bubble = peak weekly sales)",
        )
        fig.update_layout(height=440)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Stores with high std-dev relative to their average (high CoV) need larger safety-stock buffers to handle demand volatility.")


# ── 8. About ─────────────────────────────────────────────────────────────────
def page_about():
    st.markdown("## ℹ️ About the Project")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        ### Project Summary
        **Retail Store Weekly Sales Forecasting and Staffing Demand Analysis System**
        is an interactive Machine Learning decision-support application built for
        the CSE252 Artificial Intelligence and Machine Learning course at Lovely
        Professional University.

        The application uses genuine historical data from the
        **Kaggle Walmart Recruiting – Store Sales Forecasting** competition to:
        - Compare two regression models: **Linear Regression** and **Decision Tree**
        - Evaluate performance on holiday vs. non-holiday weeks separately
        - Translate sales forecasts into practical staffing recommendations
        - Provide interactive retail analytics and business intelligence features

        ---

        ### Machine Learning Models

        | Model | Role | Scaling |
        |:---|:---|:---|
        | **Linear Regression** | Interpretable baseline; models linear feature relationships | StandardScaler applied |
        | **Decision Tree Regression** | Captures non-linear, hierarchical patterns | No scaling needed |

        ---

        ### Validation Methodology
        A strict **chronological / temporal train-validation split** is used.
        Training data covers **80%** of the earliest weeks; validation uses the
        remaining **20%** of the most recent weeks.
        **No random shuffling of time-series records is performed.**

        ---

        ### Technology Stack

        | Component | Technology |
        |:---|:---|
        | Language | Python 3.13 |
        | Data Processing | Pandas 3.1, NumPy 2.5 |
        | Machine Learning | Scikit-learn 1.9 |
        | Dashboard | Streamlit 1.66 |
        | Visualisation | Plotly 7.1 |
        | Model Persistence | Joblib 1.6 |
        """)
    with c2:
        st.markdown("### Student Information")
        st.info("""
        **Student Name:** Rehan Khan  
        **Roll Number:** 29  
        **Registration Number:** 12526856  
        **Batch:** K3P25RA  
        **Course:** CSE252 – Artificial Intelligence and Machine Learning  
        **Problem ID:** KAGGLE-WALMART-SALES  
        """)

        st.markdown("### Dataset Reference")
        st.info("""
        **Source:** Kaggle Competition  
        **Name:** Walmart Recruiting – Store Sales Forecasting  
        **URL:** kaggle.com/c/walmart-recruiting-store-sales-forecasting  
        **Training Records:** 421,570 weekly observations  
        **Stores:** 45 (Types A, B, C)  
        **Departments:** 81  
        **Date Range:** 2010-02-05 to 2012-10-26  
        """)

        st.markdown("### GitHub Repository")
        st.info("github.com/RehanK09/retail-sales-staffing-forecast")

    st.divider()

    st.markdown("### ⚠️ Project Limitations")
    st.markdown("""
    - The Kaggle test set (`test.csv`) does not contain actual sales labels; evaluation is
      performed on a chronological hold-out split from the training data only.
    - MarkDown promotional data (MarkDown1–5) was introduced only from November 2011, so
      earlier weeks have zero values for these features.
    - Staffing estimates are calculated from configurable assumptions and are **not**
      based on actual Walmart employee records (which are not part of the dataset).
    - Linear Regression performs significantly worse than Decision Tree on this dataset
      because weekly retail sales exhibit strong non-linear patterns.
    - The Decision Tree may overfit on certain store/department combinations with very
      few historical data points.
    - No deep learning or ensemble methods are used (as per course scope).
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px 0;'>
            <span style='font-size:2.5rem;'>🛒</span><br>
            <span style='font-size:1.1rem; font-weight:800; letter-spacing:0.03em;'>
                Walmart Sales<br>Forecasting
            </span><br>
            <span style='font-size:0.72rem; opacity:0.8;'>CSE252 · Rehan Khan · 12526856</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Navigation**")
        page = st.radio(
            label="Go to",
            options=[
                "🏠  Executive Overview",
                "🗃️  Dataset Explorer",
                "📊  Sales Analytics",
                "🔮  Sales Forecasting",
                "🏆  Model Performance",
                "👥  Staffing Planner",
                "📦  Inventory Planning",
                "ℹ️  About Project",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        # Quick model status
        if models_exist():
            st.success("✅ Models trained & ready")
            results = get_eval_results()
            if results:
                dt_r2 = results["Decision Tree"]["overall"]["R2"]
                lr_r2 = results["Linear Regression"]["overall"]["R2"]
                st.markdown(f"**Decision Tree R²:** `{dt_r2:.4f}`")
                st.markdown(f"**Linear Reg R²:** `{lr_r2:.4f}`")
        else:
            st.error("⚠️ Models not found. Run train.py.")

        st.markdown("---")
        st.markdown('<div class="footer-note">CSE252 Project · Nov 2026</div>',
                    unsafe_allow_html=True)

    # Route to selected page
    if "Executive Overview"   in page: page_executive_overview()
    elif "Dataset Explorer"   in page: page_dataset_explorer()
    elif "Sales Analytics"    in page: page_sales_analytics()
    elif "Sales Forecasting"  in page: page_sales_forecasting()
    elif "Model Performance"  in page: page_model_performance()
    elif "Staffing Planner"   in page: page_staffing_planner()
    elif "Inventory Planning" in page: page_inventory_planning()
    elif "About Project"      in page: page_about()


if __name__ == "__main__":
    main()
