"""
staffing.py — Translate forecasted weekly sales into staffing estimates.

This is a decision-support estimation tool, NOT a claim about actual
employee records. Assumptions are clearly configurable and labelled.

Formulae:
  Labour Hours = Forecasted Weekly Sales / Sales per Associate-Hour
  FTE Staff    = Labour Hours / Weekly Working Hours per Associate

Defaults are based on standard US retail benchmarks:
  - $300 sales per associate-hour is a typical full-price apparel / general
    merchandise productivity target.
  - 40 working hours per associate per week (full-time equivalent).
"""

import pandas as pd


# ===========================================================================
# [TOPIC: Operational Planning Assumptions & Default Benchmarks]
# ===========================================================================
DEFAULT_SALES_PER_HOUR   = 300.0   # Benchmark retail productivity (USD sales / associate-hour)
DEFAULT_HOURS_PER_WEEK   = 40.0    # Full-Time Equivalent (FTE) hours / associate / week
DEFAULT_MIN_STAFF        = 2       # Minimum operational staffing floor


# ===========================================================================
# [TOPIC: Sales-to-Labor Mathematical Formulation (FTE Staffing Estimation)]
# ===========================================================================

def estimate_staffing(
    forecasted_sales:   float,
    sales_per_hour:     float = DEFAULT_SALES_PER_HOUR,
    hours_per_week:     float = DEFAULT_HOURS_PER_WEEK,
    min_staff:          int   = DEFAULT_MIN_STAFF,
) -> dict:
    """
    Compute weekly staffing demand derived from sales forecast:
      1. Labour Hours  = Forecasted Sales / Sales per Hour
      2. FTE Staff     = Labour Hours / Hours per Week
      3. Recommended   = max(min_staff, round(FTE Staff))
    """
    # Guard against negative returns producing zero staff
    adj_sales = max(0.0, float(forecasted_sales))

    labour_hours    = round(adj_sales / sales_per_hour, 1)
    fte_staff       = round(labour_hours / hours_per_week, 2)
    estimated_staff = max(min_staff, int(round(fte_staff)))

    return {
        "forecasted_sales": round(adj_sales, 2),
        "labour_hours":     labour_hours,
        "fte_staff":        fte_staff,
        "estimated_staff":  estimated_staff,
        "sales_per_hour":   sales_per_hour,
        "hours_per_week":   hours_per_week,
        "min_staff":        min_staff,
        "assumption_note":  (
            f"ESTIMATE: Labour hours = Weekly Sales ÷ ${sales_per_hour:.0f}/hr. "
            f"FTE staff = Labour hours ÷ {hours_per_week:.0f} hrs/week. "
            "Actual staffing depends on store policy, overtime, and seasonal factors. "
            "This output is an operational planning aid, not a ground-truth requirement."
        ),
    }


# ===========================================================================
# [TOPIC: Batch Multi-Week Staffing Projections]
# ===========================================================================

def staffing_from_forecast_df(
    forecast_df:    pd.DataFrame,
    sales_col:      str   = "DT_Prediction",
    sales_per_hour: float = DEFAULT_SALES_PER_HOUR,
    hours_per_week: float = DEFAULT_HOURS_PER_WEEK,
    min_staff:      int   = DEFAULT_MIN_STAFF,
) -> pd.DataFrame:
    """
    Apply staffing estimation across an entire forecasted schedule DataFrame.
    """
    df = forecast_df.copy()
    df["Labour_Hours"]    = (df[sales_col].clip(lower=0) / sales_per_hour).round(1)
    df["FTE_Staff"]       = (df["Labour_Hours"] / hours_per_week).round(2)
    df["Estimated_Staff"] = df["FTE_Staff"].apply(
        lambda x: max(min_staff, int(round(x)))
    )
    return df


# ===========================================================================
# [TOPIC: Operational Staffing Level Heuristic & Policy Recommendations]
# ===========================================================================

def staffing_recommendation(estimated_staff: int, is_holiday: bool) -> str:
    """Generate business-logic planning advice based on predicted workforce requirements."""
    holiday_note = " — HOLIDAY WEEK: consider additional coverage." if is_holiday else ""
    if estimated_staff <= 5:
        tier = "Low-volume week. Minimum staffing recommended."
    elif estimated_staff <= 15:
        tier = "Moderate volume. Standard staffing levels appropriate."
    elif estimated_staff <= 30:
        tier = "High-volume week. Consider additional shift cover."
    else:
        tier = "Very high-volume week. Full staffing and supervisory support recommended."
    return tier + holiday_note
