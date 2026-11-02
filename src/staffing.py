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


# ---------------------------------------------------------------------------
# Default operational assumptions (configurable by user in the dashboard)
# ---------------------------------------------------------------------------
DEFAULT_SALES_PER_HOUR   = 300.0   # USD sales per associate-hour
DEFAULT_HOURS_PER_WEEK   = 40.0    # working hours per FTE per week
DEFAULT_MIN_STAFF        = 2       # minimum staff regardless of sales level


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

def estimate_staffing(
    forecasted_sales:   float,
    sales_per_hour:     float = DEFAULT_SALES_PER_HOUR,
    hours_per_week:     float = DEFAULT_HOURS_PER_WEEK,
    min_staff:          int   = DEFAULT_MIN_STAFF,
) -> dict:
    """
    Estimate weekly staffing requirements from a forecasted sales figure.

    Parameters
    ----------
    forecasted_sales : float
        Forecasted weekly sales in USD (from LR or DT model prediction).
    sales_per_hour : float
        Expected USD sales generated per associate-hour worked.
    hours_per_week : float
        Standard weekly working hours per full-time-equivalent associate.
    min_staff : int
        Minimum number of staff to schedule regardless of forecast.

    Returns
    -------
    dict with keys:
      forecasted_sales, labour_hours, fte_staff, estimated_staff,
      sales_per_hour, hours_per_week, min_staff
    """
    # Guard: no negative sales prediction should result in zero staff
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


# ---------------------------------------------------------------------------
# Batch staffing estimates from a forecast DataFrame
# ---------------------------------------------------------------------------

def staffing_from_forecast_df(
    forecast_df:    pd.DataFrame,
    sales_col:      str   = "DT_Prediction",
    sales_per_hour: float = DEFAULT_SALES_PER_HOUR,
    hours_per_week: float = DEFAULT_HOURS_PER_WEEK,
    min_staff:      int   = DEFAULT_MIN_STAFF,
) -> pd.DataFrame:
    """
    Apply staffing estimation across every row of a forecast DataFrame.

    Expected columns in forecast_df: Date, LR_Prediction, DT_Prediction
    Returns the same DataFrame with added columns:
      Labour_Hours, FTE_Staff, Estimated_Staff
    """
    df = forecast_df.copy()
    df["Labour_Hours"]    = (df[sales_col].clip(lower=0) / sales_per_hour).round(1)
    df["FTE_Staff"]       = (df["Labour_Hours"] / hours_per_week).round(2)
    df["Estimated_Staff"] = df["FTE_Staff"].apply(
        lambda x: max(min_staff, int(round(x)))
    )
    return df


# ---------------------------------------------------------------------------
# Planning recommendation based on forecast level
# ---------------------------------------------------------------------------

def staffing_recommendation(estimated_staff: int, is_holiday: bool) -> str:
    """Generate a plain-English staffing recommendation."""
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
