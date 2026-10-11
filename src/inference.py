"""
Model inference for the Streamlit app.

Two scoring paths:
  1. Probabilistic 12-month CLV (BG/NBD + Gamma-Gamma).
  2. ML 6-month CLV (RandomForestRegressor pipeline).

Both accept the output of features.build_feature_row().

The lifetimes model objects (bgf, ggf) are passed in already-fitted,
so this module does not import lifetimes at module level.
"""

import pandas as pd
from src.constants import CLV_HORIZON_MONTHS, CLV_DISCOUNT_RATE_MONTHLY


# --- Probabilistic CLV (NB4 model) -------------------------------------------
def score_probabilistic(bgf, ggf, recency_days, frequency, tenure_days, monetary):
    """
    Compute 12-month probabilistic CLV for a single customer.

    Inputs must be in the lifetimes format:
        frequency = repeat purchases (total orders - 1).
        recency   = days between first and last purchase.
        T         = days from first purchase to snapshot.
        monetary  = average transaction value.

    Returns a dict with either {'clv': float} or {'error': str} if the
    customer falls outside the model's domain.
    """
    if frequency < 1:
        return {
            "error": (
                "One-time buyers (frequency = 0) fall outside the "
                "Gamma-Gamma model's validity domain. 12-month probabilistic "
                "CLV is not estimated for this profile."
            )
        }
    if monetary <= 0:
        return {"error": "Monetary value must be positive to score."}

    # lifetimes' customer_lifetime_value expects pandas Series, not scalars.
    # The single-customer inputs are wrapped in one-element Series.
    freq_s = pd.Series([frequency])
    rec_s = pd.Series([recency_days])
    T_s = pd.Series([tenure_days])
    mon_s = pd.Series([monetary])

    clv = ggf.customer_lifetime_value(
        bgf,
        frequency = freq_s,
        recency = rec_s,
        T = T_s,
        monetary_value = mon_s,
        time = CLV_HORIZON_MONTHS,
        freq = "D",
        discount_rate = CLV_DISCOUNT_RATE_MONTHLY,
    )
    # clv is a pandas Series; extract the first (and only) value
    return {"clv": float(clv.iloc[0])}

# --- ML CLV (NB5 model) ------------------------------------------------------
def score_ml(pipeline, ml_input: pd.DataFrame) -> dict:
    """
    Compute 6-month ML CLV for a single customer.

    ml_input must be the output of features.to_ml_input(), i.e. a
    single-row DataFrame with the exact feature columns used in NB5.
    """
    try:
        pred = pipeline.predict(ml_input)
        return {"clv": float(pred[0])}
    except Exception as e:
        return {"error": f"ML prediction failed: {e}"}