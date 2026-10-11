"""
Central constants for the CLV Streamlit app.
All dates, thresholds, and coverage facts are stated here so they are
never hard-coded in the UI layer.
"""

# --- Snapshot dates (reference points for each model) ---------------------
# Probabilistic (BG/NBD + Gamma-Gamma) - anchored to day after final transaction
PROBABILISTIC_SNAPSHOT = "2011-12-10"
# ML (6-month horizon) - cutoff is 6 months before final transaction
ML_CUTOFF = "2011-06-09"
ML_TARGET_END = "2011-12-09"

# --- Model hyperparameters (as fitted in NB4 / NB5) -----------------------
BG_NBD_PARAMS = {"r": 0.67, "alpha": 63.68, "a": 0.08, "b": 1.26,
                 "penalizer_coef": 0.001}
GAMMA_GAMMA_PARAMS = {"p": 11.74, "q": 0.89, "v": 11.71,
                      "penalizer_coef": 0.001}
ML_MODEL_PARAMS = {"n_estimators": 100, "min_samples_leaf": 2,
                   "max_features": "sqrt", "random_state": 42}

# --- Probabilistic CLV assumptions ----------------------------------------
CLV_HORIZON_MONTHS = 12
CLV_DISCOUNT_RATE_MONTHLY = 0.01  # 1% per month ≈ 12.7% annual

# --- Dataset facts (from NB1, NB2, NB3) -----------------------------------
DATASET_FACTS = {
    "raw_rows": 1_067_371,
    "clean_rows": 804_008,
    "unique_customers": 5_840,
    "countries": 41,
    "date_range": "2009-12-01 to 2011-12-09",
    "returns_rate": 0.0,
    "pareto_80pct_customer_share": 0.242,  # Top 24.2% = 80% of revenue
}

# --- Model coverage (which customers each model can score) ---------------
MODEL_COVERAGE = {
    "probabilistic_12m": {"customers": 4171, "pct": 71.4,
                          "excludes": "one-time buyers (frequency = 0)"},
    "ml_6m": {"customers": 4932, "pct": 84.4,
              "excludes": "customers acquired after the ML cutoff (2011-06-09)"},
}

# --- ML model performance (held-out test set, n = 987) -------------------
ML_PERFORMANCE = {
    "mae": 633.19,
    "median_ae": 221.63,
    "rmse": 3537.59,
    "r2": 0.5845,
    "baseline_mae": 1194.02,
    "baseline_rmse": 5489.26,
    "mae_improvement_pct": 47.0,
    "rmse_improvement_pct": 35.6,
}

# --- Uplift curve (from NB7) --------------------------------------------
UPLIFT_THRESHOLDS = [
    {"pct": 0.01, "captured": 0.327, "lift": 32.7},
    {"pct": 0.05, "captured": 0.524, "lift": 10.5},
    {"pct": 0.10, "captured": 0.649, "lift": 6.5},
    {"pct": 0.20, "captured": 0.792, "lift": 4.0},
]
TOTAL_6M_REVENUE_EVAL = 4_244_762.10  # £ - from NB7

# --- RFM segment distribution (from NB6) --------------------------------
RFM_SEGMENT_DISTRIBUTION = {
    "Hibernating/Low Value": 2446,
    "Champions": 1275,
    "At Risk": 827,
    "Loyal Customers": 611,
    "New/Promising": 440,
    "Need Attention": 241,
}