"""
CLV Online Retail - End-to-End Interactive Demo
================================================

Five-tab Streamlit app that makes the full notebook pipeline
operable: from problem framing through live CLV scoring,
targeting simulation, and model evaluation.

Run locally:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data_loader as dl
from src.constants import (
    PROBABILISTIC_SNAPSHOT, ML_CUTOFF, DATASET_FACTS, MODEL_COVERAGE,
    ML_PERFORMANCE, UPLIFT_THRESHOLDS, TOTAL_6M_REVENUE_EVAL,
    RFM_SEGMENT_DISTRIBUTION, BG_NBD_PARAMS, GAMMA_GAMMA_PARAMS,
    ML_MODEL_PARAMS, CLV_DISCOUNT_RATE_MONTHLY,
)
from src.features import build_feature_row, to_ml_input
from src.inference import score_probabilistic, score_ml
from src.segments import (
    rfm_label, RFM_ACTIONS, CLUSTER_ACTIONS, CLUSTER_LABEL_MAP,
    RFMQuantiles,
)

# ------------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------------
st.set_page_config(
    page_title = "CLV Online Retail - End-to-End Demo",
    layout = "wide",
)


# ------------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------------
@st.cache_resource
def get_models():
    return dl.load_bgf(), dl.load_ggf(), dl.load_ml_pipeline()


@st.cache_data
def get_data():
    return {
        "customers_features": dl.load_customers_features(),
        "customers_rfm": dl.load_customers_rfm(),
        "prob_clv": dl.load_probabilistic_clv(),
        "ml_clv": dl.load_ml_clv(),
        "segments": dl.load_segments(),
        "segment_summary": dl.load_segment_summary(),
        "monthly_summary": dl.load_monthly_summary(),
        "uplift": dl.load_uplift(),
    }


@st.cache_resource
def get_rfm_quantiles(customers_features):
    return RFMQuantiles(customers_features)


# ------------------------------------------------------------------------
# Preset archetypes (Tab 2)
# ------------------------------------------------------------------------
PRESETS = {
    "Loyal High-Value": dict(recency = 30, frequency = 20, monetary = 25_000,
                              avg_items = 250, tenure = 500, age = 600,
                              country = "United Kingdom"),
    "At-Risk VIP": dict(recency = 400, frequency = 15, monetary = 20_000,
                         avg_items = 200, tenure = 500, age = 650,
                         country = "United Kingdom"),
    "New One-Time": dict(recency= 20, frequency = 1, monetary = 300,
                         avg_items = 25, tenure = 0, age = 20,
                         country = "United Kingdom"),
    "Lapsing Mid-Value": dict(recency = 250, frequency = 4, monetary = 1_500,
                               avg_items = 120, tenure = 300, age = 450,
                               country = "Germany"),
    "Bulk Buyer": dict(recency = 60, frequency = 8, monetary= 60_000,
                        avg_items = 2_000, tenure = 400, age = 500,
                        country = "Netherlands"),
    "Churned": dict(recency = 700, frequency = 5, monetary = 5_000,
                     avg_items = 150, tenure = 300, age = 720,
                     country = "United Kingdom"),
}

COUNTRIES = sorted(data["customers_features"].assign(
    Country = "United Kingdom")["Country"].unique().tolist()) if False else [
    "United Kingdom", "Germany", "France", "Netherlands", "Ireland",
    "Belgium", "Switzerland", "Spain", "Portugal", "Italy", "Sweden",
    "Norway", "Denmark", "Finland", "Austria", "Poland", "Iceland",
    "Other / Unknown",
]

# ------------------------------------------------------------------------
# Sidebar: warm-up control
# Preloads fitted models on demand so the first score is instant.
# ------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model status")
    if "models_warm" not in st.session_state:
        st.session_state.models_warm = False

    if not st.session_state.models_warm:
        st.caption("Models load on first use or click below to pre-load.")
        if st.button("Warm up models"):
            with st.spinner("Loading fitted models..."):
                _ = get_models()
                _ = get_rfm_quantiles(get_data()["customers_features"])
            st.session_state.models_warm = True
            st.rerun()
    else:
        st.success("Models ready")

# ========================================================================
# TABS
# ========================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 · Project & Pipeline",
    "2 · Customer Scoring",
    "3 · Targeting Simulator",
    "4 · Model & Evaluation",
    "5 · About",
])


# -- TAB 1 --------------------------------------------------------------
with tab1:
    monthly = dl.load_monthly_summary()
    st.title("Customer Lifetime Value - Online Retail")
    st.caption(
        "End-to-end CLV framework on the UCI/Kaggle Online Retail dataset. "
        "Two forward-looking models (12-month probabilistic, 6-month ML), one "
        "segmentation layer, one targeting simulator - all operable in this app. "
    )

    st.markdown("### Headline facts")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Clean transactions", f"{DATASET_FACTS['clean_rows']:,}")
    c2.metric("Customers", f"{DATASET_FACTS['unique_customers']:,}")
    c3.metric("Countries", DATASET_FACTS["countries"])
    c4.metric("Date range", "24 months")
    c5.metric("Returns", "0")
    c6.metric("6-mo revenue (eval)", f"£{TOTAL_6M_REVENUE_EVAL/1e6:.2f}M")

    st.markdown("---")
    st.markdown("### Why two CLV models")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "**12-month probabilistic**  \n"
            "BG/NBD (repeat purchasing) + Gamma-Gamma (transaction value). "
            "Stable, interpretable, strategic horizon. Suited to prioritization "
            "and budget allocation."
        )
    with col_b:
        st.markdown(
            "**6-month ML**  \n"
            "Random Forest on point-in-time features (cutoff "
            f"{ML_CUTOFF}). Non-linear, operational. Suited to campaign "
            "planning and near-term forecasting."
        )

    st.markdown("---")
    st.markdown("### Pipeline (7 notebooks → 12 artifacts → this app)")
    st.markdown(
        """
        1. **Cleaning** - 1.07M raw → 804k clean transactions (0 returns, verified).
        2. **EDA** - seasonality, Pareto (top 24.2% = 80% revenue), skewness.
        3. **Customer 360** - RFM + behavioral features, 5,840 customers.
        4. **Probabilistic CLV** - 12-month, 4,171 customers scored.
        5. **ML CLV** - 6-month, 4,932 customers scored (leakage-controlled cutoff).
        6. **Segmentation** - RFM rules + K-Means, 6 segments + 4 clusters.
        7. **Uplift simulation** - targeting curve, 32.7× lift at top 1%.
        """
    )

    st.markdown("---")
    st.markdown("### Data foundation")
    st.dataframe(pd.DataFrame({
        "Item": ["Raw rows", "Clean rows", "Retention rate", "Unique customers",
                "Countries", "Date range", "Returns after cleaning"],
        "Value": [
            f"{DATASET_FACTS['raw_rows']:,}",
            f"{DATASET_FACTS['clean_rows']:,}",
            f"{DATASET_FACTS['clean_rows']/DATASET_FACTS['raw_rows']*100:.1f}%",
            f"{DATASET_FACTS['unique_customers']:,}",
            f"{DATASET_FACTS['countries']}",
            DATASET_FACTS["date_range"],
            "0 - verified in NB2",
        ],
    }), hide_index = True, use_container_width = True)


# --- TAB 2 -----------------------------------------------------------------
with tab2:
    data = get_data()
    st.title("Customer Scoring - Live")
    st.caption(
        "Score any customer profile. Presets autofill realistic archetypes; "
        "adjust sliders to test hypothetical customers. The fitted models run "
        "in real time - no lookup table."
    )

    preset_name = st.selectbox("Preset archetype", ["Custom"] + list(PRESETS.keys()))
    preset = PRESETS.get(preset_name, {})

    st.markdown("#### Customer profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.slider("Recency (days since last purchase)",
                            1, 739, preset.get("recency", 95))
        frequency = st.slider("Frequency (total orders)",
                              1, 376, preset.get("frequency", 3))
    with col2:
        monetary = st.number_input("Monetary (£ total spend)",
                                    min_value=1.0, max_value=600_000.0,
                                    value=float(preset.get("monetary", 879)),
                                    step=50.0)
        avg_items = st.slider("Avg items per order",
                              1, 4803, preset.get("avg_items", 158))
    with col3:
        tenure = st.slider("Tenure (days: first → last purchase)",
                           0, 738, preset.get("tenure", 221))
        age = st.slider("Customer age (days since first purchase)",
                        1, 739, preset.get("age", 530))

    country = st.selectbox(
        "Country (representative)",
        COUNTRIES,
        index=COUNTRIES.index(preset.get("country", "United Kingdom"))
        if preset.get("country") in COUNTRIES else 0,
    )

    aov = monetary / frequency
    if tenure == 0:
        prm = float(frequency)
    else:
        prm = frequency / ((tenure / 30) + (1 / 30))

    with st.expander("Derived features (computed, not entered)"):
        st.write(f"AvgOrderValue = Monetary / Frequency = **£{aov:,.2f}**")
        st.write(f"PurchaseRateMonthly = Frequency / (Tenure/30 + 1/30) = **{prm:.4f}**")
        if tenure == 0:
            st.caption(
                "Zero-tenure override: same-day purchasers get rate = frequency "
                "to avoid division by zero. Matches NB3/NB5."
            )

    if st.button("Score customer", type = "primary"):
        _data = get_data()
        if not st.session_state.get("models_warm", False):
            with st.spinner("Loading fitted models (first time only)..."):
                _bgf, _ggf, _ml = get_models()
                _quantiles = get_rfm_quantiles(_data["customers_features"])
            st.session_state.models_warm = True
        else:
            _bgf, _ggf, _ml = get_models()
            _quantiles = get_rfm_quantiles(_data["customers_features"])

        _prob = score_probabilistic(
            _bgf, _ggf,
            recency_days = tenure,
            frequency = max(frequency - 1, 0),
            tenure_days = age,
            monetary = aov,
        )
        _row = build_feature_row(
            customer_id = -1,
            recency_days = recency,
            frequency = frequency,
            monetary = monetary,
            avg_items_per_order = avg_items,
            tenure_days = tenure,
            customer_age_days = age,
            country = country,
        )
        _ml_res = score_ml(_ml, to_ml_input(_row))
        _r, _f, _m = _quantiles.score(recency, frequency, monetary)
        _seg = rfm_label(_r, _f, _m)
        _rfm_score = _quantiles.composite(_r, _f, _m)

        st.session_state["t2_result"] = dict(
            prob = _prob, ml = _ml_res, seg = _seg,
            rfm_score = _rfm_score, r = _r, f = _f, m = _m,
        )

    if "t2_result" in st.session_state:
        res = st.session_state["t2_result"]

        st.markdown("---")
        st.markdown("### Results")

        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown(f"**12-Month Probabilistic CLV**  \n_as of {PROBABILISTIC_SNAPSHOT}_")
            if "error" in res["prob"]:
                st.warning(res["prob"]["error"])
            else:
                st.metric("Probabilistic CLV",
                          f"£{res['prob']['clv']:,.0f}",
                          label_visibility="collapsed")
        with colB:
            st.markdown(f"**6-Month ML CLV**  \n_features as of {ML_CUTOFF}_")
            if "error" in res["ml"]:
                st.error(res["ml"]["error"])
            else:
                st.metric("ML CLV",
                          f"£{res['ml']['clv']:,.0f}",
                          label_visibility = "collapsed")
        with colC:
            st.markdown("**RFM Segment**")
            st.metric("Segment", res["seg"],
                      delta = f"RFM score {res['rfm_score']}",
                      label_visibility = "collapsed")

        st.markdown("#### Recommended action")
        st.info(RFM_ACTIONS[res["seg"]])
        st.caption(
            f"R/F/M quintile scores: R = {res['r']}, F = {res['f']}, M = {res['m']} "
            "(bucketed against the full 5,840-customer population)."
        )

    with st.expander("Assumptions & coverage"):
        st.markdown(
            f"""
            - Probabilistic snapshot: **{PROBABILISTIC_SNAPSHOT}** ·
              ML cutoff: **{ML_CUTOFF}**
            - Discount rate: {CLV_DISCOUNT_RATE_MONTHLY*100:.1f}%/month
              (≈ {((1+CLV_DISCOUNT_RATE_MONTHLY)**12-1)*100:.1f}%/year)
            - BG/NBD parameters: {BG_NBD_PARAMS}
            - Gamma-Gamma parameters: {GAMMA_GAMMA_PARAMS}
            - ML model: {ML_MODEL_PARAMS}
            - **One-time buyers** (frequency = 0) have no probabilistic CLV -
              outside the Gamma-Gamma domain.
            - **New post-cutoff customers** have no ML CLV - no historical
              features at cutoff.
            - **Slider semantics vs lifetimes:** the 'Tenure' slider maps to 
              lifetimes' `recency` (first→last days); the 'Customer age' slider 
              maps to lifetimes' `T` (first→snapshot days). 'Recency' (days since 
              last purchase) is used by the ML model and RFM segmentation, not by 
              BG/NBD.
            - **RFM bucketing:** quintile boundaries are computed from the raw 
              population values. This reproduces NB6's segment labels for all 
              sampled customers. At exact tie boundaries (e.g., many customers 
              sharing Frequency = 1), the notebook's rank-based tie-break is 
              arbitrary; the app uses natural quantile boundaries.
            """
        )


# --- TAB 3 -----------------------------------------------------------------
with tab3:
    uplift_df = dl.load_uplift()
    seg_summary = dl.load_segment_summary()
    st.title("Targeting Simulator")
    st.caption(
        "Simulates the operational decision: which customers to target with "
        "a limited budget, ranked by predicted CLV."
    )

    uplift_df = data["uplift"].copy()

    pct_target = st.slider(
        "Target top X% of customers by predicted 6-month CLV",
        min_value = 1, max_value = 100, value = 10, step = 1,
    ) / 100.0

    # Capture at requested percentile
    idx = int(len(uplift_df) * pct_target) - 1
    idx = max(idx, 0)
    captured = uplift_df.loc[idx, "cum_actual_rev_pct"]
    lift = captured / pct_target
    captured_gbp = captured * TOTAL_6M_REVENUE_EVAL
    random_gbp = pct_target * TOTAL_6M_REVENUE_EVAL
    incremental_gbp = captured_gbp - random_gbp

    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue captured", f"{captured*100:.1f}%")
    c2.metric("Lift vs random", f"{lift:.1f}×")
    c3.metric("Incremental £ vs random", f"£{incremental_gbp:,.0f}")

    st.caption(
    f"Random targeting at {pct_target*100:.0f}% would capture "
    f"£{random_gbp:,.0f}. CLV-guided targeting captures "
    f"£{captured_gbp:,.0f} - an incremental £{incremental_gbp:,.0f}."
    )

    # Uplift curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = uplift_df["customers_pct"], y = uplift_df["cum_actual_rev_pct"],
        mode = "lines", name = "CLV-based targeting",
        line = dict(color = "#2E4057", width = 3),
    ))
    fig.add_trace(go.Scatter(
        x = [0, 1], y = [0, 1], mode = "lines", name = "Random baseline",
        line = dict(color = "#95A5A6", width = 2, dash = "dash"),
    ))
    fig.add_trace(go.Scatter(
        x = [pct_target], y = [captured], mode = "markers",
        marker = dict(color = "#C0392B", size = 12),
        name = f"Current: top {pct_target*100:.0f}%",
    ))
    fig.update_layout(
        xaxis_title = "Fraction of customers targeted",
        yaxis_title = "Fraction of 6-month revenue captured",
        xaxis = dict(range = [0, 0.3]), yaxis = dict(range = [0, 1]),
        height = 500, hovermode = "x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Cohort composition by RFM segment")
    seg_summary = data["segment_summary"].copy()
    seg_summary["Share of base"] = (
        seg_summary["Customers"] / seg_summary["Customers"].sum() * 100
    ).round(1)
    st.dataframe(
        seg_summary[["RFM_Segment", "Customers", "Share of base",
                      "AvgCLV12", "AvgCLV6"]]
        .rename(columns = {"AvgCLV12": "Avg 12M CLV (£)",
                          "AvgCLV6": "Avg 6M CLV (£)"}),
        hide_index = True, use_container_width = True,
    )

    st.caption(
    "Note: At Risk and Need Attention show higher short-term (6M) than "
    "long-term (12M) CLV - an inversion that signals near-term win-back "
    "potential worth testing."
    )

    st.warning(
        "**In-sample caveat.** The uplift curve is in-sample - the ML model "
        "was trained on data from the same period being evaluated. Results "
        "represent model ranking capability, not field performance. A "
        "rolling-origin validation would be needed for out-of-sample estimates."
    )


# --- TAB 4 ------------------------------------------------------------------
with tab4:
    st.title("Model & Evaluation")
    st.caption("Parameters, metrics, coverage, and honest caveats.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### BG/NBD - repeat purchasing")
        st.json(BG_NBD_PARAMS)
        st.caption(
            "Mean purchase rate ≈ 0.011/day (~4/year). "
            "Mean dropout probability ≈ 6% per purchase. "
            "Typical for non-contractual retail."
        )
        st.markdown("### Gamma-Gamma - transaction value")
        st.json(GAMMA_GAMMA_PARAMS)
        st.caption(
            "Mean transaction ≈ £70-80 for repeat buyers. "
            "q < 1: higher-frequency customers tend to spend slightly less per order."
        )
    with colB:
        st.markdown("### ML model - 6-month revenue forecast")
        st.json(ML_MODEL_PARAMS)
        st.caption(
            "Trained on 3,945 customers, tested on 987 held-out customers. "
            "Baseline = predict the training mean for every customer."
            )
        st.markdown("**Performance (held-out test, n = 987)**")
        st.dataframe(pd.DataFrame({
            "Metric": ["MAE", "Median AE", "RMSE", "R²",
                        "Baseline MAE", "Baseline RMSE",
                        "MAE improvement", "RMSE improvement"],
            "Value": [
                f"£{ML_PERFORMANCE['mae']:,.2f}",
                f"£{ML_PERFORMANCE['median_ae']:,.2f}",
                f"£{ML_PERFORMANCE['rmse']:,.2f}",
                f"{ML_PERFORMANCE['r2']:.4f}",
                f"£{ML_PERFORMANCE['baseline_mae']:,.2f}",
                f"£{ML_PERFORMANCE['baseline_rmse']:,.2f}",
                f"{ML_PERFORMANCE['mae_improvement_pct']:.1f}%",
                f"{ML_PERFORMANCE['rmse_improvement_pct']:.1f}%",
            ],
        }), hide_index = True, use_container_width = True)
        st.caption(
            "Median AE is the representative metric for the typical customer; "
            "MAE is pulled upward by a small number of high-value outliers."
        )

    st.markdown("---")
    st.markdown("### Uplift summary (from NB7)")
    st.dataframe(pd.DataFrame(UPLIFT_THRESHOLDS).rename(columns={
        "pct": "Top %", "captured": "Revenue captured", "lift": "Lift vs random",
    }).assign(**{
        "Top %": lambda d: (d["Top %"] * 100).astype(int).astype(str) + "%",
        "Revenue captured": lambda d: (d["Revenue captured"] * 100).round(1).astype(str) + "%",
        "Lift vs random": lambda d: d["Lift vs random"].round(1).astype(str) + "×",
    }), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Honest caveats")
    st.markdown(
        f"""
        - **Coverage boundaries.** Probabilistic CLV covers {MODEL_COVERAGE['probabilistic_12m']['pct']}%
          of customers (excludes {MODEL_COVERAGE['probabilistic_12m']['excludes']}).
          ML CLV covers {MODEL_COVERAGE['ml_6m']['pct']}%
          (excludes {MODEL_COVERAGE['ml_6m']['excludes']}).
        - **Random split.** The ML train/test split is random 80/20. Temporal
          separation is enforced at the feature level (cutoff {ML_CUTOFF}).
          A time-ordered split would be stricter validation.
        - **In-sample uplift.** NB7's uplift curve is in-sample. Ranking
          capability, not field performance.
        - **No returns in data.** Every negative-quantity row was tied to a
          cancelled invoice. Zero genuine returns - verified twice (NB1, NB2).
        - **Model version.** The shipped ML model is a 100-tree RandomForest;
          the notebook's original model used 400 trees. Same hyperparameters,
          data, and split. Reduced for deployment efficiency, 400 trees
          serializes to ~30 MB / 20 s cold start, 100 trees to ~2 MB / 6 s.
          Held-out metrics are equivalent within run-to-run variance.
        """
    )


# ── TAB 5 ───────────────────────────────────────────────────────────────
with tab5:
    st.title("About")
    st.markdown(
        """
        **Project.** End-to-end CLV framework on the Online Retail dataset
        (UCI/Kaggle). Notebooks cover data cleaning, EDA, feature engineering,
        two CLV models, segmentation, and uplift simulation.

        **This app** makes the pipeline operable. The fitted models are
        shipped and called live - this is not a lookup table.

        **Notebook index.**
        1. Data cleaning & preparation.
        2. Exploratory data analysis.
        3. Customer 360 table construction.
        4. Probabilistic CLV (BG/NBD + Gamma-Gamma).
        5. ML CLV (6-month horizon, leakage-controlled).
        6. Customer segmentation (RFM + K-Means).
        7. Targeting uplift simulation & export.

        """
    )

st.markdown("---")
st.caption(
        "CLV Online Retail · 7 pipeline notebooks · 12 artifacts · "
        "Built with Streamlit, scikit-learn, and lifetimes."
)    