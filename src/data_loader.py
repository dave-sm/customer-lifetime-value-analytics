"""
Cached artifact loading.

Heavy imports (lifetimes, sklearn-adjacent) are deferred inside the
loader functions so that importing this module is fast. The Streamlit
app can then render its shell immediately and only pay the heavy cost
when a model is actually needed.

All artifacts live under ./data and ./models relative to the repo root.
"""

import joblib
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_customers_features() -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / "customers_features.csv",
        parse_dates=["FirstPurchase", "LastPurchase"],
    )


def load_customers_rfm() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "customers_rfm.csv")


def load_probabilistic_clv() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "customers_clv_probabilistic_12m.csv")


def load_ml_clv() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "customers_clv_ml_6m.csv")


def load_segments() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "customer_segments.csv")


def load_segment_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "segment_summary.csv")


def load_monthly_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "monthly_summary.csv")


def load_uplift() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "uplift_simulation.csv")


def load_bgf():
    # Import deferred: lifetimes pulls in scipy, which is slow to import.
    from lifetimes import BetaGeoFitter
    bgf = BetaGeoFitter()
    bgf.load_model(str(MODELS_DIR / "bgf.pkl"))
    return bgf


def load_ggf():
    from lifetimes import GammaGammaFitter
    ggf = GammaGammaFitter()
    ggf.load_model(str(MODELS_DIR / "ggf.pkl"))
    return ggf


def load_ml_pipeline():
    return joblib.load(MODELS_DIR / "ml_pipeline.pkl")