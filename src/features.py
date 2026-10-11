"""
Shared feature-engineering logic.

This module is the single source of truth for how a customer profile
becomes a model input row. It mirrors the formulas in Notebook 3
(Customer 360) and Notebook 5 (ML CLV) exactly. This ensures that the 
Streamlit app and notebooks do not drift.

Imported by the app.
"""

import pandas as pd


def build_feature_row(
    customer_id: float,
    recency_days: int,
    frequency: int,
    monetary: float,
    avg_items_per_order: float,
    tenure_days: int,
    customer_age_days: int,
    country: str,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the feature vector used in NB5.

    Derived fields (never entered by user):
        AvgOrderValue      = Monetary / Frequency
        PurchaseRateMonthly = Frequency / (TenureDays / 30 + 1/30)
                              with the zero-tenure override from NB3/NB5.

    Column order matches NB5's `X` (excluding CustomerID and Country for
    the numeric block - the pipeline handles Country via ColumnTransformer).
    """
    avg_order_value = monetary / frequency if frequency > 0 else 0.0

    if tenure_days == 0:
        # NB3/NB5 zero-tenure rule: set rate = frequency to avoid div-by-zero
        purchase_rate_monthly = float(frequency)
    else:
        purchase_rate_monthly = frequency / ((tenure_days / 30) + (1 / 30))

    row = {
        "CustomerID": customer_id,
        "Frequency": frequency,
        "Monetary": monetary,
        "AvgOrderValue": avg_order_value,
        "AvgItemsPerOrder": avg_items_per_order,
        "RecencyDays": recency_days,
        "TenureDays": tenure_days,
        "CustomerAgeDays": customer_age_days,
        "PurchaseRateMonthly": purchase_rate_monthly,
        "Country": country,
    }
    return pd.DataFrame([row])


def to_ml_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the feature row for the ML pipeline.

    The pipeline drops CustomerID via ColumnTransformer remainder = 'drop',
    so it can pass it through; but for cleanliness it is excluded here.
    Column order matches NB5's `numeric_columns` + `categorical_columns`.
    """
    ml_cols = [
        "Frequency", "Monetary", "AvgOrderValue", "AvgItemsPerOrder",
        "RecencyDays", "TenureDays", "CustomerAgeDays", "PurchaseRateMonthly",
        "Country",
    ]
    return df[ml_cols].copy()