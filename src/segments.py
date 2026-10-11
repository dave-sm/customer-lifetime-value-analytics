"""
RFM segmentation rules and action mappings.

The rule function is copied verbatim from NB6 so the app's segment
labels match the exported `customer_segments.csv` exactly.

Quantile boundaries for R/F/M bucketing are computed once at app startup
from `customers_features.csv` so hypothetical slider inputs are bucketed
against the same population boundaries used in the notebook.
"""

import numpy as np
import pandas as pd


# --- Rule-based RFM label (verbatim from NB6) --------------------------------
def rfm_label(r: int, f: int, m: int) -> str:
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 4 and f >= 3:
        return "Loyal Customers"
    if r >= 4 and f <= 2:
        return "New/Promising"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and f <= 2 and m >= 3:
        return "Need Attention"
    return "Hibernating/Low Value"


# --- Recommended actions per RFM segment (from NB6) ------------------------------
RFM_ACTIONS = {
    "Champions": (
        "Top priority. Dedicated account management, premium loyalty rewards, "
        "early access, referral programs. Goal: retain and maximize lifetime value."
    ),
    "Need Attention": (
        "Elevated priority. ML identifies latent near-term value despite long "
        "recency. Prioritize reactivation tests; test personalized win-back offers."
    ),
    "At Risk": (
        "Time-sensitive retention. Win-back campaigns with targeted offers, "
        "satisfaction surveys, personalized outreach within 90 days."
    ),
    "Loyal Customers": (
        "Growth opportunity. Upsell and cross-sell, tiered loyalty progression "
        "toward Champion status, personalized product recommendations."
    ),
    "New/Promising": (
        "Development focus. Onboarding sequences, first-repeat-purchase "
        "incentives, habit-forming engagement."
    ),
    "Hibernating/Low Value": (
        "Selective approach. Automated low-cost email win-back; accept natural "
        "attrition among one-time buyers; test on 2-3 purchase sub-group."
    ),
}

# --- Cluster labels + actions (from NB6) ---------------------------------------
CLUSTER_ACTIONS = {
    "VIP/Whales": (
        "Key account management. Individual relationship management; "
        "dedicated account manager. Not a programmatic campaign segment."
    ),
    "Bulk/Business Buyers": (
        "Key account management. Custom pricing and service agreements; "
        "account management rather than programmatic campaigns."
    ),
    "Core Customers": (
        "Operational backbone. Steady engagement; standard loyalty programs; "
        "predictable purchase cadence."
    ),
    "Low-Value/Dormant": (
        "Low-cost automated reactivation only. Accept natural attrition."
    ),
}

CLUSTER_LABEL_MAP = {
    0: "Core Customers",
    1: "Low-Value/Dormant",
    2: "Bulk/Business Buyers",
    3: "VIP/Whales",
}


# --- Quantile bucketing for hypothetical inputs -------------------------------
class RFMQuantiles:
    """
    Holds the R/F/M quintile boundaries computed from the population.

    Recency is inverted (recent = high score), matching NB6's
    `pd.qcut(-RecencyDays, 5, labels = [1..5])`.

    Frequency and Monetary use raw-value quantiles so that hypothetical
    slider inputs are bucketed in the same space they're entered.
    (NB6 uses `.rank(method = 'first')` which breaks ties arbitrarily;
    for hypothetical inputs we use the natural quintile boundaries.)
    """

    def __init__(self, customers_features: pd.DataFrame):
        cf = customers_features.copy()
        # Recency: negate so recent = high
        self.r_cuts = pd.qcut(-cf["RecencyDays"], 5, retbins = True)[1]
        # Frequency and Monetary: raw-value quantiles
        self.f_cuts = pd.qcut(cf["Frequency"], 5, retbins = True,
                              duplicates="drop")[1]
        self.m_cuts = pd.qcut(cf["Monetary"], 5, retbins = True,
                              duplicates="drop")[1]

    def score(self, recency_days, frequency, monetary):
        r = int(np.clip(np.digitize(-recency_days, self.r_cuts[1:-1]) + 1, 1, 5))
        f = int(np.clip(np.digitize(frequency, self.f_cuts[1:-1]) + 1, 1, 5))
        m = int(np.clip(np.digitize(monetary, self.m_cuts[1:-1]) + 1, 1, 5))
        return r, f, m

    def composite(self, r, f, m):
        return r * 100 + f * 10 + m