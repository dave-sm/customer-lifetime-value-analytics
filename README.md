# Customer Lifetime Value (CLV) Analytics for E-Commerce Retail

An end-to-end customer analytics and CLV modeling pipeline on the Online Retail II dataset (UCI / Kaggle), with an interactive Streamlit app that operationalizes the fitted models.

> **Live Demo:** [Customer Lifetime Value Analytics](https://customer-lifetime-value-analytics-dsm.streamlit.app/)

## Project Overview

This project implements a complete customer analytics and CLV modeling pipeline using the Online Retail II dataset from Kaggle. The work follows the sequence of a commercial data science engagement: raw transaction data moves through cleaning, analysis, modeling, validation, and final outputs formatted for business intelligence tools.

The endpoint is a validated customer dataset, two complementary CLV models, customer segmentation, a targeting uplift simulation, and an interactive Streamlit app that ships the fitted models and scores any customer profile in real time.

Rather than only describing the pipeline, this repo **operates** it.

## Business Problem

Retail organizations commonly face three interconnected challenges:

- Revenue is heavily concentrated among a small subset of customers.
- Descriptive reporting does not reveal who will drive future value.
- Marketing spend is allocated without a clear view of customer lifetime value.

This project answers four specific questions:

1. Which customers account for the majority of revenue?
2. Which customers will generate the most value over the next 6 to 12 months?
3. How should customers be grouped for retention offers versus upsell campaigns?
4. Where does a limited marketing budget produce the highest return?

## Dataset

**Source:** Online Retail II (UCI Machine Learning Repository, via Kaggle).  
**Granularity:** Transaction-level e-commerce data spanning December 2009 to December 2011.  
**Key fields:** `Invoice`, `InvoiceDate`, `CustomerID`, `StockCode`, `Description`, `Quantity`, `Price`, `Country`.

**Data quality considerations:** The dataset contains missing customer identifiers, canceled invoices, and pricing anomalies, thus data quality handling is the first critical step.

**Structural finding:** All negative-quantity transactions in this dataset are tied to canceled invoices (prefixed with "C"). After removing cancellations, the cleaned dataset contains zero returns. This is an empirical characteristic of the data, not a data loss error. Verified twice (NB1 and NB2).

## Headline Results

|Metric |Value |
|-------|------|
|Raw transactions |1,067,371 |
|Clean transactions |804,008 |
|Customers |5,840 |
|Countries |41 |
|Observation window |24 months (2009-12-01 → 2011-12-09).
|Genuine returns in data |0 - verified twice (NB1, NB2) |
|Revenue concentration | Top 24.2% of customers = 80% of revenue.|
|6-month revenue (eval scope) | £4,244,762 |
|ML MAE (held-out, n = 987) | £633.19 |
|ML MAE improvement vs baseline | 47.0% |
|Uplift at top 1% targeted | 32.7× vs random |
|Uplift at top 10% targeted |6.5× vs random |

## Project Architecture

### Notebooks (Kaggle / Python)

|# |Notebook |Purpose |Key Outputs |
|--|---------|--------|------------|
|0 |Project overview |Framing, structure, definitions | - |
|1 | `01_data_quality_and_cleaning` |Resolve cancellations, missing IDs, invalid pricing; produce canonical transaction table. |	 `transactions_clean.csv` |
|2 | `02_eda_sales_and_customers` |Revenue trends, Pareto concentration, order distribution, returns validation. | `monthly_summary.csv` |
|3 | `03_feature_engineering_customer_table` |Build Customer 360 table with RFM and behavioral features. | `customers_features.csv`, `customers_rfm.csv` |
|4 | `04_clv_probabilistic_12m` |BG/NBD + Gamma-Gamma for 12-month CLV estimation. | `customers_clv_probabilistic_12m.csv`, `bgf.pkl`, `ggf.pkl` |
|5 | `05_clv_ml_regression_6m` |Random Forest for 6-month revenue prediction with temporal validation. | `customers_clv_ml_6m.csv`, `ml_pipeline.pkl` |
|6 | `06_customer_segmentation` |RFM scoring and K-Means clustering with segment profiles. | `customer_segments.csv`, `segment_summary.csv` |
|7 | `07_uplift_simulation` |Targeting uplift analysis comparing model-driven vs random selection. | `uplift_simulation.parquet`, `uplift_simulation.csv` |
|8 | `08_powerbi_exports` |Star schema fact/dimension tables, date dimension, data dictionary. | Power BI export set. |

## Streamlit App

Five tabs, meant to be read in order:

1. **Project & Pipeline** - framing, data foundation, headline facts.
2. **Customer Scoring** - live scoring of any customer profile.
3. **Targeting Simulator** - how CLV ranking converts to incremental revenue.
4. **Model & Evaluation** - parameters, metrics, and honest caveats.
5. **About & Links** - repo, notebooks, contact.

## Power BI

- Star-schema data model (`FactTransactions`, `DimCustomer`, `DimDate`).
- Executive dashboards.
- Segmentation and targeting analytics.
- Customer drill-through profiles.

## GitHub

- Final notebooks.
- Streamlit app source and fitted model artifacts.
- Power BI files.
- Exported CSVs and Parquet files.
- Documentation and visuals.

## Two CLV Models, Deliberately

The project runs two complementary CLV models, not one. Together they cover the two timelines on which marketing decisions actually get made.

### 12-Month Probabilistic CLV

Classical marketing science approach using:

- **BG/NBD** for repeat purchase frequency prediction.
- **Gamma-Gamma** for monetary value estimation.
- Combined into 12-month discounted CLV (1% monthly discount rate).

**Coverage:** 4,171 of 5,840 customers (71.4% - repeat buyers only). One-time buyers fall outside Gamma-Gamma's validity domain.

**Key results:**

- Mean CLV: £1,808
- Median CLV: £725
- Mean-to-median ratio of 2.5X confirms heavy right-skew.

### 6-Month ML CLV

Supervised learning approach predicting 6-month revenue using:

- Time-based cutoff (June 9, 2011) separating features from target.
- Random Forest Regressor (400 trees in notebooks; 100 trees in deployment).
- Temporal leakage prevention through strict feature-cutoff separation.

**Coverage:** 4,932 of 5,840 customers (84.4% - pre-cutoff customers only).

**Model performance (held-out test set):**

- MAE: £651 (£633 with deployed 100-tree variant).
- Median AE: £220
- RMSE: £3,504 (£3,538 with deployed variant).
- R²: 0.592 (0.585 with deployed variant).
- 45.5% MAE improvement over mean-prediction baseline.

## Data Cleaning and Preparation

Key cleaning steps:

- Removal of canceled invoices (prefixed with "C").
- Removal of records with missing `CustomerID`, `InvoiceDate`, or `Invoice`.
- Filtering of zero and negative prices.
- Outlier removal at the 99.9th percentile for quantity and price.
- Investigation of negative quantities: all tied to cancellations, none retained in final dataset.
- Construction of `TotalPrice` as the single source of truth for transaction revenue.

**Output:** 804,008 transaction records across 5,840 unique customers.

## Feature Engineering: Customer 360 View

Transactions are aggregated in two stages (transactions → orders → customers) to prevent double-counting within multi-item invoices. Customer-level features include:

- **RecencyDays** - days since last purchase at snapshot.
- **Frequency** - number of unique orders.
- **Monetary** - total historical revenue.
- **AvgOrderValue** - mean revenue per order.
- **AvgItemsPerOrder** - mean items per order.
- **TenureDays** - days between first and last purchase.
- **CustomerAgeDays** - days since first purchase at snapshot.
- **PurchaseRateMonthly** - orders per month, with zero-tenure correction.

**Output:** 5,840 customer records with 11 features.

## Customer Segmentation

Two complementary approaches:

### RFM Scoring

Customers assigned to quintiles (1-5) on Recency, Frequency, and Monetary value. Composite scores map to six business-aligned segments:

|Segment |Customers |Share |Avg 12m CLV|
|--------|----------|------|-----------|
|Champions | 1,275 | 21.8% | £4,280 |
|Loyal Customers | 611 | 10.5% | £884 |
|New/Promising | 440 | 7.5% | £939 |
|At Risk | 827 | 14.2% | £381 |
|Need Attention | 241 | 4.1% | £337 |
|Hibernating/Low Value | 2,446 | 41.9% | £880 |

## K-Means Clustering

Four data-driven clusters based on standardized behavioral features:

|Cluster |Customers |Share |Profile |
|--------|----------|------|--------|
|Core Customers | 3,832 | 65.6% | Steady, predictable purchasing. |
|Low-Value/Dormant | 1,976 | 33.8% | Lapsed or one-time buyers. |
|Bulk/Business Buyers | 12 |0.2% |	Extraordinary purchase rate (31 orders per month). |
|VIP/Whales | 20 | 0.3% | Exceptional value (£166K average spend). |

**Key insight:** Need Attention segment shows higher short-term CLV (£532) than At Risk (£420) or Loyal (£369), suggesting latent near-term value that RFM scores alone would miss.

## Targeting and Uplift Analysis

The uplift simulation ranks customers by predicted 6-month CLV and compares revenue capture against random selection:

|Targeting Threshold |Revenue Captured |Lift vs Random |
|--------------------|-----------------|---------------|
|Top 1%| 33% | 33X |
|Top 5% | 52% | 10.5X |
|Top 10% | 65% | 6.5X |
|Top 20% | 79% | 4.0X |

**Caveat:** These results are in-sample. The model was trained on the same period being evaluated. Results represent model capability, not guaranteed field performance.

## Power BI Dashboard

The Power BI report includes:

- **Executive Overview** - revenue, customers, CLV KPIs.
- **Customer Segmentation** - RFM and cluster views.
- **CLV Distributions** - long-term vs short-term value.
- **Targeting and Revenue Uplift** - campaign planning.
- **Retention and Recency Analysis** - churn risk monitoring.
- **Customer Drill-Through Profiles** - individual customer views.

## Star Schema Data Model

- `fact_transactions` - transaction-level records (804,008 rows).
- `dimension_customers` - customer dimension (5,840 rows).
- `dimension_date` - date dimension for time intelligence.

**Relationships:**

- `fact_transactions.CustomerID` → `dimension_customers.CustomerID`
- `fact_transactions.DateKey` → `dimension_date.DateKey`

## Repository Structure
```
customer-lifetime-value-analytics/
    app.py                     # Streamlit app - 5 tabs
    src/
        constants.py           # Snapshot dates, thresholds, metrics
        features.py            # Shared feature engineering (NB3/NB5)
        inference.py           # Live scoring for both models
        segments.py            # RFM rules + action mappings
        data_loader.py         # Cached artifact loading
    models/                    # Fitted model artifacts (pkl)
        bgf.pkl
        ggf.pkl
        ml_pipeline.pkl
    data/
        ...               # Customer-level and monthly CSVs
    notebooks/
        00-project-overview.ipynb
        01-data-quality-and-cleaning.ipynb
        02-eda-sales-and-customers.ipynb
        03-feature-engineering-customer-table.ipynb
        04-clv-probabilistic-12m.ipynb
        05-clv-ml-regression-6m.ipynb
        06-customer-segmentation.ipynb
        07-uplift-simulation.ipynb
        08-powerbi-exports-and-data-dictionary.ipynb
        09-embedded-power-bi-dashboards.ipynb
    power-bi/
        dax_measures.txt
        model_notes.md
    .streamlit/
        config.toml
    requirements.txt
    README.md
```
## How to Read the Demo

Two things worth knowing before exploring the Streamlit app:

- **Two reference dates, not one.** Probabilistic CLV is anchored to the day after the final transaction (2011-12-10). ML CLV uses a stricter cutoff six months earlier (2011-06-09) to enforce leakage control.
- **Coverage is not 100%.** One-time buyers have no probabilistic CLV (outside Gamma-Gamma's domain). Customers acquired after the ML cutoff have no ML CLV. The app states this explicitly rather than hiding it.

## Key Findings

1. **Revenue is highly concentrated.** 24.2% of customers drive 80% of total revenue. The top 1% of ML-ranked customers capture 34% of 6-month revenue.
2. **Seasonality shapes customer value.** Monthly revenue shows consistent Q4 peaks and January slumps. CLV estimates must be interpreted within seasonal context.
3. **Customer behavior is deeply heterogeneous.** RFM and K-Means reveal distinct archetypes - from Champions (£4,280 avg CLV) to Hibernating (£880 avg CLV) - requiring segment-specific strategies.
4. **Probabilistic and ML CLV provide complementary lenses.** The 12-month BG/NBD + Gamma-Gamma model offers stable long-term estimates (median £725). The Random Forest model excels at short-term ranking (median error £220). Together they support strategic planning and tactical targeting.
5. **CLV-driven targeting delivers measurable uplift.** Targeting the top 5% of predicted customers captures 52% of near-term revenue (10.5X lift), enabling highly constrained campaigns to yield outsized returns.
6. **The Need Attention segment warrants elevated priority.** Despite RFM classification as low-value, these customers show short-term CLV (£532) exceeding At Risk (£420) and Loyal (£369), indicating latent re-engagement potential.

## Key Analytical Decisions

**Returns were investigated, not assumed away.** Every negative-quantity row in the raw data is tied to a canceled invoice - there are zero genuine product returns. Removing cancellations removes all negative quantities. This was verified twice: once at cleaning (NB1), once against the working dataset (NB2).

**Leakage was controlled at the feature level.** All ML features are constructed from data observed on or before the cutoff date (2011-06-09). No future information enters training. Point-in-time discipline is enforced explicitly.

**Seasonality is treated as structural.** Q4 peaks, January-February troughs, confirmed across both observed years. CLV estimates are always shown with their reference date so seasonal context is never lost.

**Heavy-tail distributions drive modeling choices.** Order revenue skewness is 13.96; customer value is right-skewed. Model evaluation reports Median Absolute Error alongside MAE so the typical customer's error is not obscured by a few high-value outliers.

## Reproducibility Notes

**ML model version.** The shipped app uses a 100-tree RandomForest; the notebooks originally used 400. Same hyperparameters, data, split, and evaluation protocol. The reduction was made for deployment efficiency: 400 trees serializes to ~30 MB and adds ~20 seconds to cold start; 100 trees serializes to ~2 MB and loads in ~6 seconds locally (~2 s on Streamlit Cloud). Held-out metrics are equivalent within run-to-run variance: MAE £633 vs £651, RMSE £3,538 vs £3,504, R² 0.585 vs 0.592. No methodological change.

**Train/test split.** The ML split is random 80/20. Temporal separation is enforced at the feature level. A time-ordered split would be stricter validation; the current design accepts mild optimism for training stability.

**Uplift curve is in-sample.** The ML model was trained on data from the same period being evaluated. The uplift curve demonstrates ranking capability, not field performance. A rolling-origin validation would be required for out-of-sample estimates.

**RFM bucketing.** The app computes quintile boundaries from raw population values, reproducing NB6's segment labels for all sampled customers. At exact tie boundaries, the notebook's `.rank(method = 'first')` tie-break is arbitrary (row-order dependent); the app uses natural quantile boundaries.

## How to Reproduce

### Full pipeline

1. pip install -r requirements.txt
2. Run notebooks in order (00 through 08) on Kaggle or locally.
3. Export datasets from each notebook's output.
4. Download the exported CSVs and PKLs.
5. Place CSVs in `data/`, PKLs in `models/`.
6. Load Parquet or CSV files into Power BI.
7. Set relationships:   
    `fact_transactions.CustomerID` → `dimension_customers.CustomerID`   
    `fact_transactions.DateKey` → `dimension_date.DateKey`
8. Mark `dimension_date` as a date table.
9. Build dashboards using exported measures - see documentation in `power_bi/`.

## Streamlit app

- python -m streamlit run app.py

Requires the artifacts in `data/` and `models/`. The app does not read transactions_clean.csv - all inference is customer-level.

## Environment

Pinned to match the training environment:

- Python 3.12
- pandas 2.2.2, numpy 2.0.2, scipy 1.15.3
- scikit-learn 1.6.1, lifetimes 0.11.3, joblib 1.5.3
- Streamlit 1.40.1

## Future Refinements

The current pipeline provides a solid foundation for CLV-driven decision-making. Several extensions would further strengthen analytical rigor and operational effectiveness:

**1. Out-of-Sample Uplift Validation**  
The current uplift simulation is in-sample. A rolling-origin validation - where the model is trained on data up to a point and evaluated on subsequent periods - would provide field-realistic performance estimates. This requires a longer observation window or multiple cutoff points within the existing data.

**2. Log-Transformed Clustering**   
K-Means clustering currently produces extreme separation between core customers and VIP/whales due to the heavy right-skew in monetary features. Applying log transformations to Monetary and AvgOrderValue before scaling would likely yield more balanced clusters and reveal intermediate customer archetypes that are currently absorbed into the dominant core cluster.

**3. Split Hibernating/Low Value Segment**   
The Hibernating/Low Value segment contains 41.9% of the customer base and masks internal heterogeneity. Splitting this segment by purchase frequency - separating one-time buyers from customers with 2-3 historical orders - would distinguish natural attrition from recoverable lapsed customers, enabling more precise reactivation targeting.

**4. Incorporate Non-Transactional Engagement Signals**   
The current features derive exclusively from transaction history. Adding engagement signals such as email opens, site visits, or app usage would improve short-term CLV predictions, particularly for recently inactive customers where transactional data provides limited signal (should the data be available in future).

**5. Segment Stability Validation**   
Re-running the segmentation on different time windows would confirm that segment boundaries are stable and not artifacts of the specific observation period. This is particularly relevant for the small clusters (Bulk/Business Buyers, VIP/Whales) where a few customers can shift boundaries materially.

## Skills Demonstrated

- E-commerce / retail analytics.
- Data cleaning and validation.
- Feature engineering (two-stage aggregation).
- Probabilistic modeling (BG/NBD, Gamma-Gamma).
- Machine learning (Random Forest, temporal validation, baseline comparison).
- Customer segmentation (RFM, K-Means).
- Business impact simulation (uplift curves).
- Power BI data modeling (star schema).
- Interactive deployment (Streamlit).
- End-to-end project communication.

