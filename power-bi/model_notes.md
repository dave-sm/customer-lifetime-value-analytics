# Model Design Notes

Design rationale, coverage caveats, and reconciliation notes for the semantic model. Read alongside `dax_measures.txt`. The `.pbix` file is available on request; the model structure is fully documented below.

## 1. Model type & storage mode

- **Mode:** Import.
- **Rationale:** The dataset is static (Online Retail II, Dec 2009 - Dec 2011),
  small in Power BI terms (~800K fact rows, 5.8K dimension rows), and the
  report is a portfolio demonstration. Import gives full DAX capability.

## 2. Star schema
```

                          dimension_date
                          (calculated table)
                    Date · Year · Month · MonthNum
                          YearMonth · Quarter
                                  |
                                  |  1 : *
                                  |  fact.Date → dim.Date
                                  ▼

  dimension_customers   ← * : 1  fact_transactions
  CustomerID (PK)                Invoice · StockCode
  RFM · CLV · Cluster            Quantity · Price
  Segment                        Country · Date
  5,840 rows                     804,008 rows

  fact_transactions.CustomerID  →  dimension_customers.CustomerID
```
- many-to-one relationship from fact to date dimension, cross-filter direction: Single.
- Auxiliary tables not shown (not part of the star, connected via no relationships): `TopPct`, `revenue_comparisons`, `data_dictionary`, `uplift_simulation`.

### Relationship table

| From (fact side) | To (dimension side) | Cardinality | Cross-filter |
|------------------|---------------------|-------------|--------------|
| `fact_transactions.CustomerID` | `dimension_customers.CustomerID` | * : 1 | Single |
| `fact_transactions.Date` | `dimension_date.Date` | * : 1 | Single |
| `uplift_simulation.CustomerID` | `dimension_customers.CustomerID` | 1 : 1 (unidirectional to fact) | Both |

**Note on the third relationship:** In the current `.pbix`, the relationship
between `uplift_simulation.CustomerID` and `dimension_customers.CustomerID`
is registered as `one → one` with **both Directions** cross-filtering.
This is the correct behaviour for the uplift page: `uplift_simulation`
carries the pre-computed cumulative percentages that the uplift curve
needs, and it must respond to segment filters applied on
`dimension_customers` without creating a circular filter path back through
`fact_transactions`.

### Reasoning

- Every analytical measure routes through `fact_transactions` for
  transactional counts and revenue, or through `dimension_customers` for
  customer-level attributes and CLV estimates.
- `fact_transactions` filters to `dimension_customers` and
  `dimension_date` via single-directional many-to-one relationships. There
  is no snowflaking.
- `uplift_simulation` is deliberately a **pre-computed fact** - the
  cumulative percentages are computed in the pipeline (NB7) rather than in
  DAX, because a full cumulative-rank computation at the fact grain would
  be far more expensive than reading 20 pre-computed rows.

## 3. Tables in the model

### `fact_transactions` (Import, from Parquet)

The cleaned transaction table from NB1. One row per invoice line item.
804,008 rows.

Key columns used in the model:
- `Invoice` - distinct-counted to get order count.
- `Date` (Date-typed) - used for the relationship to `dimension_date`.
- `InvoiceDate` (datetime) - retained for drill-through only.
- `CustomerID` - FK to `dimension_customers`.
- `Quantity`, `Price`, `TotalPrice` - revenue math. `TotalPrice` is the
  source of truth for revenue.

`is_cancelled` is present but always `FALSE` in the cleaned data and is retained
as a defensive column so future refreshes with raw data would surface
cancellations rather than silently include them.

**Not used for analytics:** `StockCode`, `Description` are present for
drill-through only. No product-level measures exist in the report.

### `dimension_customers` (Import, from Parquet)

One row per unique `CustomerID`, 5,840 rows total. Merges:
- RFM features (RecencyDays, Frequency, Monetary) from NB3/NB6.
- Behavioral features (AvgOrderValue, AvgItemsPerOrder, TenureDays,
  PurchaseRate) from NB3.
- Segment labels (`RFM_Score`, `RFM_Segment`) from NB6.
- Clustering (`Cluster`) from NB6.
- Both CLV estimates (`clv_12m`, `clv_6m_pred`) from NB4/NB5.

**Denormalized on purpose.** In a transactional system, RFM and CLV would
be separate fact-like tables. Here they are collapsed into a single
customer dimension because every analytical query groups *by customer*,
not *by model*. This keeps the model flat and every visual fast.

### `dimension_date` (Calculated table)

Generated with `CALENDAR(MIN(fact_transactions[Date]), MAX(fact_transactions[Date]))`
and enriched with Year, MonthNum, Month (abbreviated), YearMonth, and
Quarter.

**Marked as a date table.** Time intelligence functions are not used (the
dataset is a fixed 24-month window), but marking the date table is still
best practice and lets Power BI's auto-date behaviour be disabled
correctly.

### `TopPct` (Calculated table)

DATATABLE("TopPct", DOUBLE, {{0.01}, {0.05}, {0.10}, {0.20}})


A disconnected parameter table driving the `[selected_customers]` measure
on Page 4. The four values are the targeting thresholds shown in the
uplift narrative. Adding a value requires updating the DATATABLE and
regenerating the parameter slicer, but **no measure change is required**
as `[selected_customers]` reads whatever value is selected.

### `revenue_comparisons` (Calculated table)

DATATABLE("Metric", STRING, {
{"Revenue Captured (top %)"},
{"Total Revenue"}
})


A two-row helper table for the side-by-side revenue comparison bar on
Page 4. The `[revenue_value]` measure switches on `SELECTEDVALUE(Metric)`
to route each bar to the right underlying measure.

### `data_dictionary` (Import, from Parquet)

Static column-definition reference from NB8. Not used in visuals. Kept in
the model so the report ships with a self-documenting field list.

## 4. Coverage & null handling

**Critical: nulls in CLV columns are structural, not data quality issues.**

| Column | Nulls | Share | Cause |
|--------|-------|-------|-------|
| `clv_12m` | 1,669 | 28.6% | One-time buyers - Gamma-Gamma requires ≥ 2 purchases. |
| `clv_6m_pred` | 908 | 15.5% | Acquired after the ML cutoff date (2011-06-09). |

**Implications for DAX:**
- `AVERAGE()` and `MEDIAN()` skip blanks automatically - coverage is
  implicit in the measure.
- `SUM()` treats blanks as zero - safe for revenue, but note that
  `[total_clv_12m]` is **not** an estimate of total future revenue; it is
  the sum over modelled customers only.
- `COALESCE(clv_12m, 0)` or similar should never be used. Substituting zero would
  create a false cluster of "zero value" customers at the bottom of every
  distribution and would drag every average down by an unknown amount.

## 5. RFM bucketing and rank ties

The notebook's RFM scoring uses `rank(method = 'first')` on the Quintile
variable. At exact tie boundaries, this resolves arbitrarily by row order.

The Power BI model does **not** recompute RFM scoring, it imports the
already-assigned `RFM_Segment` from the pipeline. This keeps the model
aligned with the notebook labels by construction, but means any future
change to the RFM logic must be made in NB6 and re-imported.

**Implication:** if a segment filter returns a different count in Power BI
than in the Streamlit app, the app is recomputing quintiles at runtime
while Power BI is importing the frozen labels. The Streamlit app is the
one that could diverge; Power BI is the authoritative version.

---

## 6. What is deliberately **not** in the model

- **Time intelligence measures (YTD, MoM, rolling averages).** The
  dataset is for a static 2 year period Dec 2009 - Dec 2011.
- **Customer acquisition cohort analysis.** FirstPurchase exists but no
  cohort dimension. Adding one is a documented future refinement.
- **Product dimension.** StockCode/Description are on the fact only.
  This report is customer-centric; product analytics is out of scope.
- **Row-level security (RLS).** This is a portfolio report with a single
  audience. For production purposes, an RLS role on
  `dimension_customers[Country]` would be the first addition.

## 7. Change log for future maintainers

When changing anything in this model, update in this order:

1. **Upstream (notebooks)** - change the pipeline, re-export, re-import.
2. **Model relationships** - if a column is renamed, update the
   relationship before any measure.
3. **Calculated columns** - any change to `recency_bands` or
   `cluster_label` requires remapping slicers on Pages 2 and 5.
4. **Measures** - append to `dax_measures.txt` in the same commit; do not
   silently rename hidden measures.
5. **Report pages** - if a measure referenced on a page is renamed, the
   visual breaks silently at the label level. Always spot-check Page 4
   after a targeting-measure change.
6. **Documentation** - update this file's `#5` if any cluster label changes.

## 8. Future Refinements to the Layout

1. **Add a rolling-origin validation page** should the data be updated in future. Current uplift page carries an explicit caveat until then.
2. **Log-scale toggle** on the distribution page for audiences comfortable
   with log axes; keeps linear view as default.
3. **Play axis on the trend chart** (year-over-year overlay) to make the
   Q4 pattern even more visible.
4. **Bookmarks** for "Executive view" vs "Analyst view" - its the same report,
   different default pages expanded.