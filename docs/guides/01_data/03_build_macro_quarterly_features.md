# Guide: 03_build_macro_quarterly_features

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01_data/03_build_macro_quarterly_features.ipynb`.

This data module builds the datasets used throughout the project: a macro panel (FRED) and a micro cross-section (Census/ACS).

### Key Terms (defined)
- **API endpoint**: a URL path that returns a specific dataset.
- **Caching**: saving raw responses locally so experiments are reproducible and fast.
- **Frequency alignment**: converting mixed-frequency series (daily/monthly/quarterly) onto a common timeline.
- **Quarter-end timestamp**: representing a quarter by its final date to make merges unambiguous.


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Aggregate monthly -> quarterly
- Complete notebook section: Add lags
- Complete notebook section: Merge with GDP/labels
- Complete notebook section: Save macro_quarterly.csv
- Fetch or load sample data and inspect schemas (columns, dtypes, index).
- Build the GDP growth series and recession label exactly as specified.
- Create a quarterly modeling table with `target_recession_next_q` and no obvious leakage.

### Alternative Example (Not the Notebook Solution)
```python
# Toy GDP growth + technical recession label (not real GDP):
import pandas as pd

idx = pd.date_range('2018-03-31', periods=12, freq='QE')
gdp = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 102, 103, 104, 105], index=idx)

growth_qoq = 100 * (gdp / gdp.shift(1) - 1)
recession = ((growth_qoq < 0) & (growth_qoq.shift(1) < 0)).astype(int)
target_next_q = recession.shift(-1)
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Core Data: build datasets you can trust (schema, timing, reproducibility)

Good modeling is downstream of good data. In econometrics, “good data” means more than “no missing values”:
it means the dataset matches the real timing and measurement of the economic problem.

#### 1) Intuition (plain English)

Most downstream mistakes trace back to one of these upstream issues:
- mixing frequencies incorrectly (monthly vs quarterly),
- misaligning timestamps (month-start vs month-end),
- using revised data as if it were known in real time,
- silently changing transformations (growth rate vs level) mid-project.

**Story example:** You merge monthly unemployment to quarterly GDP growth.
If you accidentally align the *end-of-quarter* unemployment with *start-of-quarter* GDP, you change the meaning of “what was known when.”

#### 2) Notation + setup (define symbols)

We will use the “time + horizon” language throughout the repo:
- $t$: time index (month/quarter/year),
- $X_t$: features available at time $t$,
- $y_{t+h}$: outcome/label defined $h$ periods ahead.

Data pipeline layers (repo convention):

1) **Raw data** (`data/raw/`)
- closest representation of the source response (API output, raw CSV)
- should be cacheable and re-creatable

2) **Processed data** (`data/processed/`)
- cleaned, aligned frequencies, derived columns
- “analysis-ready” but not necessarily “model-ready”

3) **Modeling table**
- final table where:
  - features are past-only,
  - labels are shifted to the forecast horizon,
  - missingness from lags/rolls is handled,
  - you can split without leakage.

#### 3) Assumptions (and why we state them explicitly)

Every dataset embeds assumptions:
- measurement units (percent vs fraction),
- seasonal adjustment,
- timing conventions (month-end, quarter-end),
- whether revisions matter (real-time vs revised),
- whether the sample composition changes over time.

You cannot remove assumptions; you can only make them explicit and test sensitivity.

#### 4) Mechanics: the minimum reliable pipeline

**(a) Ingest + cache**
- Always cache API responses to disk.
- Your code should be able to re-run without changing results.

**(b) Parse + type**
- Parse dates, set a proper index (DatetimeIndex for macro; MultiIndex for panels).
- Coerce numeric columns to numeric types (watch out for strings).

**(c) Align frequency**
- Resample to a common timeline (month-end, quarter-end).
- Decide and document whether you use `.last()` or `.mean()` (interpretation differs).

**(d) Create derived variables**
- growth rates, log differences, rolling windows, lag features.
- each transform changes interpretation; keep a small “data dictionary.”

**(e) Build the modeling table**
- define labels (e.g., next-quarter recession),
- shift targets forward and features backward (lags),
- drop missing rows created by lags/rolls.

#### 5) Inference and data (why timing affects standard errors)

Even inference topics (SE, p-values) depend on data structure:
- time series residuals are autocorrelated → HAC SE,
- panels share shocks within groups → clustered SE.

So “data” is not just a preprocessing step; it determines the correct inference method.

#### 6) Diagnostics + robustness (minimum set)

1) **Schema + units check**
- print `df.dtypes`, confirm units (percent vs fraction), and inspect summary stats.

2) **Index + frequency check**
- confirm sorted index, expected frequency, and no duplicate timestamps.

3) **Missingness check**
- print missingness per column before/after merges and transforms.

4) **Timing check**
- for a few rows, manually verify that features come from the past relative to the label.

5) **Sensitivity check**
- re-run a result using an alternative alignment (mean vs last) and see if conclusions change.

#### 7) Interpretation + reporting

When you present a result downstream, always include:
- which dataset version you used (processed vs sample),
- the frequency and timestamp convention,
- key transformations (diff, logdiff, growth rates),
- any known limitations (revisions, breaks).

**What this does NOT mean**
- “More features” does not equal “better data.”
- A perfectly clean dataset can still be conceptually wrong if timing is wrong.

#### Exercises

- [ ] For one dataset, write a 5-line data dictionary (column meaning + units + frequency).
- [ ] Demonstrate how `.last()` vs `.mean()` changes a resampled series and interpret the difference.
- [ ] Pick one merge/join and verify alignment by printing a few timestamps and values.
- [ ] Show one example where a shift direction would create leakage, and explain why.

### Deep Dive: Quarterly feature engineering — lags, changes, and rolling windows

Feature engineering on macro panels is largely about **timing discipline**.

#### 1) Intuition (plain English)

In macro forecasting, the most useful features are often:
- lags (what happened recently),
- changes (growth rates),
- rolling summaries (recent averages/volatility).

But every transform can introduce leakage if done incorrectly.

#### 2) Notation + setup (define symbols)

Let $x_t$ be a quarterly series.

Common transforms:

- Lag:
$$
x_{t-1}
$$

- Difference:
$$
\\Delta x_t = x_t - x_{t-1}
$$

- Percent change (approx):
$$
\\%\\Delta x_t \\approx \\frac{x_t - x_{t-1}}{x_{t-1}}
$$

- Rolling mean over $k$ quarters (past-only):
$$
\\overline{x}_{t,k} = \\frac{1}{k}\\sum_{j=0}^{k-1} x_{t-j}
$$

#### 3) Assumptions

These transforms assume:
- the series is measured comparably over time,
- timestamps reflect when values are “known,”
- rolling windows use past data only (no centering).

#### 4) Mechanics: practical feature-building rules

1) Build features in a copy of the DataFrame and keep column naming consistent.
2) After creating lags/rolls, `dropna()` to get a clean modeling table.
3) Always validate that a lag feature at time $t$ uses data from $t-1$ (not $t+1$).

#### 5) Inference: transforms change dependence

Differencing can reduce nonstationarity but can increase noise.
Rolling means smooth noise but can increase persistence.

So transforms affect:
- stationarity tests,
- SE choices (HAC),
- model stability.

#### 6) Diagnostics + robustness (minimum set)

1) **Spot-check rows**
- print a few timestamps and verify lag alignment manually.

2) **Compare transforms**
- levels vs differences vs growth rates; do relationships change?

3) **Overfitting guard**
- more features increase variance; use time-aware evaluation.

#### 7) Interpretation + reporting

When reporting features, always specify:
- “lag 1 quarter,”
- “12-quarter rolling mean,” etc.

#### Exercises

- [ ] Create lag and rolling features for two macro indicators and verify alignment by printing rows.
- [ ] Compare a model using levels vs differences; interpret the change.
- [ ] Create a leakage feature intentionally (shift the wrong way) and show how it inflates performance.

### Project Code Map
- `src/fred_api.py`: FRED client (`fetch_series_meta`, `fetch_series_observations`, `observations_to_frame`)
- `src/census_api.py`: Census/ACS client (`fetch_variables`, `fetch_acs`)
- `src/macro.py`: GDP + labels (`gdp_growth_qoq`, `gdp_growth_yoy`, `technical_recession_label`, `monthly_to_quarterly`)
- `scripts/build_datasets.py`: end-to-end dataset builder
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Merging quarterly GDP with monthly predictors without explicit aggregation (silent misalignment).
- Using future quarterly features (e.g., lag -1) by accident.
- Forgetting that daily series need resampling before joining.

<a id="summary"></a>
## Summary + Suggested Readings

You now have a reproducible macro dataset with an explicit recession label and a micro dataset for cross-sectional inference.
From here, the project focuses on modeling and interpretation.


Suggested readings:
- FRED API documentation (series, observations)
- US Census API documentation (ACS endpoints, geography parameters)
- pandas documentation: resampling, merging/joining time series
