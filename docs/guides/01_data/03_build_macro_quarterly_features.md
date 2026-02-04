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

### Core Data: Build Datasets You Can Trust

Modeling is downstream of data.
If you do not trust your data processing, you cannot trust your model output.

#### The three layers of a real data pipeline
> **Definition:** **Raw data** is the closest representation of what the source returned.

> **Definition:** **Processed data** is cleaned and aligned for analysis.

> **Definition:** A **modeling table** is the final table with features and targets aligned and ready for splitting.

#### Timing is part of the schema
When you build features, you are also defining "what was known when".
That is why frequency alignment and target shifting are first-class topics in this project.

#### A practical habit
Keep a simple data dictionary as you go:
- what is each column?
- what are its units?
- what frequency is it observed?
- what transformations did you apply?

### Deep Dive: Monthly -> Quarterly Features (No-Leakage Engineering)

Your target is quarterly, but many predictors are monthly/daily.
You must transform predictors into quarterly features that were available at the time.

#### Key terms (defined)
> **Definition:** **Aggregation** summarizes higher-frequency data into lower-frequency data (monthly -> quarterly).

> **Definition:** A **lag** uses past values as features (e.g., $x_{t-1}$).

> **Definition:** The **prediction horizon** is how far ahead you predict (here: next quarter).

#### Two common quarterly feature definitions
- Quarter-average: mean of monthly values inside the quarter.
- Quarter-end: last available value in the quarter.

Both can be defensible. The goal is to choose intentionally and document it.

#### Leakage risk: "inside the target quarter"
Be explicit about the timestamp meaning.
- If you predict recession_{t+1} using quarter t features, features can use info up to the end of quarter t.
- If you want a mid-quarter prediction (nowcasting), you need partial-quarter features.

#### Python demo: quarterly aggregation + lags (commented)
```python
import pandas as pd

panel_monthly = pd.read_csv('data/sample/panel_monthly_sample.csv', index_col=0, parse_dates=True)

# 1) Aggregate
q_mean = panel_monthly.resample('QE').mean()
q_last = panel_monthly.resample('QE').last()

# 2) Choose one representation (or keep both with prefixes)
q = q_mean.add_prefix('mean_')

# 3) Add lags (past-only)
q['mean_UNRATE_lag1'] = q['mean_UNRATE'].shift(1)
q['mean_UNRATE_lag2'] = q['mean_UNRATE'].shift(2)

# 4) Drop rows created by lagging
q = q.dropna()
```

#### Debug checks for leakage
1. All shifts should be non-negative (past-only): `shift(+k)`.
2. Target is shifted the correct direction (`shift(-1)` for next period label).
3. After dropping NaNs, the final table has aligned indices for X and y.

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
