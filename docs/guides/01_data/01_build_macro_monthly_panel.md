# Guide: 01_build_macro_monthly_panel

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01_data/01_build_macro_monthly_panel.ipynb`.

This data module builds the datasets used throughout the project: a macro panel (FRED) and a micro cross-section (Census/ACS).

### Key Terms (defined)
- **API endpoint**: a URL path that returns a specific dataset.
- **Caching**: saving raw responses locally so experiments are reproducible and fast.
- **Frequency alignment**: converting mixed-frequency series (daily/monthly/quarterly) onto a common timeline.
- **Quarter-end timestamp**: representing a quarter by its final date to make merges unambiguous.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Load series
- Complete notebook section: Month-end alignment
- Complete notebook section: Missingness
- Complete notebook section: Save processed panel
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

### Deep Dive: Frequency Alignment (Daily/Monthly/Quarterly)

Economic indicators arrive at different frequencies:
- GDP is quarterly.
- CPI/unemployment are monthly.
- Yield curve spreads can be daily.

To build a single modeling table, you must choose a timeline and convert everything onto it.

#### Key terms (defined)
> **Definition:** **Resampling** converts a time series to a new frequency (e.g., daily -> monthly).

> **Definition:** **Aggregation** combines many observations into one (mean/last/sum).

> **Definition:** **Forward-fill (ffill)** carries the last known value forward until updated.

> **Definition:** A **quarter-end timestamp** represents the quarter by its final date (used for merges).

#### Why month-end is a pragmatic choice
- Many macro series are monthly.
- Daily series can be summarized to month-end (`last`) or month-average (`mean`).
- Month-end timestamps make quarterly aggregation easier.

#### Python demo: daily -> month-end (last vs mean)
```python
import numpy as np
import pandas as pd

idx = pd.date_range('2020-01-01', periods=120, freq='D')

# Toy daily series
x_daily = pd.Series(np.random.default_rng(0).normal(size=len(idx)).cumsum(), index=idx)

# Two common summaries
x_me_last = x_daily.resample('ME').last()
x_me_mean = x_daily.resample('ME').mean()
```

#### Forward-fill (what it assumes)
Forward-fill assumes the last published value remains "true" until updated.
This is often reasonable for slow-moving series, but can hide missingness or create fake stability.

#### Debug checks
1. Index is datetime, sorted, unique.
2. After resampling, frequency looks right (ME for monthly, QE for quarterly).
3. Joins do not introduce unexpected NaNs.
4. You can explain why you chose mean vs last.

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
