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

### Deep Dive: Monthly -> Quarterly Features (No-Leakage Engineering)

Your target is quarterly (GDP growth / recession label), but many predictors are monthly/daily.
You must transform predictors into quarterly features that were available at the time.

#### Key Terms (defined)
- **Aggregation window**: the period you summarize (e.g., quarter).
- **Quarter-average**: average monthly values within the quarter.
- **Quarter-end**: last available value in the quarter.
- **Lag**: a past value used as a feature (e.g., x_{t-1}).
- **Horizon**: how far ahead you are predicting (here: next quarter).

#### Two defensible quarterly feature definitions
- Quarter-average features (mean): captures typical conditions during the quarter.
- Quarter-end features (last): captures conditions at the end of the quarter.

#### Leakage risk: using information from inside the target quarter
Be explicit about what your prediction timestamp means.
- If you predict recession_{t+1} using quarter t features, ensure features only use information up to end of quarter t.
- If you predict *during* quarter t, you would need partial-quarter features (nowcasting).

#### Python demo: quarterly aggregation + lags
```python
import pandas as pd

panel_monthly = pd.read_csv('data/sample/panel_monthly_sample.csv', index_col=0, parse_dates=True)

q_mean = panel_monthly.resample('QE').mean()
q_last = panel_monthly.resample('QE').last()

# Example lag features
q = q_mean.add_prefix('mean_')
q['mean_UNRATE_lag1'] = q['mean_UNRATE'].shift(1)
q = q.dropna()
```

#### Debug checks for leakage
1. Are all lags non-negative (shift(+k))?
2. Does `target_recession_next_q` shift the correct direction?
3. Does the feature table end at the same quarter as the target (after dropna)?
4. If you recompute with a different aggregation (mean vs last), do conclusions change?


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
