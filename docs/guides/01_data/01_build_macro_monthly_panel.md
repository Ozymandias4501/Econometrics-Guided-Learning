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

### Deep Dive: Frequency Alignment (Daily/Monthly/Quarterly)

Economic indicators arrive at different frequencies:
- GDP is quarterly.
- CPI/unemployment are monthly.
- Yield curve spreads can be daily.

To build a single modeling table, you must choose a *timeline* and convert everything onto it.

#### Key Terms (defined)
- **Resampling**: converting a time series to a new frequency (e.g., daily -> monthly).
- **Aggregation**: combining many observations into one (mean/last/sum).
- **Forward-fill (ffill)**: carrying the last known value forward until updated.
- **As-of merge**: joining on the most recent available observation (common in finance).

#### Why month-end is a pragmatic choice
- Many macro series are monthly.
- Daily series can be summarized to month-end (`last`) or month-average (`mean`).
- Month-end timestamps make quarterly aggregation easier.

#### Python demo: daily -> month-end (last vs mean)
```python
import numpy as np
import pandas as pd

idx = pd.date_range('2020-01-01', periods=120, freq='D')
x_daily = pd.Series(np.random.default_rng(0).normal(size=len(idx)).cumsum(), index=idx)

x_me_last = x_daily.resample('ME').last()
x_me_mean = x_daily.resample('ME').mean()
```

#### Missing data and forward-fill (what it assumes)
- Forward-fill assumes the last published value remains "true" until updated.
- This is often reasonable for policy rates between meetings, but can be questionable for volatile indicators.
- Always inspect long gaps: forward-filling across multi-month gaps can create fake stability.

#### Debug checks
- Are timestamps sorted and unique?
- After resampling, do you have the expected frequency (`ME` rows, `QE` rows)?
- Do joins create NaNs? If so, which series and why?


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
