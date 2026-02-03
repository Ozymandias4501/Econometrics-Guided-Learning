# Guide: 02_gdp_growth_and_recession_label

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01_data/02_gdp_growth_and_recession_label.ipynb`.

This data module builds the datasets used throughout the project: a macro panel (FRED) and a micro cross-section (Census/ACS).

### Key Terms (defined)
- **API endpoint**: a URL path that returns a specific dataset.
- **Caching**: saving raw responses locally so experiments are reproducible and fast.
- **Frequency alignment**: converting mixed-frequency series (daily/monthly/quarterly) onto a common timeline.
- **Quarter-end timestamp**: representing a quarter by its final date to make merges unambiguous.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Fetch GDP
- Complete notebook section: Compute growth
- Complete notebook section: Define recession label
- Complete notebook section: Define next-quarter target
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

### Deep Dive: GDP Growth Math + Technical Recession Labels

GDP is a *level* series (a quantity). A recession label requires turning levels into *growth rates*.

#### Key Terms (defined)
- **Level**: the raw value of a series (e.g., real GDP in chained dollars).
- **Growth rate**: percent change of a level over a period.
- **QoQ (quarter-over-quarter)**: compares GDP_t to GDP_{t-1}.
- **YoY (year-over-year)**: compares GDP_t to GDP_{t-4}.
- **Annualized growth**: converting a quarterly growth rate into an annual pace.
- **Technical recession (proxy)**: two consecutive negative quarters of QoQ GDP growth.

#### Growth formulas (and what they mean)
- QoQ percent growth:
  - `g_qoq[t] = 100 * (GDP[t]/GDP[t-1] - 1)`
- QoQ annualized percent growth:
  - `g_ann[t] = 100 * ((GDP[t]/GDP[t-1])**4 - 1)`
- YoY percent growth:
  - `g_yoy[t] = 100 * (GDP[t]/GDP[t-4] - 1)`

Why multiple growth measures?
- QoQ is more "responsive" but noisier.
- YoY is smoother but reacts later.

#### Label construction (this project)
- Recession at quarter t:
  - `recession[t] = 1 if (g_qoq[t] < 0 and g_qoq[t-1] < 0) else 0`
- Next-quarter target:
  - `target_recession_next_q[t] = recession[t+1]`

#### Critical limitation (interpretation)
- This is a clean teaching label, but it is **not** an official recession dating rule.
- Official recession dating uses multiple indicators and can disagree with the 2-quarter rule.

#### Python demo: label edge cases
```python
import pandas as pd

growth = pd.Series([1.0, -0.1, -0.2, 0.3, -0.1, -0.1])
recession = ((growth < 0) & (growth.shift(1) < 0)).astype(int)
target_next = recession.shift(-1)
pd.DataFrame({'growth': growth, 'recession': recession, 'target_next': target_next})
```

#### Macro caveat: revisions
- GDP is revised. If you re-fetch later, the computed label can change.
- That is one reason we emphasize caching raw data.


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
