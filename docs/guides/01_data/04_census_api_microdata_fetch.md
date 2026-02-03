# Guide: 04_census_api_microdata_fetch

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01_data/04_census_api_microdata_fetch.ipynb`.

This data module builds the datasets used throughout the project: a macro panel (FRED) and a micro cross-section (Census/ACS).

### Key Terms (defined)
- **API endpoint**: a URL path that returns a specific dataset.
- **Caching**: saving raw responses locally so experiments are reproducible and fast.
- **Frequency alignment**: converting mixed-frequency series (daily/monthly/quarterly) onto a common timeline.
- **Quarter-end timestamp**: representing a quarter by its final date to make merges unambiguous.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Browse variables
- Complete notebook section: Fetch county data
- Complete notebook section: Derived rates
- Complete notebook section: Save processed data
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

### GDP Growth Formulas
- QoQ growth (percent): `100 * (GDP_t / GDP_{t-1} - 1)`
- Annualized QoQ (percent): `100 * ((GDP_t / GDP_{t-1})^4 - 1)`
- YoY growth (percent): `100 * (GDP_t / GDP_{t-4} - 1)`

### Technical Recession Label (Proxy)
- We define a recession quarter as: two consecutive quarters of negative QoQ GDP growth.
- This is *not* an official recession dating rule, but a clear, computable proxy for teaching.

### Frequency Alignment and Aggregation Choices
- Monthly predictors aggregated to quarterly can use:
  - **quarter-average**: captures typical conditions across the quarter.
  - **quarter-end**: captures end-of-quarter conditions.
- Both can be defensible; the point is to choose intentionally and document the choice.


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
