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

### Deep Dive: GDP Growth Math + Technical Recession Labels

GDP is a level series. A recession label requires turning levels into growth rates.

#### Key terms (defined)
> **Definition:** A **level** is the raw value of a series (e.g., real GDP in chained dollars).

> **Definition:** A **growth rate** is the percent change over a period.

> **Definition:** **QoQ** (quarter-over-quarter) compares $GDP_t$ to $GDP_{t-1}$.

> **Definition:** **YoY** (year-over-year) compares $GDP_t$ to $GDP_{t-4}$.

> **Definition:** **Annualized QoQ** converts a quarterly growth rate into an annual pace.

#### Growth formulas (math)
QoQ percent growth:

$$
 g_{qoq,t} = 100 \cdot \left(\frac{GDP_t}{GDP_{t-1}} - 1\right)
$$

Annualized QoQ percent growth (quarterly compounding):

$$
 g_{ann,t} = 100 \cdot \left(\left(\frac{GDP_t}{GDP_{t-1}}\right)^4 - 1\right)
$$

YoY percent growth:

$$
 g_{yoy,t} = 100 \cdot \left(\frac{GDP_t}{GDP_{t-4}} - 1\right)
$$

#### Why compute multiple growth measures?
- QoQ is responsive but noisy.
- YoY is smoother but slower to react.
- Annualized QoQ is common in macro reporting.

> **Definition:** A **log growth rate** uses differences of logs: $\Delta \log(GDP_t) = \log(GDP_t) - \log(GDP_{t-1})$.
Log growth is often convenient because it approximates percent growth for small changes and makes compounding math cleaner.

#### Technical recession label used in this project
> **Definition:** A **technical recession** (teaching proxy here) is two consecutive quarters of negative QoQ GDP growth.

Label:

$$
 recession_t = \mathbb{1}[g_{qoq,t} < 0 \;\wedge\; g_{qoq,t-1} < 0]
$$

Next-quarter prediction target:

$$
 target_{t} = recession_{t+1}
$$

#### Edge cases (what to watch)
- Missing GDP values will propagate to growth.
- The first growth observation is undefined (needs a prior quarter).
- YoY growth needs 4 prior quarters.

#### Python demo: compute growth + label (commented)
```python
import pandas as pd

# gdp: Series of GDP levels indexed by quarter-end dates
# gdp = ...

# QoQ growth (percent)
# growth_qoq = 100 * (gdp / gdp.shift(1) - 1)

# Technical recession label
# Two consecutive negative quarters:
# - current quarter growth < 0
# - previous quarter growth < 0
# recession = ((growth_qoq < 0) & (growth_qoq.shift(1) < 0)).astype(int)

# Next-quarter target
# Predict next quarter's label using information as-of this quarter:
# target_next = recession.shift(-1)
```

#### Project touchpoints (where this logic lives in code)
- `src/macro.py` implements these transforms explicitly:
  - `gdp_growth_qoq`, `gdp_growth_qoq_annualized`, `gdp_growth_yoy`
  - `technical_recession_label`
  - `next_period_target`

#### Python demo: using the project helper functions (commented)
```python
from src import macro

# levels: quarterly GDP level series
# levels = gdp['GDPC1']

# Growth variants
# qoq = macro.gdp_growth_qoq(levels)
# yoy = macro.gdp_growth_yoy(levels)

# Label + next-period target
# recession = macro.technical_recession_label(qoq)
# target = macro.next_period_target(recession)
```

#### Important limitation
This is a clean, computable teaching proxy.
It is not an official recession dating rule.

#### Macro caveat: revisions
GDP is revised. If you re-fetch later, historical values can change, which can change your computed label.
This is one reason caching matters.

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
