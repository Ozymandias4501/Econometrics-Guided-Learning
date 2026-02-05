# Guide: 00_stationarity_unit_roots

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/08_time_series_econ/00_stationarity_unit_roots.ipynb`.

This module covers classical time-series econometrics: stationarity, cointegration/ECM, and VAR/IRFs.

### Key Terms (defined)
- **Stationarity**: stable statistical properties over time.
- **Unit root**: nonstationary process where shocks accumulate (random walk-like).
- **Cointegration**: nonstationary series with a stationary long-run relationship.
- **VAR**: multivariate autoregression.
- **IRF**: impulse response function (shock propagation over time).


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Load macro series
- Complete notebook section: Transformations
- Complete notebook section: ADF/KPSS tests
- Complete notebook section: Spurious regression demo
- Plot series in levels before running tests.
- Justify each transformation (diff/logdiff) in words.
- State what your IRF identification assumes (ordering or structure).

### Alternative Example (Not the Notebook Solution)
```python
# Random walk vs stationary series (ADF intuition):
import numpy as np
from statsmodels.tsa.stattools import adfuller

rng = np.random.default_rng(0)
rw = rng.normal(size=400).cumsum()
st = rng.normal(size=400)

adf_rw_p = adfuller(rw)[1]
adf_st_p = adfuller(st)[1]
adf_rw_p, adf_st_p
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Stationarity and Unit Roots (ADF/KPSS)

> **Definition:** A time series is **stationary** if its statistical properties (mean/variance/autocovariance) are stable over time.

Many macro series in levels are not stationary (they trend).

#### Unit root intuition
A unit root process behaves like a random walk:
- shocks accumulate,
- the series does not “mean revert” in levels.

#### Why this matters: spurious regression
Regressing one trending series on another can produce:
- high $R^2$,
- significant t-stats,
even when there is no meaningful relationship.

#### Common tools
- **ADF test**: null = unit root (nonstationary)
- **KPSS test**: null = stationary

Practical habit:
- Plot the series.
- Try differences / growth rates.
- Use tests as supporting evidence, not as the only decision.

### Project Code Map
- `data/sample/panel_monthly_sample.csv`: offline macro panel
- `src/features.py`: safe lag/diff/rolling feature helpers
- `src/macro.py`: GDP growth + label helpers (for context)
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Running levels-on-levels regressions without checking stationarity (spurious regression).
- Interpreting Granger causality as structural causality.
- Choosing VAR lags mechanically without sanity checks.
- For IRFs: forgetting that orthogonalized IRFs depend on variable ordering.

<a id="summary"></a>
## Summary + Suggested Readings

You now have a classical macro time-series toolkit that complements the ML workflow in this repo.
Use it to avoid spurious inference and to reason about dynamics.


Suggested readings:
- Hamilton: Time Series Analysis (classic reference)
- Hyndman & Athanasopoulos: Forecasting: Principles and Practice (applied)
- Stock & Watson: Introduction to Econometrics (time-series chapters)
