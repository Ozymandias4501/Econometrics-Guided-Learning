# Guide: 01_cointegration_error_correction

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/08_time_series_econ/01_cointegration_error_correction.ipynb`.

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
- Complete notebook section: Construct cointegrated pair
- Complete notebook section: Engle-Granger test
- Complete notebook section: Error correction model
- Complete notebook section: Interpretation
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

### Cointegration and Error Correction Models (ECM)

Two series can be individually nonstationary but move together in the long run.

> **Definition:** $x_t$ and $y_t$ are **cointegrated** if some linear combination is stationary:
$$
y_t - \\beta x_t \\text{ is stationary}
$$

#### Engle–Granger (two-step) idea
1) Regress $y_t$ on $x_t$ in levels to estimate $\\hat\\beta$.
2) Test whether the residual $\\hat u_t = y_t - \\hat\\beta x_t$ is stationary.

#### Error correction model
An ECM links short-run changes to long-run deviations:

$$
\\Delta y_t = \\alpha( y_{t-1} - \\beta x_{t-1}) + \\Gamma \\Delta x_t + \\varepsilon_t
$$

Interpretation:
- $(y_{t-1} - \\beta x_{t-1})$ is the “error” from the long-run relationship.
- $\\alpha$ is the speed of adjustment back to equilibrium.

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
