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


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

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

### Stationarity and Unit Roots: why levels-on-levels can lie

Classical time-series econometrics starts with one question:

> **Is this series stable enough over time that our regression assumptions make sense?**

#### 1) Intuition (plain English)

Many macro series in **levels** trend upward over decades (GDP, price level, money supply).
If two series both trend, they can look strongly related even when one does not cause the other.

**Story example:** GDP and total credit both trend up.
A regression “GDP on credit” in levels can produce a high $R^2$ even if the relationship is spurious.

Stationarity is the formal way to ask:
- “Do shocks die out (mean reversion)?” or
- “Do shocks accumulate forever (random walk)?”

#### 2) Notation + setup (define symbols)

Let $\\{y_t\\}_{t=1}^T$ be a time series.

Key population objects:
- mean: $\\mu = \\mathbb{E}[y_t]$
- variance: $\\gamma_0 = \\mathrm{Var}(y_t)$
- autocovariance at lag $k$:
$$
\\gamma_k = \\mathrm{Cov}(y_t, y_{t-k}).
$$

**Weak (covariance) stationarity** means:
1) $\\mathbb{E}[y_t]$ is constant over time,
2) $\\mathrm{Var}(y_t)$ is constant over time,
3) $\\mathrm{Cov}(y_t, y_{t-k})$ depends only on $k$, not on $t$.

**Strict stationarity** is stronger (the full joint distribution is time-invariant). In practice, weak stationarity is the common working definition for linear models.

#### 3) I(0) vs I(1) (integrated processes)

Econometrics often classifies series by how many differences are needed to make them “stationary-ish”:

- $y_t$ is **I(0)** if it is stationary (in levels).
- $y_t$ is **I(1)** if $\\Delta y_t = y_t - y_{t-1}$ is I(0).

Typical examples:
- growth rates, inflation rates, spreads: often closer to I(0),
- price levels, GDP levels: often closer to I(1).

#### 4) Unit roots via AR(1) intuition

Consider an AR(1):

$$
y_t = \\rho y_{t-1} + \\varepsilon_t,
\\qquad
\\varepsilon_t \\sim (0, \\sigma^2).
$$

**What each term means**
- $\\rho$: persistence parameter.
- $\\varepsilon_t$: innovation (new shock at time $t$).

Cases:
- If $|\\rho| < 1$, the process is stationary and shocks decay (mean reversion).
- If $\\rho = 1$, you have a **unit root** (random walk-like behavior).

Random walk:
$$
y_t = y_{t-1} + \\varepsilon_t
\\quad \\Rightarrow \\quad
y_t = y_0 + \\sum_{s=1}^{t} \\varepsilon_s.
$$

**Key implication**
- the variance of $y_t$ grows with $t$ (shocks accumulate),
- the series does not settle around a fixed mean in levels.

Differencing removes the unit root:
$$
\\Delta y_t = y_t - y_{t-1} = \\varepsilon_t,
$$
which is stationary if innovations are stable.

#### 5) Why this matters: spurious regression

If $x_t$ and $y_t$ are both I(1), then a regression like:

$$
y_t = \\alpha + \\beta x_t + u_t
$$

can show:
- large t-stats,
- high $R^2$,
even if $x_t$ and $y_t$ are unrelated in any causal or structural sense.

The reason (intuition):
- both series share trending behavior,
- residuals can be highly persistent,
- classic OLS inference assumptions break.

**Practical rule:** do not treat a levels-on-levels regression as meaningful until you have checked stationarity / cointegration logic.

#### 6) ADF test: what it is actually doing

The Augmented Dickey–Fuller test fits a regression of changes on lagged levels:

$$
\\Delta y_t = a + bt + \\gamma y_{t-1} + \\sum_{j=1}^{p} \\phi_j \\Delta y_{t-j} + e_t.
$$

**What each term means**
- $\\Delta y_t$: change in the series.
- $a$: intercept (allows non-zero mean).
- $bt$: trend term (optional).
- $y_{t-1}$: lagged level (detects unit root).
- lagged differences: soak up serial correlation in $e_t$.

Null vs alternative (common interpretation):
- **Null:** unit root (nonstationary) → roughly $\\gamma = 0$ (equivalently $\\rho = 1$).
- **Alternative:** stationary (mean-reverting) → $\\gamma < 0$ (equivalently $|\\rho|<1$).

Important: ADF has low power in small samples; “fail to reject” does not mean “definitely a unit root.”

#### 7) KPSS test: complementary null

KPSS flips the null:
- **Null:** stationary,
- **Alternative:** unit root / nonstationary.

That’s why people often run ADF and KPSS together:
- ADF rejects + KPSS fails to reject → evidence for stationarity.
- ADF fails to reject + KPSS rejects → evidence for nonstationarity.
- Conflicts happen often → treat tests as diagnostics, not commandments.

#### 8) Mapping to code (statsmodels)

In Python:
- `statsmodels.tsa.stattools.adfuller(x)` returns a test statistic and a p-value.
- `statsmodels.tsa.stattools.kpss(x, regression='c' or 'ct')` does the KPSS test.

Practical habits:
- drop missing values before testing,
- test both levels and differences,
- specify whether you include a trend term (economic series often trend).

#### 9) Diagnostics + robustness (minimum set)

1) **Plot the level series**
- Do you see a clear trend or structural break?

2) **Plot the differenced / growth-rate series**
- Does it look more stable? Mean-reverting?

3) **ADF + KPSS on both levels and differences**
- Do results agree? If not, explain why (trend term, breaks, sample size).

4) **ACF/PACF or residual autocorrelation**
- Persistent residuals suggest misspecification.

#### 10) Interpretation + reporting

When you report stationarity checks:
- state whether you tested levels and differences,
- state whether you included a constant/trend,
- show at least one plot alongside test results.

**What this does NOT mean**
- A small p-value is not a proof of stationarity in “the real world.”
- Structural breaks can fool unit-root tests (you can reject/accept for the wrong reason).

#### Exercises

- [ ] Pick one macro series and classify it as “likely I(0)” or “likely I(1)” with a plot-based argument.
- [ ] Run ADF and KPSS on the level series and on its first difference; interpret the pair.
- [ ] Demonstrate spurious regression by regressing one random walk on another and reporting $R^2$.
- [ ] Choose a transformation (difference, log-difference, growth rate) and justify it in 4 sentences.

### Cointegration + ECM: long-run equilibrium with short-run dynamics

Cointegration is the key “exception” to the rule that nonstationary levels regressions are spurious.

#### 1) Intuition (plain English)

Two series can trend over time but still be meaningfully linked:
- consumption and income,
- prices and money aggregates,
- wages and productivity (in some settings).

The idea:
- each series drifts,
- but some combination of them is stable in the long run.

Cointegration formalizes “move together over decades.”

#### 2) Notation + setup (define symbols)

Let $y_t$ and $x_t$ be time series.
Assume each is I(1) (nonstationary in levels, stationary in first differences).

They are **cointegrated** if there exists a parameter $\\beta$ such that:

$$
u_t = y_t - \\beta x_t
\\quad \\text{is I(0) (stationary).}
$$

**What each term means**
- $\\beta$: long-run relationship (“equilibrium” slope).
- $u_t$: deviation from the long-run equilibrium; should mean-revert if cointegration holds.

#### 3) Why cointegration matters

If $y_t$ and $x_t$ are cointegrated:
- a regression in levels can capture a real long-run relationship,
- but you must model short-run dynamics carefully.

If they are not cointegrated:
- levels regression can be spurious,
- differencing is usually safer.

#### 4) Engle–Granger two-step procedure (mechanics)

Step 1: estimate the long-run relationship in levels:

$$
y_t = a + \\beta x_t + e_t.
$$

Step 2: test whether the residuals are stationary:
$$
\\hat u_t = y_t - \\hat a - \\hat\\beta x_t.
$$

If $\\hat u_t$ is stationary, that supports cointegration.

Important practical note:
- residual-based cointegration tests use nonstandard critical values (packages like `statsmodels` handle this for you).

#### 5) Error Correction Model (ECM): connecting short-run changes to long-run gaps

If $y_t$ and $x_t$ are cointegrated, an ECM often makes sense:

$$
\\Delta y_t = c + \\alpha \\,(y_{t-1} - \\beta x_{t-1}) + \\Gamma \\Delta x_t + \\varepsilon_t.
$$

**What each term means**
- $\\Delta y_t$: short-run change in $y$.
- $(y_{t-1} - \\beta x_{t-1})$: last period’s equilibrium error (how far you were from long-run relationship).
- $\\alpha$: “speed of adjustment” (typically negative if deviations are corrected).
- $\\Gamma \\Delta x_t$: short-run effect of changes in $x$.

Interpretation of $\\alpha$:
- If $y$ is above its long-run equilibrium relative to $x$ (positive error),
  a negative $\\alpha$ pulls $\\Delta y_t$ downward to correct.

#### 6) Assumptions and practical caveats

Cointegration/ECM is most appropriate when:
- both series are integrated of the same order (often I(1)),
- the long-run relationship is stable over the sample,
- there are no major structural breaks (breaks can change $\\beta$).

If there are breaks (policy regime shifts, measurement changes), cointegration results can be misleading.

#### 7) Mapping to code (statsmodels)

Useful tools:
- `statsmodels.tsa.stattools.coint(y, x)` runs an Engle–Granger cointegration test.
- Build the ECM manually with `statsmodels.api.OLS`:
  - compute the lagged error term $(y_{t-1} - \\hat\\beta x_{t-1})$,
  - regress $\\Delta y_t$ on that error and $\\Delta x_t$ (and possibly lags).

#### 8) Diagnostics + robustness (minimum set)

1) **Confirm integration order**
- Test whether $x_t$ and $y_t$ are I(1) (ADF/KPSS on levels and differences).

2) **Residual stationarity**
- After Step 1, test whether $\\hat u_t$ is stationary; plot it too.

3) **ECM residual checks**
- Check autocorrelation of ECM residuals; add lags if needed.

4) **Stability**
- Re-estimate cointegration relationship on subperiods; do parameters drift?

#### 9) Interpretation + reporting

When reporting cointegration/ECM:
- clearly separate long-run relationship ($\\beta$) from short-run dynamics ($\\Gamma$),
- interpret $\\alpha$ as speed of adjustment (sign matters!),
- include plots of levels, residual (error-correction term), and differenced series.

**What this does NOT mean**
- Cointegration does not prove causality; it describes a stable long-run relationship.
- A significant $\\beta$ in levels is not meaningful if residuals are nonstationary.

#### Exercises

- [ ] Pick two trending macro series and test whether each is I(1).
- [ ] Run a cointegration test; interpret the result and plot the residual $\\hat u_t$.
- [ ] Fit an ECM and interpret $\\alpha$ in words (speed of adjustment).
- [ ] Try adding one lag of $\\Delta y_t$ and $\\Delta x_t$ and see whether residual autocorrelation improves.

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
