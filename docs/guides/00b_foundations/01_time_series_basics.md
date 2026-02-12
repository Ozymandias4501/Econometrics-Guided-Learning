# Guide: 01_time_series_basics

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00b_foundations/01_time_series_basics.ipynb`.

Time-series data is the backbone of health economics research: quarterly national health expenditure accounts, monthly hospital admission counts, weekly flu surveillance reports, daily ICU census data. The defining feature is that **observations are ordered in time and neighboring observations are correlated**. This single fact breaks the i.i.d. assumption that underlies most introductory statistics and many ML defaults.

This guide builds the core time-series intuition you need before touching any forecasting or causal-inference notebook. If you skip these foundations, every model you build later is at risk of silent failure.

### Key Terms (defined)
- **Stationarity**: a process whose statistical properties (mean, variance, autocorrelation structure) do not change over time. Most regression and forecasting methods assume or require stationarity.
- **Autocorrelation**: the correlation of a series with its own lagged values. Positive autocorrelation means high values tend to follow high values.
- **ACF (autocorrelation function)**: the correlation between $y_t$ and $y_{t-k}$ for each lag $k$, including indirect effects through intermediate lags.
- **PACF (partial autocorrelation function)**: the correlation between $y_t$ and $y_{t-k}$ after removing the linear effect of lags $1, \dots, k-1$. Useful for identifying the "direct" influence of each lag.
- **Differencing**: computing $\Delta y_t = y_t - y_{t-1}$ to remove trends and achieve stationarity.
- **Trend**: a long-run increase or decrease in the level of the series.
- **Seasonality**: a regular, calendar-driven pattern that repeats at a fixed period (e.g., 12 months, 52 weeks).
- **Cyclical component**: fluctuations that are not of fixed period, often driven by business or health-system cycles.
- **White noise**: a series of uncorrelated random variables with constant mean and variance; the "no signal" baseline.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Toy series
- Complete notebook section: Resampling
- Complete notebook section: Lag and rolling features
- Complete notebook section: Leakage demo
- Run the bootstrap cell and confirm `PROJECT_ROOT` points to the repo root.
- Complete all TODOs (no `...` left).
- Plot the ACF and PACF of at least one series and write a 3-sentence interpretation.
- Difference a trending series and confirm the differenced version looks stationary.
- Write a short paragraph explaining how seasonal patterns in health data (e.g., flu admissions) would appear in an ACF plot.

### Alternative Example (Not the Notebook Solution)
```python
# Decomposing a synthetic health-spending series (not the notebook data):
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
quarters = pd.date_range("2010-01-01", periods=60, freq="QS")

# Components: upward trend + seasonal (Q1 spike from flu season) + noise
trend = np.linspace(100, 160, len(quarters))
seasonal = 8 * np.tile([3, -1, -2, 0], len(quarters) // 4)  # Q1 high
noise = rng.normal(scale=2, size=len(quarters))

health_spending = pd.Series(trend + seasonal + noise, index=quarters, name="spending_idx")

# Visual check
print("First 8 quarters:")
print(health_spending.head(8).round(1).to_string())

# Differencing to remove the trend
diff1 = health_spending.diff().dropna()
print(f"\nOriginal mean: {health_spending.mean():.1f}, Differenced mean: {diff1.mean():.2f}")
print(f"Original std:  {health_spending.std():.1f}, Differenced std:  {diff1.std():.2f}")

# ACF at lag 4 should be high (seasonal)
from statsmodels.tsa.stattools import acf
acf_vals = acf(diff1, nlags=8, fft=False)
print(f"\nACF of differenced series at lag 4: {acf_vals[4]:.3f}  (seasonal signal)")
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

This guide covers the core properties of time-series data that you must understand before building any forecasting or time-indexed regression model.

### Why time-series data violates i.i.d. assumptions

Most econometric and ML methods assume observations are drawn independently from the same distribution. Time-series data typically violates both parts:

1. **Not independent.** Today's hospital admissions are correlated with yesterday's. This quarter's health spending is correlated with last quarter's. Ignoring this correlation leads to understated standard errors, inflated t-statistics, and false "significance."

2. **Not identically distributed.** Health spending grows over time (trend). Flu admissions peak every winter (seasonality). The COVID-19 pandemic introduced a structural break. If the distribution shifts, a model trained on old data may not generalize.

**Health econ example:** Suppose you regress monthly ER visits on staffing levels using OLS and find $p < 0.01$. If ER visits are serially correlated (they are), the effective sample size is smaller than the nominal $n$, and the true $p$-value could be 0.10 or higher. This is not a minor nuance -- it can reverse your conclusion.

### Deep Dive: Stationarity -- the assumption that makes time-series analysis work

#### 1) Intuition (plain English)

A stationary series "looks the same" statistically no matter when you observe it. The mean does not trend upward. The variance does not grow over time. The correlation between observations $k$ periods apart is always the same.

Why does this matter? Because most time-series models (ARIMA, VAR, Granger causality tests) assume stationarity. If you fit these models to non-stationary data, the results can be spurious -- you get "significant" relationships between unrelated trending variables.

**Story example:** U.S. national health expenditure and the number of smartphone users both trend upward from 2010-2020. A regression of one on the other gives $R^2 > 0.95$ and a tiny $p$-value. This is a **spurious regression** -- two unrelated trends that happen to move together. Differencing both series would eliminate the trend and reveal that the relationship disappears.

#### 2) Formal definition

A time series $\{y_t\}$ is **(weakly/covariance) stationary** if:

1. $\mathbb{E}[y_t] = \mu$ for all $t$ (constant mean),
2. $\mathrm{Var}(y_t) = \sigma^2$ for all $t$ (constant variance),
3. $\mathrm{Cov}(y_t, y_{t-k}) = \gamma(k)$ depends only on the lag $k$, not on $t$ (autocovariance depends only on displacement).

A **unit root** process like a random walk $y_t = y_{t-1} + \varepsilon_t$ is not stationary because its variance grows with $t$:

$$
\mathrm{Var}(y_t) = t \cdot \sigma_\varepsilon^2.
$$

#### 3) Testing for stationarity

The **Augmented Dickey-Fuller (ADF) test** is the standard diagnostic:

- $H_0$: the series has a unit root (non-stationary).
- $H_1$: the series is stationary.

A small $p$-value (e.g., $< 0.05$) rejects the unit root and supports stationarity.

```python
from statsmodels.tsa.stattools import adfuller

# Example: test whether quarterly health spending is stationary
result = adfuller(health_spending, autolag="AIC")
print(f"ADF statistic: {result[0]:.3f}")
print(f"p-value:       {result[1]:.4f}")
print(f"Lags used:     {result[2]}")

# If p > 0.05, try differencing
result_diff = adfuller(health_spending.diff().dropna(), autolag="AIC")
print(f"\nAfter differencing:")
print(f"ADF statistic: {result_diff[0]:.3f}")
print(f"p-value:       {result_diff[1]:.4f}")
```

**Interpretation rule of thumb:** If the ADF test fails to reject (high $p$-value), difference the series and re-test. Most macroeconomic and health-spending series are I(1) -- stationary after first differencing.

#### 4) When stationarity matters vs. when it does not

| Situation | Stationarity needed? | What to do |
|---|---|---|
| ARIMA forecasting | Yes (after differencing) | Let the "I" in ARIMA handle it |
| OLS regression with time-series data | Yes, or use cointegration | Difference, or test for cointegration |
| ML prediction (random forest, XGBoost) | Not formally, but helps | Detrend/difference features to reduce distribution shift |
| Granger causality testing | Yes | Difference to stationarity first |
| Difference-in-differences (causal) | Assumes parallel trends, not stationarity per se | Check pre-trends |

### Deep Dive: Autocorrelation -- ACF and PACF interpretation

#### 1) Intuition (plain English)

Autocorrelation measures how much a series is correlated with its own past. If you know that hospital admissions were high last week, autocorrelation tells you how much that helps predict this week.

The **ACF** at lag $k$ includes both direct and indirect effects. The **PACF** at lag $k$ isolates the direct effect of lag $k$ after controlling for lags $1, \dots, k-1$.

#### 2) Notation

The autocorrelation at lag $k$:

$$
\rho(k) = \frac{\gamma(k)}{\gamma(0)} = \frac{\mathrm{Cov}(y_t, y_{t-k})}{\mathrm{Var}(y_t)}.
$$

The partial autocorrelation $\phi_{kk}$ is the coefficient on $y_{t-k}$ in the regression:

$$
y_t = \phi_{k1} y_{t-1} + \phi_{k2} y_{t-2} + \cdots + \phi_{kk} y_{t-k} + \varepsilon_t.
$$

#### 3) Reading ACF/PACF plots -- the practical rules

| ACF pattern | PACF pattern | Suggests |
|---|---|---|
| Decays slowly (many significant lags) | Cuts off after lag $p$ | AR($p$) process |
| Cuts off after lag $q$ | Decays slowly | MA($q$) process |
| Decays slowly | Decays slowly | ARMA or differencing needed |
| Significant spike at lag $s, 2s, 3s, \dots$ | Significant spike at lag $s$ | Seasonal pattern with period $s$ |

**Health econ example:** Monthly flu hospitalization data will typically show:
- ACF: significant positive autocorrelation decaying slowly, with spikes at lags 12, 24, 36 (annual seasonality).
- PACF: significant spikes at lags 1-2 and at lag 12, then cuts off. This suggests an AR structure with a seasonal AR component.

#### 4) Worked example

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# Simulate an AR(2) process (e.g., hospital admissions with momentum)
rng = np.random.default_rng(0)
n = 300
y = np.zeros(n)
for t in range(2, n):
    y[t] = 0.6 * y[t - 1] + 0.2 * y[t - 2] + rng.normal()

# Compute ACF and PACF
acf_vals = acf(y, nlags=20, fft=False)
pacf_vals = pacf(y, nlags=20, method="ywm")

print("ACF  lags 1-5:", [f"{v:.3f}" for v in acf_vals[1:6]])
print("PACF lags 1-5:", [f"{v:.3f}" for v in pacf_vals[1:6]])
# Expected: PACF significant at lags 1 and 2, then cuts off
# ACF decays gradually
```

#### 5) Why autocorrelation breaks standard OLS inference

If you run OLS on time-series data and the residuals are autocorrelated, the standard errors are **wrong**. Specifically:

- Positive autocorrelation in residuals $\Rightarrow$ OLS underestimates SEs $\Rightarrow$ $t$-statistics too large $\Rightarrow$ false rejections.
- The Durbin-Watson statistic is a quick diagnostic: values near 2 suggest no first-order autocorrelation; values near 0 suggest strong positive autocorrelation.

**Fix:** Use Newey-West (HAC) standard errors, which account for serial correlation:

```python
import statsmodels.api as sm

# Suppose you regress health_outcome on policy_variable with time-series data
# model = sm.OLS(y, X).fit()                       # naive SEs (wrong if autocorrelated)
# model_hac = sm.OLS(y, X).fit(cov_type="HAC",     # Newey-West SEs
#                               cov_kwds={"maxlags": 4})
```

### Deep Dive: Trend, seasonality, and cyclical patterns

#### 1) Decomposition

Any time series can be decomposed (additively or multiplicatively) into:

$$
y_t = T_t + S_t + C_t + \varepsilon_t
$$

where:
- $T_t$ = trend (long-run direction),
- $S_t$ = seasonal component (fixed calendar period),
- $C_t$ = cyclical component (variable-length fluctuations),
- $\varepsilon_t$ = irregular/noise component.

**Health econ examples:**
- **Trend:** U.S. health spending per capita has grown roughly 4-5% per year for decades.
- **Seasonality:** Hospital admissions spike in Q1 (flu season) and dip in summer. Mental health ER visits often spike around holidays.
- **Cyclical:** Recessions reduce elective procedures and employer-sponsored insurance enrollment; these are cyclical but not seasonal (no fixed period).

#### 2) Practical decomposition in Python

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Requires a series with a frequency set (e.g., "MS" for month-start)
# decomposition = seasonal_decompose(monthly_admissions, model="additive", period=12)
# decomposition.plot()

# For health spending (quarterly):
# decomposition = seasonal_decompose(quarterly_spending, model="additive", period=4)
```

#### 3) Differencing: removing trends and seasonality

**First differencing** removes a linear trend:

$$
\Delta y_t = y_t - y_{t-1}.
$$

**Seasonal differencing** removes seasonal patterns:

$$
\Delta_s y_t = y_t - y_{t-s}
$$

where $s$ is the seasonal period (e.g., $s=12$ for monthly data with annual seasonality, $s=4$ for quarterly data).

You can combine both:

$$
\Delta \Delta_s y_t = \Delta y_t - \Delta y_{t-s}.
$$

**Rule of thumb:** Difference until the ADF test rejects the unit root. Most health-econ series need $d=1$ (one regular difference). Some also need one seasonal difference.

#### 4) Worked example: differencing quarterly health spending

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

rng = np.random.default_rng(7)
quarters = pd.date_range("2005-01-01", periods=80, freq="QS")

# Simulated quarterly health spending: trend + seasonality + noise
trend = np.linspace(200, 340, 80)
seasonal = 6 * np.tile([2.5, -0.5, -1.5, -0.5], 20)
spending = pd.Series(trend + seasonal + rng.normal(scale=3, size=80),
                     index=quarters, name="spending")

# Level: non-stationary
print("Level ADF p-value:", adfuller(spending, autolag="AIC")[1].round(4))

# First difference: removes trend but seasonal pattern remains
diff1 = spending.diff().dropna()
print("Diff1 ADF p-value:", adfuller(diff1, autolag="AIC")[1].round(4))

# Seasonal difference (lag 4): removes both trend and seasonality
diff_seasonal = spending.diff(4).dropna()
print("Seasonal diff ADF p-value:", adfuller(diff_seasonal, autolag="AIC")[1].round(4))
```

### Deep Dive: Leakage in time-series contexts

> For the general treatment of leakage and time-aware evaluation, see the [setup guide](00_setup.md).

Here we focus on leakage patterns specific to time-series feature engineering.

#### Time-series-specific leakage traps

1. **Wrong shift direction.** `.shift(-1)` looks into the future; `.shift(1)` looks into the past. The sign convention is easy to confuse.

2. **Centered rolling windows.** `rolling(7, center=True)` uses 3 future observations. For forecasting, always use `center=False`.

3. **Resampling before splitting.** If you resample (e.g., daily to monthly) using the full dataset, month boundaries near the train/test split can leak.

4. **Seasonal adjustment on full sample.** Running X-13 or STL decomposition on the entire dataset before train/test split lets future seasonal factors influence the training set.

5. **Publication lags in health data.** CMS releases National Health Expenditure data with a 2-3 year lag. If you align it as if it were available in real time, your backtest is unrealistic.

### Common Mistakes (time-series-specific)
- Regressing two trending series on each other and interpreting a high $R^2$ as evidence of a relationship (spurious regression).
- Using `train_test_split(shuffle=True)` on time-ordered data -- this destroys the temporal structure.
- Ignoring autocorrelation in OLS residuals, leading to artificially small standard errors.
- Differencing more than necessary ($d=2$ when $d=1$ suffices), which over-differences and introduces unnecessary noise.
- Confusing **seasonal** patterns (fixed calendar period) with **cyclical** patterns (variable length) -- they require different modeling strategies.
- Fitting a seasonal ARIMA without first checking whether the seasonal period is correct (e.g., using $s=12$ on quarterly data).
- Applying the ADF test to a series with a structural break (the test has low power in this case; consider a Zivot-Andrews test instead).

### Project Code Map
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)

#### Exercises

- [ ] Take a health-related time series (e.g., monthly flu hospitalizations). Plot the ACF and PACF. Write 5 sentences describing what you see.
- [ ] Test the series for stationarity using the ADF test. If non-stationary, difference and re-test.
- [ ] Decompose the series into trend, seasonal, and residual components using `seasonal_decompose`. Which component dominates?
- [ ] Create an intentional leakage feature (future shift or centered rolling window) and show how it inflates test-set $R^2$ compared to a legitimate lagged feature.
- [ ] Run an OLS regression on time-series data and compare naive SEs with Newey-West (HAC) SEs. How much do the $t$-statistics change?
- [ ] Find an example of two trending health-econ series. Regress one on the other in levels (spurious regression), then in first differences. Compare results.

<a id="summary"></a>
## Summary + Suggested Readings

This guide covered the foundational time-series concepts you will use in every forecasting and time-indexed regression notebook:

1. **Stationarity** is the gateway assumption. Test for it (ADF), and difference if needed.
2. **Autocorrelation** (ACF/PACF) tells you the memory structure of the series and guides model selection.
3. **Trend, seasonality, and cycles** are the three components you must identify and handle before modeling.
4. **Differencing** is the primary tool for achieving stationarity; seasonal differencing handles periodic patterns.
5. **Time-series data violates i.i.d.**, which means standard OLS inference (without HAC corrections) and random train/test splits are unreliable.

The key insight for a health economist: health data is almost always time-indexed, seasonal, and autocorrelated. Ignoring these features does not just reduce accuracy -- it produces misleading inference.

### Suggested Readings
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (free online) -- Chapters 2-3 on time-series graphics and decomposition, Chapter 8 on ARIMA.
- Hamilton, *Time Series Analysis* -- the graduate-level reference for stationarity and unit roots.
- Wooldridge, *Introductory Econometrics* -- Chapter 10 on time-series basics, Chapter 12 on serial correlation.
- Stock & Watson, *Introduction to Econometrics* -- accessible treatment of time-series regression and HAC standard errors.
- CDC FluView (https://www.cdc.gov/flu/weekly/) -- real seasonal health data for practice.
