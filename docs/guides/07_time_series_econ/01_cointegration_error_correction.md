# Guide: 01_cointegration_error_correction

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/07_time_series_econ/01_cointegration_error_correction.ipynb`.

> **Note:** Cointegration and error correction models are primarily used in **macroeconomics and finance**. For health economics, panel data methods (fixed effects, DiD, IV) are far more central. This guide provides a solid overview for general econometric literacy, but it is not a priority topic for health econ coursework.

**Prerequisite:** [Stationarity guide](00_stationarity_unit_roots.md) — you should understand I(0)/I(1) classification and the ADF test.

### Key Terms (defined)
- **Cointegration**: two or more I(1) series are cointegrated if a linear combination of them is I(0). They drift together — deviations from their long-run relationship are temporary.
- **Equilibrium error**: $u_t = y_t - \beta x_t$ — the deviation from the long-run relationship.
- **Error correction model (ECM)**: models short-run changes as a function of the lagged equilibrium error plus short-run changes in other variables.
- **Speed of adjustment ($\alpha$)**: the ECM coefficient on the lagged error. Negative $\alpha$ means the system corrects toward equilibrium.
- **Engle–Granger procedure**: (1) estimate the long-run relationship by OLS in levels, (2) test whether the residuals are stationary.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Confirm both series are I(1) using ADF + KPSS.
- Estimate the cointegrating regression: $y_t = a + \beta x_t + u_t$.
- Plot the residual $\hat{u}_t$ — does it look stationary?
- Test the residual using `statsmodels.tsa.stattools.coint()` (uses correct critical values).
- Estimate the ECM: $\Delta y_t = c + \alpha \hat{u}_{t-1} + \gamma \Delta x_t + \varepsilon_t$.
- Interpret $\hat\alpha$ (speed of adjustment) and $\hat\gamma$ (short-run effect).

### Alternative Example (Not the Notebook Solution)
```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

rng = np.random.default_rng(99)
T = 300

# Common stochastic trend (shared random walk)
trend = rng.normal(size=T).cumsum()
income = 50 + trend + rng.normal(scale=0.5, size=T)
consumption = 10 + 0.8 * trend + rng.normal(scale=1.5, size=T)

# Cointegration test
t_stat, p_val, crit = coint(consumption, income)
print(f"Cointegration test: t-stat={t_stat:.3f}, p-value={p_val:.4f}")

# ECM
df = pd.DataFrame({'c': consumption, 'y': income})
df['error_lag'] = (df['c'] - 0.8 * df['y']).shift(1)
df['dc'] = df['c'].diff()
df['dy'] = df['y'].diff()
df = df.dropna()
ecm = sm.OLS(df['dc'], sm.add_constant(df[['error_lag', 'dy']])).fit()
print(ecm.summary())
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### The core idea

The [stationarity guide](00_stationarity_unit_roots.md) established that regressing one I(1) series on another usually produces spurious results. Cointegration is the exception: when two I(1) series share a common stochastic trend, their linear combination can be stationary.

**Economic examples:**
- Consumption and income (the savings rate is roughly stable over time)
- Short-term and long-term interest rates (the yield spread is bounded)
- Wages and productivity (competitive pressures keep them linked)

The key insight: **differencing destroys the long-run information.** If the series are cointegrated, you want to model both the levels relationship (equilibrium) and the changes (short-run dynamics).

### Formal definition

Let $y_t$ and $x_t$ each be I(1). They are **cointegrated** if there exists $\beta$ such that:

$$
u_t = y_t - \beta x_t \quad \text{is I(0)}.
$$

- $\beta$: the long-run relationship (e.g., long-run marginal propensity to consume).
- $u_t$: equilibrium error — mean-reverting, does not trend.

### The Engle–Granger procedure

**Step 1:** Estimate $y_t = a + \beta x_t + u_t$ by OLS.

**Step 2:** Test whether $\hat{u}_t$ is stationary using ADF — but with **cointegration-specific critical values** (more negative than standard ADF). Use `coint()` from statsmodels, which handles this automatically.

**Why nonstandard critical values?** You are testing residuals from an estimated regression, not a raw series. The estimation step makes it easier to "find" stationarity by chance, so the critical values must be stricter to compensate.

**Super-consistency:** When cointegration holds, $\hat\beta$ converges to the true $\beta$ at rate $T$ (not $\sqrt{T}$). The coefficient is very precise. However, OLS standard errors from the cointegrating regression are **not valid for inference** — use the Johansen procedure or DOLS/FMOLS for proper confidence intervals on $\beta$.

### The Error Correction Model (ECM)

If cointegration holds, the Granger representation theorem guarantees an ECM exists:

$$
\Delta y_t = c + \alpha (y_{t-1} - \beta x_{t-1}) + \gamma \Delta x_t + \varepsilon_t.
$$

| Term | Meaning |
|---|---|
| $\alpha (y_{t-1} - \beta x_{t-1})$ | Error correction: pulls $y$ back toward equilibrium |
| $\gamma \Delta x_t$ | Short-run effect of changes in $x$ |

**Interpreting $\alpha$:**
- $\alpha = -0.1$: corrects 10% of the gap each period (half-life $\approx$ 6.6 periods)
- $\alpha = -0.5$: fast correction (half-life $\approx$ 1.4 periods)
- $\alpha \approx 0$: no correction — casts doubt on cointegration
- $\alpha > 0$: diverges from equilibrium — misspecification

**Numerical example:** Long-run relationship: consumption = 10 + 0.8 × income. At $t-1$: income=100, consumption=95 (equilibrium = 90). Error = +5. With $\alpha = -0.15$: predicted $\Delta$consumption $\approx -0.15 \times 5 = -0.75$.

### Code template

```python
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

# Test for cointegration
t_stat, p_val, crit = coint(y_series, x_series, trend='c')

# Long-run regression
lr = sm.OLS(y_series, sm.add_constant(x_series)).fit()
residuals = lr.resid

# ECM
df['error_lag'] = residuals.shift(1)
df['dy'] = y_series.diff()
df['dx'] = x_series.diff()
df = df.dropna()
ecm = sm.OLS(df['dy'], sm.add_constant(df[['error_lag', 'dx']])).fit(cov_type='HC3')
```

### Diagnostics checklist

1. **Confirm both series are I(1)** — if either is I(0), use standard regression instead.
2. **Plot the equilibrium error** — should look mean-reverting.
3. **Check $\alpha$ sign** — must be negative for error correction.
4. **ECM residual autocorrelation** — add lagged $\Delta y$, $\Delta x$ if needed.
5. **Subsample stability** — does $\hat\beta$ change across periods?

### What this does NOT mean
- Cointegration is not causality.
- OLS t-stats from the cointegrating regression are not valid inference.
- Cointegration is a *long-run* concept — says nothing about short-run dynamics without the ECM.

#### Exercises

- [ ] Choose two trending macro series, confirm both are I(1), and test for cointegration.
- [ ] Estimate the ECM and interpret $\hat\alpha$ in words. Compute the half-life.
- [ ] Create two independent random walks and confirm the test fails to reject.

### Project Code Map
- `data/sample/panel_monthly_sample.csv`: offline macro panel
- `src/features.py`: safe lag/diff/rolling feature helpers
- `src/data.py`: caching helpers

### Common Mistakes
- Using standard ADF critical values for residual-based cointegration tests (too liberal).
- Interpreting OLS t-stats from the cointegrating regression as valid inference.
- Differencing cointegrated series — this discards the long-run equilibrium information.
- Ignoring the sign of $\hat\alpha$: positive means divergence, not correction.

<a id="summary"></a>
## Summary + Suggested Readings

Cointegration captures stable long-run relationships between nonstationary series. The ECM then models how the system adjusts back to equilibrium. The workflow: **confirm I(1) → test cointegration → estimate ECM → interpret $\alpha$.**

Suggested readings:
- Engle & Granger (1987): "Co-Integration and Error Correction"
- Murray (1994): "A Drunk and Her Dog" — an intuitive cointegration analogy
- Hamilton (1994): *Time Series Analysis* — Ch. 19
