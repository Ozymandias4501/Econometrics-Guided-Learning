# Guide: 02_var_impulse_responses

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/08_time_series_econ/02_var_impulse_responses.ipynb`.

> **Note:** VARs and impulse response functions are primarily used in **macroeconomics**. For health economics, panel methods (FE, DiD, IV) are far more relevant. This guide provides a solid overview for econometric literacy — VARs may appear in your macro electives but are unlikely to be central to your research.

**Prerequisites:** [Stationarity guide](00_stationarity_unit_roots.md) and [cointegration guide](01_cointegration_error_correction.md).

### Key Terms (defined)
- **VAR($p$)**: a system of $k$ equations where each variable is regressed on $p$ lags of all $k$ variables.
- **Impulse response function (IRF)**: the time path of all variables after a one-time shock to one variable.
- **Orthogonalized IRF**: IRF computed after transforming correlated innovations into uncorrelated shocks via Cholesky decomposition. Results depend on variable ordering.
- **FEVD**: forecast error variance decomposition — what fraction of forecast uncertainty comes from each shock.
- **Granger causality**: does lagged $x$ help predict $y$ beyond $y$'s own lags? A forecasting test, not a causal one.
- **Stability**: all companion matrix eigenvalues inside the unit circle — shocks die out.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Select 3–4 macro variables, confirm stationarity (transform if needed).
- Fit a VAR; use AIC/BIC for lag selection, then check residual autocorrelation.
- Verify stability (all eigenvalues inside unit circle).
- Run Granger causality tests — interpret as "predictive content," not causation.
- Plot orthogonalized IRFs with confidence bands under two variable orderings.
- Compute FEVDs at a meaningful horizon.
- Write a paragraph interpreting IRFs and stating identification assumptions.

### Alternative Example (Not the Notebook Solution)
```python
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

rng = np.random.default_rng(42)
T = 200

# Simulated stationary data with cross-variable dynamics
data = pd.DataFrame({
    'unrate_diff': rng.normal(0, 0.3, T),
    'fedfunds_diff': rng.normal(0, 0.5, T),
    'indpro_growth': rng.normal(0, 0.8, T),
})
for t in range(2, T):
    data.loc[t, 'unrate_diff'] += 0.3 * data.loc[t-1, 'unrate_diff'] - 0.1 * data.loc[t-1, 'fedfunds_diff']
    data.loc[t, 'indpro_growth'] += 0.2 * data.loc[t-1, 'indpro_growth'] + 0.15 * data.loc[t-1, 'fedfunds_diff']

model = VAR(data)
results = model.fit(maxlags=4, ic='aic')
print("Stable?", results.is_stable())

irf = results.irf(periods=20)
irf.plot(orth=True)
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Why VAR?

Macro variables form interconnected systems with feedback loops. A VAR models all variables simultaneously, letting each depend on lags of every other variable. This avoids the single-equation problem of ignoring feedback (e.g., interest rates affect output, but output also affects interest rate policy).

### The VAR($p$) model

Let $y_t$ be a $k \times 1$ vector of stationary variables:

$$
y_t = c + A_1 y_{t-1} + \cdots + A_p y_{t-p} + \varepsilon_t,
$$

where $A_j$ are $k \times k$ coefficient matrices and $\varepsilon_t$ has covariance $\Sigma$.

**Parameter count:** $k^2 p + k$. With $k=4$, $p=4$: 68 parameters. Keep $k$ and $p$ small for short macro samples.

**Estimation:** OLS equation-by-equation (each equation has the same regressors).

### Lag selection

Use AIC/BIC as a starting point, then check residual autocorrelation:

$$
\text{AIC}(p) = \log|\hat\Sigma(p)| + \frac{2k^2 p}{T},
\qquad
\text{BIC}(p) = \log|\hat\Sigma(p)| + \frac{k^2 p \log T}{T}.
$$

If AIC and BIC disagree, try both and check residual diagnostics. Always verify stability.

### Granger causality

$x$ "Granger-causes" $y$ if lags of $x$ help predict $y$ beyond $y$'s own lags. This is tested with an F-test on the relevant lag coefficients.

**What it IS:** a test of predictive content.
**What it is NOT:** structural causality. A policy rate may "Granger-cause" output because the rate reacts to forward-looking information, not because rate changes directly cause output movements.

### IRFs: the moving-average representation

If the VAR is stable, it has an MA form:

$$
y_t = \mu + \sum_{s=0}^{\infty} \Phi_s \varepsilon_{t-s}.
$$

$\Phi_s[m,n]$ = effect on variable $m$ at horizon $s$ from a unit shock to variable $n$.

**Problem:** Innovations $\varepsilon_t$ are correlated across equations. A "shock to variable 1" also involves variable 2.

### Orthogonalized IRFs (Cholesky)

Decompose $\Sigma = P P'$ (Cholesky, lower-triangular). Define uncorrelated shocks $u_t = P^{-1}\varepsilon_t$. Then:

$$
\Theta_s = \Phi_s P.
$$

**The Cholesky ordering matters.** It imposes a recursive structure:
- Variable 1 (first) is affected only by its own shock contemporaneously.
- Variable 2 is affected by shocks to variables 1 and 2 contemporaneously.
- And so on.

Swapping the order can change IRF signs and magnitudes. **Always try at least two orderings** and check robustness.

### FEVD

The forecast error variance decomposition answers: "What fraction of forecast uncertainty at horizon $h$ comes from each shock?"

$$
\text{FEVD}_{m \leftarrow n}(h) = \frac{\sum_{s=0}^{h-1} \Theta_s[m,n]^2}{\sum_{s=0}^{h-1} \sum_{j=1}^{k} \Theta_s[m,j]^2}.
$$

At short horizons, each variable's own shock dominates. At longer horizons, cross-variable shocks become more important.

### Code template

```python
from statsmodels.tsa.api import VAR
import numpy as np

data = df[['var1', 'var2', 'var3']].dropna()
model = VAR(data)

# Lag selection
print(model.select_order(maxlags=12).summary())

# Fit
results = model.fit(maxlags=4, ic='aic')
print("Stable?", results.is_stable())

# Granger causality
gc = results.test_causality('var1', ['var2'], kind='f')
print(gc.summary())

# IRFs with confidence bands
irf = results.irf(periods=24)
irf.plot(orth=True)

# FEVD
fevd = results.fevd(periods=24)
fevd.plot()
```

### Diagnostics checklist

1. **All variables must be stationary** — non-stationary variables produce unreliable IRFs.
2. **Stability** — all companion matrix eigenvalues must have modulus < 1.
3. **Residual autocorrelation** — if present, add lags.
4. **Ordering sensitivity** — try multiple orderings; if conclusions flip, identification is fragile.
5. **Confidence bands** — always report; a response not significantly different from zero is not reliable.

### What this does NOT mean
- VAR coefficients are forecasting relationships, not causal effects.
- Granger causality ≠ structural causality.
- Orthogonalized IRFs are conditional on the ordering assumption.
- An insignificant IRF may reflect insufficient data, not "no effect."

#### Exercises

- [ ] Fit a 3-variable VAR, report lag selection and stability.
- [ ] Run Granger causality tests; interpret one carefully as "predictive content."
- [ ] Plot IRFs under two orderings; identify one robust and one sensitive response.
- [ ] Compute FEVDs at horizons 1, 6, and 24.

### Project Code Map
- `data/sample/panel_monthly_sample.csv`: offline macro panel
- `src/features.py`: safe lag/diff/rolling feature helpers
- `src/data.py`: caching helpers

### Common Mistakes
- Fitting a VAR on I(1) levels without testing stationarity/cointegration.
- Interpreting Granger causality as economic causality.
- Not checking ordering sensitivity for Cholesky IRFs.
- Including too many variables in short samples (overparameterization).
- Ignoring confidence bands on IRFs.

<a id="summary"></a>
## Summary + Suggested Readings

VARs model multivariate dynamics; IRFs trace shock propagation; FEVDs attribute forecast uncertainty. The key caveat: everything depends on identification (variable ordering for Cholesky). Always check robustness.

Suggested readings:
- Stock & Watson (2001): "Vector Autoregressions" — accessible survey
- Lütkepohl (2005): *New Introduction to Multiple Time Series Analysis*
- Hamilton (1994): *Time Series Analysis* — Ch. 11–12
