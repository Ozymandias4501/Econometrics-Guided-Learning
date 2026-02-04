### Deep Dive: HAC / Newey-West Standard Errors (Time-Series Inference)

Time series often violate the classic OLS assumptions.

> **Definition:** **Autocorrelation** means errors are correlated over time: $\mathrm{Cov}(\varepsilon_t, \varepsilon_{t-k}) \ne 0$.

> **Definition:** **Heteroskedasticity** means error variance changes over time: $\mathrm{Var}(\varepsilon_t)$ is not constant.

OLS coefficient estimates can remain the same, but the uncertainty (standard errors) can be wrong.

#### What OLS estimates vs what OLS assumes
OLS coefficients:

$$
\hat\beta = (X'X)^{-1}X'y
$$

Under the simplest assumptions (homoskedastic, uncorrelated errors), the covariance of $\hat\beta$ is:

$$
\widehat{\mathrm{Var}}(\hat\beta) = \hat\sigma^2 (X'X)^{-1}
$$

But if errors are autocorrelated and/or heteroskedastic, this variance estimate is not reliable.

#### HAC intuition (sandwich estimator)
HAC uses a "sandwich" form:

$$
\widehat{\mathrm{Var}}_{HAC}(\hat\beta) = (X'X)^{-1} \; (X'\widehat{\Omega}X) \; (X'X)^{-1}
$$

- The "bread" is $(X'X)^{-1}$.
- The "meat" is an estimate of the error covariance structure, including lagged autocovariances.

Newey-West is a common HAC estimator that downweights higher lags.

One way to think about the "meat" term is:
- compute residuals $\hat\varepsilon_t$
- estimate autocovariances of $\hat\varepsilon_t$ up to a maximum lag $L$
- combine them with weights $w_k$ (Newey-West commonly uses Bartlett weights)

You will often see formulas like:

$$
\widehat{\Omega}_{NW} = \Gamma_0 + \\sum_{k=1}^{L} w_k (\\Gamma_k + \\Gamma_k')
$$

Where:
- $\\Gamma_k$ estimates the lag-$k$ covariance contribution
- $w_k = 1 - \\frac{k}{L+1}$ (Bartlett) downweights higher lags

#### What changes and what does not
- Coefficients $\hat\beta$ do **not** change.
- Standard errors, t-stats, p-values, and confidence intervals **do** change.

#### Python demo: AR(1) errors make naive SE too small (commented)
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)

# Simulate a regression with autocorrelated errors
n = 200
x = rng.normal(size=n)

eps = np.zeros(n)
for t in range(1, n):
    # AR(1) structure: today's error depends on yesterday's
    eps[t] = 0.8 * eps[t-1] + rng.normal(scale=1.0)

y = 1.0 + 0.5 * x + eps

X = sm.add_constant(pd.DataFrame({'x': x}))
res = sm.OLS(y, X).fit()

# HAC with maxlags=4 (common to try for quarterly)
res_hac = res.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': 4})

print('coef:', res.params.to_dict())
print('naive SE:', res.bse.to_dict())
print('HAC SE  :', res_hac.bse)
```

#### Choosing `maxlags`
There is no perfect choice.
- Quarterly data: common to try 1, 2, 4.
- Monthly data: common to try 3, 6, 12.

Practical approach:
- try a small set of maxlags
- report sensitivity
- if inference flips wildly, treat the result as fragile

#### Project touchpoints (where HAC shows up in this repo)
- `src/econometrics.py` wraps this as `fit_ols_hac(df, y_col=..., x_cols=..., maxlags=...)`.
- Regression notebooks compare naive OLS SE to HAC SE and ask you to report sensitivity to `maxlags`.

#### Practical macro warning
In macro time series, p-values can be misleading due to:
- nonstationarity
- structural breaks
- small sample sizes (quarterly data has few points)

Use HAC as a minimum correction, then focus on stability and out-of-sample checks.
