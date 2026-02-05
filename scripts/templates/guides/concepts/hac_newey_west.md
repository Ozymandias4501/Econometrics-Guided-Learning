### Deep Dive: HAC / Newey–West standard errors (time-series inference)

HAC (heteroskedasticity-and-autocorrelation consistent) standard errors are the minimum correction for many time-series regressions.

#### 1) Intuition (plain English)

Time series residuals are rarely independent:
- shocks persist (serial correlation),
- variance changes across regimes (heteroskedasticity).

If you use naive OLS SE, you often understate uncertainty and overstate “significance.”

#### 2) Notation + setup (define symbols)

Consider a time-series regression:

$$
y_t = x_t'\\beta + u_t, \\quad t = 1,\\dots,T.
$$

Stacking:
$$
\\mathbf{y} = \\mathbf{X}\\beta + \\mathbf{u}.
$$

Classical OLS SE assume:
$$
\\mathrm{Var}(\\mathbf{u} \\mid \\mathbf{X}) = \\sigma^2 I_T.
$$

But in time series, we often have:
- $\\mathrm{Var}(u_t)$ changes over time (heteroskedasticity),
- $\\mathrm{Cov}(u_t, u_{t-k}) \\neq 0$ for some lags $k$ (autocorrelation).

#### 3) Estimation mechanics: coefficients vs uncertainty

OLS coefficients still equal:
$$
\\hat\\beta = (X'X)^{-1}X'y.
$$

HAC changes only the variance estimate:

$$
\\widehat{\\mathrm{Var}}_{HAC}(\\hat\\beta)
= (X'X)^{-1} \\left(X'\\hat\\Omega X\\right) (X'X)^{-1}.
$$

**What each term means**
- $\\hat\\Omega$ estimates error covariance across time (including lagged autocovariances).

#### 4) Newey–West: a common HAC choice

Newey–West constructs $\\hat\\Omega$ by combining residual autocovariances up to a maximum lag $L$:

$$
\\hat\\Omega_{NW} = \\hat\\Gamma_0 + \\sum_{k=1}^{L} w_k (\\hat\\Gamma_k + \\hat\\Gamma_k'),
\\quad w_k = 1 - \\frac{k}{L+1}.
$$

**What each term means**
- $\\hat\\Gamma_0$: contemporaneous covariance term.
- $\\hat\\Gamma_k$: lag-$k$ covariance contribution.
- $w_k$: Bartlett weights downweight higher lags.
- $L$: maximum lag included (tuning choice).

#### 5) Choosing `maxlags` (why it’s a sensitivity parameter)

There is no universally correct $L$.
Reasonable habits:
- quarterly data: try 1, 2, 4,
- monthly data: try 3, 6, 12,
- report sensitivity if inference changes.

If results flip sign/significance dramatically across plausible $L$, treat inference as fragile.

#### 6) Mapping to code (statsmodels)

In `statsmodels`:
- fit OLS normally,
- request HAC covariance:

```python
res_hac = res.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': 4})
```

#### 7) Diagnostics + robustness (minimum set)

1) **Residual autocorrelation**
- inspect ACF/PACF or Durbin–Watson style diagnostics.

2) **HAC sensitivity**
- try several `maxlags` values; report if conclusions change.

3) **Stationarity checks**
- nonstationarity can create spurious inference even with HAC.

4) **Stability**
- fit on subperiods or rolling windows; do coefficients drift?

#### 8) Interpretation + reporting

HAC SE are a correction for time dependence in errors.
They do **not**:
- fix omitted variables,
- fix nonstationarity,
- identify causal effects.

Report:
- coefficient + HAC SE (and chosen `maxlags`),
- a short statement about why HAC is needed (autocorrelation evidence),
- stability/sensitivity checks when possible.

#### Exercises

- [ ] Fit a time-series regression and compute naive SE and HAC SE; compare.
- [ ] Vary `maxlags` across a small set and report sensitivity of your main CI.
- [ ] Simulate AR(1) errors and show that naive SE are too small relative to HAC.
- [ ] Write 5 sentences: “When HAC is appropriate” vs “when HAC is not enough.”
