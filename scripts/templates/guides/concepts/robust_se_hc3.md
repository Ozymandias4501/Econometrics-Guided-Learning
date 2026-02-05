### Deep Dive: Robust Standard Errors (HC3) for Cross-Sectional Data

Robust standard errors are about **honest uncertainty** when error variance differs across observations.

#### 1) Intuition (plain English)

In cross-sectional micro data, the variance of “unexplained” outcomes often differs systematically:
- income has higher variance at higher education levels,
- spending variability rises with income,
- measurement error differs across regions.

If you assume constant error variance when it is not true, your coefficient estimate may be fine, but your **standard errors** can be wrong—often too small.

#### 2) Notation + setup (define symbols)

Regression model:

$$
\\mathbf{y} = \\mathbf{X}\\beta + \\mathbf{u}.
$$

**What each term means**
- $\\mathbf{y}$: $n\\times 1$ outcome vector.
- $\\mathbf{X}$: $n\\times p$ design matrix (includes intercept).
- $\\beta$: $p\\times 1$ coefficient vector.
- $\\mathbf{u}$: $n\\times 1$ error vector.

Classical OLS inference assumes homoskedasticity:
$$
\\mathrm{Var}(\\mathbf{u} \\mid \\mathbf{X}) = \\sigma^2 I_n.
$$

Heteroskedasticity means:
$$
\\mathrm{Var}(\\mathbf{u} \\mid \\mathbf{X}) = \\Omega,
\\quad \\text{where } \\Omega \\text{ is not } \\sigma^2 I_n.
$$

Often $\\Omega$ is diagonal with unequal variances (but we don’t need to specify the exact pattern to build robust SE).

#### 3) Estimation mechanics: what changes and what does not

OLS coefficients:
$$
\\hat\\beta = (X'X)^{-1}X'y.
$$

**Key fact**
- $\\hat\\beta$ is the same whether you use classical or robust SE.
- What changes is the estimated variance of $\\hat\\beta$ (and therefore t-stats, p-values, and CI).

#### 4) The robust “sandwich” covariance estimator

A general heteroskedasticity-robust variance estimator has the form:

$$
\\widehat{\\mathrm{Var}}(\\hat\\beta)
= (X'X)^{-1} \\left(X'\\hat\\Omega X\\right) (X'X)^{-1}.
$$

**What each term means**
- “Bread”: $(X'X)^{-1}$ is the usual OLS matrix.
- “Meat”: $X'\\hat\\Omega X$ estimates the error variance structure.

Different HC estimators choose different $\\hat\\Omega$.

#### 5) Why HC3 specifically? (leverage adjustment)

HC3 is designed to be more conservative in finite samples when some points have high leverage.

Define:
- residuals $\\hat u_i$,
- leverage $h_{ii}$ (diagonal of the hat matrix $H = X(X'X)^{-1}X'$):
$$
h_{ii} = x_i'(X'X)^{-1}x_i.
$$

HC3 uses:
$$
\\hat\\Omega_{ii}^{HC3} = \\frac{\\hat u_i^2}{(1-h_{ii})^2}.
$$

**Interpretation**
- if a point has high leverage ($h_{ii}$ large), it gets more conservative variance contribution.

#### 6) Mapping to code (statsmodels)

In `statsmodels`, you can request HC3 in two common ways:
- `res = sm.OLS(y, X).fit(cov_type='HC3')`
- or `res_hc3 = res.get_robustcov_results(cov_type='HC3')`

#### 7) Diagnostics + robustness (minimum set)

1) **Residual vs fitted plot**
- look for “fan shapes” where variance increases with fitted values.

2) **Compare naive vs HC3 SE**
- report the ratio; big changes mean heteroskedasticity mattered.

3) **Leverage / influential points**
- if a few points dominate, inference is fragile; consider robust checks.

4) **Spec sensitivity**
- add/remove plausible controls; see if estimates are stable (robust SE does not fix omitted variables).

#### 8) Interpretation + reporting

HC3 improves uncertainty estimates under heteroskedasticity.
It does **not**:
- fix bias from confounding,
- make a coefficient causal,
- correct misspecification.

Report:
- coefficient + HC3 SE,
- sample size,
- a quick heteroskedasticity diagnostic (plot or comparison).

#### Exercises

- [ ] Simulate heteroskedastic data and compare naive vs HC3 SE; explain why the coefficient stays similar.
- [ ] Fit the same regression with HC0/HC1/HC3 (if available) and compare SE; which is most conservative?
- [ ] Identify a high-leverage point and explain how HC3 changes its influence on uncertainty.
- [ ] Write 5 sentences: “What robust SE fixes” vs “what it does not fix.”
