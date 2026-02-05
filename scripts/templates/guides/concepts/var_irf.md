### VAR + IRF: multivariate dynamics and “shock propagation”

Vector autoregressions (VARs) are a core tool for macro dynamics: they model how multiple variables move together over time.

#### 1) Intuition (plain English)

Unemployment, production, and interest rates influence each other with lags.
A VAR is a flexible way to model these feedback loops without imposing a full structural model.

An impulse response function (IRF) then answers:
- “If we hit the system with a one-time shock today, how do variables respond over the next few periods?”

**Story example:** A policy rate shock might raise unemployment over several months and reduce production.

#### 2) Notation + setup (define symbols)

Let $y_t$ be a $k \\times 1$ vector of variables at time $t$:
$$
y_t =
\\begin{bmatrix}
\\text{UNRATE}_t \\\\
\\text{FEDFUNDS}_t \\\\
\\text{INDPRO}_t
\\end{bmatrix}.
$$

A VAR($p$) is:

$$
y_t = c + A_1 y_{t-1} + \\cdots + A_p y_{t-p} + \\varepsilon_t,
$$

where:
- $c$ is a $k \\times 1$ intercept vector,
- $A_j$ are $k \\times k$ coefficient matrices,
- $\\varepsilon_t$ is a $k \\times 1$ innovation vector with covariance matrix $\\Sigma$.

**What each term means**
- Each equation predicts one variable using lags of *all* variables.
- $\\Sigma$ captures contemporaneous correlation among innovations (important for IRFs).

#### 3) Assumptions: stationarity and stability

VAR inference typically assumes the system is stable (stationary).
Intuition: shocks should not blow up forever.

Formally, stability requires eigenvalues of the companion matrix to lie inside the unit circle.
Most software reports a stability check.

If your variables are not stationary, common fixes include:
- differencing / log-differencing,
- modeling cointegration with a VECM (not covered deeply here),
- restricting to stationary transformations.

#### 4) Estimation mechanics: OLS equation-by-equation

VAR coefficients can be estimated by OLS for each equation because the regressors are the same across equations.

Define the regressor vector:
$$
x_t' = [1, y_{t-1}', \\dots, y_{t-p}'].
$$

Then each equation is:
$$
y_{m,t} = x_t' \\theta_m + e_{m,t},
\\quad m = 1,\\dots,k,
$$
and OLS provides $\\hat\\theta_m$.

**What this means practically**
- Estimation is straightforward,
- the hard part is choosing transformations and lag length responsibly.

#### 5) Lag selection: why AIC/BIC are a starting point, not the finish line

Common criteria:
- AIC tends to choose more lags (better fit, higher variance),
- BIC tends to choose fewer lags (more parsimonious).

Practical approach:
- check a range of lags (e.g., 1–8 for monthly),
- confirm residual autocorrelation is not severe,
- prefer interpretability and stability over maximizing in-sample fit.

#### 6) Granger causality: predictive content, not structural causality

Variable $x$ “Granger-causes” $y$ if lagged $x$ terms help predict $y$ beyond lagged $y$.

This is a **forecasting** statement:
- it does not establish causal structure,
- it can be driven by omitted variables or common shocks.

#### 7) IRFs: from VAR to moving-average (MA) representation

If the VAR is stable, it has an MA form:

$$
y_t = \\mu + \\sum_{s=0}^{\\infty} \\Psi_s \\varepsilon_{t-s}.
$$

The matrices $\\Psi_s$ map an innovation today into future outcomes.
An IRF traces rows/columns of $\\Psi_s$ over horizons $s=0,1,2,\\dots$.

#### 8) Identification: orthogonalized IRFs and why ordering matters

Problem: VAR innovations $\\varepsilon_t$ are often correlated (covariance $\\Sigma$ is not diagonal).

To interpret a “one-unit shock,” you often orthogonalize innovations via a Cholesky factorization:
$$
\\Sigma = P P'.
$$

Define structural shocks $u_t$ with identity covariance:
$$
\\varepsilon_t = P u_t,
\\qquad \\mathrm{Var}(u_t)=I.
$$

Now a “shock to variable 1” is a shock to $u_{1t}$, which is orthogonal to others.

**Key implication**
- The Cholesky decomposition depends on the ordering of variables.
- So orthogonalized IRFs are conditional on that identification assumption.

#### 9) Diagnostics + robustness (minimum set)

1) **Stationarity / stability**
- Confirm transformations lead to a stable VAR (software stability check).

2) **Residual autocorrelation**
- If residuals remain autocorrelated, your lag length may be too short.

3) **Ordering sensitivity (for orth IRFs)**
- Re-order variables and see if qualitative IRF conclusions change.

4) **Out-of-sample forecasting sanity**
- Even if your goal is IRFs, forecasting performance can reveal misspecification.

#### 10) Interpretation + reporting

When reporting a VAR/IRF analysis:
- specify transformations,
- specify lag choice method,
- specify identification (ordering for Cholesky orth IRFs),
- report stability checks and diagnostics.

**What this does NOT mean**
- Granger causality ≠ structural causality.
- Orthogonalized IRFs ≠ “true policy shocks” unless the identification is defensible.

#### Exercises

- [ ] Fit a VAR on stationary transformations and report the selected lag length.
- [ ] Run a Granger causality test and interpret it as “predictive content,” not causality.
- [ ] Plot IRFs under two different variable orderings; compare and explain differences.
- [ ] Check residual autocorrelation; increase lags and see whether diagnostics improve.
