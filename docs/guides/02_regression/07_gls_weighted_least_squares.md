# Guide: 07_gls_weighted_least_squares

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/07_gls_weighted_least_squares.ipynb`.

When OLS assumptions about the error structure fail -- heteroskedasticity or autocorrelation -- GLS (Generalized Least Squares) provides a principled way to recover efficient estimation and valid inference. WLS (Weighted Least Squares) is the special case for heteroskedasticity. FGLS (Feasible GLS) estimates the error structure from the data in a first step. This module covers all three, plus GLSAR for autocorrelated errors.

### Key Terms (defined)
- **GLS (Generalized Least Squares)**: transforms the model by the inverse square root of the error covariance matrix $\Omega$ to restore efficient estimation.
- **WLS (Weighted Least Squares)**: the diagonal-$\Omega$ special case; each observation is weighted inversely to its error variance.
- **FGLS (Feasible GLS)**: a two- or three-step procedure that first estimates $\Omega$ from OLS residuals, then applies GLS.
- **GLSAR**: GLS with AR($p$) errors; iteratively estimates the autocorrelation parameters and transforms the model.
- **Efficiency**: among unbiased estimators, the one with the smallest variance; GLS is efficient when $\Omega$ is known.
- **Gauss-Markov theorem**: OLS is BLUE (Best Linear Unbiased Estimator) under homoskedasticity and no autocorrelation. When these fail, GLS becomes BLUE.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Diagnose heteroskedasticity (BP/White test from 04a)
- Complete notebook section: Construct weights and fit WLS
- Complete notebook section: Compare OLS vs WLS coefficients and standard errors
- Complete notebook section: Implement FGLS three-step procedure
- Complete notebook section: Fit GLSAR for autocorrelated errors
- Produce side-by-side coefficient comparison tables (OLS, OLS-HC3, WLS, FGLS).
- Plot weighted residuals to verify that WLS has stabilized the variance.

### Alternative Example (Not the Notebook Solution)
```python
# Toy WLS example (not the notebook data):
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(42)
n = 500
x = rng.uniform(1, 20, n)
sigma_i = 0.3 * x  # variance proportional to x
y = 5 + 2 * x + rng.normal(0, sigma_i, n)

X = sm.add_constant(x)
res_ols = sm.OLS(y, X).fit()

# WLS with weights = 1/x^2 (inverse variance)
weights = 1.0 / x**2
res_wls = sm.WLS(y, X, weights=weights).fit()
print("OLS SE:", res_ols.bse.round(4))
print("WLS SE:", res_wls.bse.round(4))
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: WLS mechanics and weight choice

WLS is the most common form of GLS in cross-sectional applications. Understanding weight construction is the key practical skill.

#### 1) Intuition (plain English)

Under heteroskedasticity, some observations have noisier errors than others. OLS treats all observations equally, which is inefficient -- it over-weights noisy observations. WLS down-weights noisy observations and up-weights precise ones, producing tighter standard errors and more efficient estimates.

#### 2) Notation and the GLS/WLS estimator

The general linear model with non-spherical errors:

$$
y = X\beta + \varepsilon, \quad \mathrm{Var}(\varepsilon) = \sigma^2 \Omega
$$

where $\Omega$ is a known $n \times n$ positive definite matrix.

The GLS estimator:

$$
\hat\beta_{\text{GLS}} = (X'\Omega^{-1}X)^{-1} X'\Omega^{-1} y
$$

with variance:

$$
\mathrm{Var}(\hat\beta_{\text{GLS}}) = \sigma^2 (X'\Omega^{-1}X)^{-1}.
$$

**WLS special case**: when $\Omega$ is diagonal, $\Omega = \mathrm{diag}(\omega_1, \dots, \omega_n)$, the GLS estimator simplifies to WLS with weights $w_i = 1/\omega_i$:

$$
\hat\beta_{\text{WLS}} = \arg\min_\beta \sum_{i=1}^n w_i (y_i - x_i'\beta)^2.
$$

#### 3) How to choose weights

The ideal weight for observation $i$ is $w_i = 1 / \mathrm{Var}(\varepsilon_i)$. In practice, you estimate the variance function:

| Assumed variance form | Weight $w_i$ |
|---|---|
| $\mathrm{Var}(\varepsilon_i) \propto x_i$ | $w_i = 1/x_i$ |
| $\mathrm{Var}(\varepsilon_i) \propto x_i^2$ | $w_i = 1/x_i^2$ |
| $\mathrm{Var}(\varepsilon_i) \propto \bar{n}_i^{-1}$ (group means) | $w_i = \bar{n}_i$ (group size) |
| Estimated from auxiliary regression | See FGLS below |

A common diagnostic: plot $|\hat\varepsilon_i|$ or $\hat\varepsilon_i^2$ against each regressor. The pattern tells you the variance function.

#### 4) WLS as a transformed OLS

WLS is equivalent to applying OLS to the transformed model:

$$
\sqrt{w_i}\, y_i = \sqrt{w_i}\, x_i'\beta + \sqrt{w_i}\, \varepsilon_i
$$

or in matrix form, letting $P = \mathrm{diag}(\sqrt{w_1}, \dots, \sqrt{w_n})$:

$$
Py = PX\beta + P\varepsilon.
$$

The transformed errors $P\varepsilon$ are homoskedastic if the weights are correct.

#### 5) Verifying WLS worked

After fitting WLS, check the *weighted* residuals $\sqrt{w_i}\,\hat\varepsilon_i$ for remaining heteroskedasticity:

```python
import matplotlib.pyplot as plt

weighted_resid = np.sqrt(weights) * res_wls.resid
plt.scatter(res_wls.fittedvalues, weighted_resid, alpha=0.3)
plt.axhline(0, color='grey', ls='--')
plt.xlabel('Fitted values')
plt.ylabel('Weighted residuals')
plt.title('Weighted residual plot (should show constant spread)')
plt.show()
```

#### Exercises

- [ ] Fit OLS on data with variance proportional to $x$. Run the BP test, then fit WLS with $w_i = 1/x_i$. Compare SE.
- [ ] Show that WLS with equal weights ($w_i = 1$) reproduces OLS exactly.
- [ ] Plot unweighted and weighted residuals side by side. Does WLS stabilize the spread?
- [ ] Try two different weight functions (e.g., $1/x$ vs $1/x^2$) and compare; which fits the true DGP better?

---

### Deep Dive: FGLS three-step procedure

In practice, you rarely know the true $\Omega$. FGLS estimates it from the data, then proceeds as if $\Omega$ were known.

#### 1) Intuition (plain English)

FGLS is a "learn-then-correct" approach. First, fit OLS to get residuals. Second, model the variance pattern from those residuals. Third, use the estimated variances as weights in WLS. This is feasible (hence the F) because it uses the data itself to estimate the error structure.

#### 2) The three steps

**Step 1**: Fit OLS and save residuals $\hat\varepsilon_i$.

**Step 2**: Estimate the variance function. A common approach:

$$
\ln(\hat\varepsilon_i^2) = \gamma_0 + \gamma_1 z_{i1} + \cdots + \gamma_p z_{ip} + v_i
$$

where $z_i$ are variables you believe drive the heteroskedasticity (often the same regressors). The fitted values give estimated log-variances:

$$
\hat\sigma_i^2 = \exp(\hat\gamma_0 + \hat\gamma_1 z_{i1} + \cdots).
$$

**Step 3**: Use WLS with weights $w_i = 1/\hat\sigma_i^2$.

#### 3) Python implementation

```python
import numpy as np
import statsmodels.api as sm

# Step 1: OLS
res_ols = sm.OLS(y, X).fit()

# Step 2: model the variance
log_resid_sq = np.log(res_ols.resid**2)
Z = sm.add_constant(df[['x1', 'x2']])  # variance drivers
aux_res = sm.OLS(log_resid_sq, Z).fit()
sigma2_hat = np.exp(aux_res.fittedvalues)

# Step 3: WLS
weights = 1.0 / sigma2_hat
res_fgls = sm.WLS(y, X, weights=weights).fit()
```

#### 4) Properties and caveats

- FGLS is *asymptotically* efficient: as $n \to \infty$, it achieves the same variance as GLS with known $\Omega$.
- In finite samples, FGLS can be *less* efficient than OLS if $\Omega$ is badly estimated. This is especially risky with small samples or a misspecified variance function.
- **Rule of thumb**: if $n < 100$ or the variance function is uncertain, prefer OLS with HC3 robust SE over FGLS.
- FGLS standard errors assume $\hat\Omega$ is the true $\Omega$; they do not account for estimation uncertainty in the first step.

#### Exercises

- [ ] Implement the full three-step FGLS procedure from scratch (no built-in FGLS function).
- [ ] Compare FGLS SE with OLS-HC3 SE. Which are smaller? Is the efficiency gain worth the additional assumptions?
- [ ] Deliberately misspecify the variance function (e.g., use $1/x$ weights when the true DGP has $\sigma \propto x^2$). Show that FGLS can be worse than OLS.
- [ ] Iterate the FGLS procedure (re-estimate $\Omega$ from FGLS residuals) and check whether the estimates converge.

---

### Deep Dive: GLSAR for autocorrelated errors

When the error term follows an AR process, GLS can be applied by estimating the autocorrelation and transforming the data. This is the GLSAR (GLS with AR errors) approach.

#### 1) Intuition (plain English)

In time-series regressions, errors often follow an AR(1) pattern: $\varepsilon_t = \rho \varepsilon_{t-1} + u_t$. If you know $\rho$, you can "quasi-difference" the data to remove the autocorrelation. GLSAR estimates $\rho$ iteratively and applies the transformation.

#### 2) The AR(1) error model

$$
y_t = x_t'\beta + \varepsilon_t, \quad \varepsilon_t = \rho\,\varepsilon_{t-1} + u_t, \quad u_t \sim \text{i.i.d.}(0, \sigma_u^2)
$$

The Cochrane-Orcutt transformation: for $t \geq 2$,

$$
y_t - \rho\, y_{t-1} = (x_t - \rho\, x_{t-1})'\beta + u_t.
$$

This transformed model has i.i.d. errors, so OLS on the transformed data is efficient.

#### 3) The Prais-Winsten modification

Cochrane-Orcutt drops the first observation. Prais-Winsten retains it by transforming:

$$
\sqrt{1 - \rho^2}\, y_1 = \sqrt{1 - \rho^2}\, x_1'\beta + \sqrt{1 - \rho^2}\, \varepsilon_1.
$$

This is more efficient in small samples.

#### 4) Iterative GLSAR in statsmodels

```python
import statsmodels.api as sm

# GLSAR with AR(1) errors, iterated to convergence
mod = sm.GLSAR(y, X, rho=1)  # rho=1 means AR(1) order
res_glsar = mod.iterative_fit(maxiter=50)
print(f"Estimated rho: {mod.rho:.3f}")
print(res_glsar.summary())
```

The `rho` parameter in `sm.GLSAR()` specifies the AR order (number of lags), not the correlation value. The `iterative_fit` method alternates between estimating $\beta$ and estimating $\rho$ until convergence.

#### 5) When to use GLSAR vs HAC

| Situation | Recommended approach |
|---|---|
| Mild autocorrelation, inference is primary goal | HAC (Newey-West) SE |
| Strong autocorrelation, efficiency matters | GLSAR or Prais-Winsten |
| Unknown autocorrelation structure | HAC SE (more robust) |
| Small sample, known AR(1) structure | GLSAR (more efficient) |

In general, HAC SE are safer because they do not require specifying the autocorrelation structure correctly. GLSAR offers efficiency gains when the AR structure is well-specified.

#### Exercises

- [ ] Fit GLSAR on a time-series regression and report the estimated $\hat\rho$. Compare with the autocorrelation of OLS residuals.
- [ ] Compare OLS, OLS-HAC, and GLSAR standard errors. Which produces the tightest confidence intervals?
- [ ] Manually implement the Cochrane-Orcutt transformation for one iteration and verify it matches the GLSAR first-step result.
- [ ] Simulate AR(2) errors and show that GLSAR with AR(1) mis-specification still leaves residual autocorrelation.

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`), design matrix (`design_matrix`)
- `src/evaluation.py`: regression metrics helpers (`regression_metrics`)
- `src/features.py`: feature helpers (`add_lag_features`, `add_rolling_features`)
- `statsmodels.regression.linear_model`: `WLS`, `GLS`, `GLSAR`
- `statsmodels.stats.diagnostic`: `het_breuschpagan`, `het_white`, `acorr_breusch_godfrey`

### Common Mistakes
- Using WLS weights that are the variance (not the inverse variance). statsmodels `WLS` expects $w_i = 1/\sigma_i^2$.
- Treating FGLS standard errors as if $\hat\Omega$ were the true $\Omega$; they understate uncertainty in small samples.
- Applying FGLS when $n$ is small and the variance function is uncertain -- OLS with HC3 is often more reliable.
- Confusing the `rho` parameter in `sm.GLSAR()` (the AR order) with the autocorrelation coefficient itself.
- Forgetting to check weighted residuals after WLS to verify that heteroskedasticity has been corrected.

<a id="summary"></a>
## Summary + Suggested Readings

GLS, WLS, and FGLS provide efficient estimation when the OLS error assumptions fail. You should now be able to:
- diagnose heteroskedasticity and construct appropriate WLS weights,
- implement the three-step FGLS procedure and understand its finite-sample caveats,
- apply GLSAR for autocorrelated errors, and
- choose between robust SE and GLS-family estimators based on sample size and knowledge of the error structure.

Suggested readings:
- Wooldridge, *Introductory Econometrics*, Ch. 8 (heteroskedasticity, WLS, FGLS)
- Greene, *Econometric Analysis*, Ch. 9 (GLS, FGLS, autocorrelation)
- Cochrane & Orcutt (1949), "Application of Least Squares Regression to Relationships Containing Auto-Correlated Error Terms" (*JASA*)
- statsmodels documentation: WLS, GLS, GLSAR
