# Guide: 04a_residual_diagnostics

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/04a_residual_diagnostics.ipynb`.

After fitting an OLS model, the residuals are the first line of defence for detecting specification problems. This module covers formal tests for heteroskedasticity, serial correlation, functional-form misspecification, and structural breaks. When these tests reject, you know the model needs repair -- either through robust standard errors, re-specification, or a different estimator entirely.

### Key Terms (defined)
- **Residual** ($\hat\varepsilon_i = y_i - \hat y_i$): the observed prediction error; your window into what the model missed.
- **Heteroskedasticity**: error variance that depends on one or more regressors; OLS coefficients remain unbiased but standard errors are wrong.
- **Serial correlation (autocorrelation)**: errors correlated across observations (typically time); standard errors and test statistics are unreliable.
- **Breusch-Pagan (BP) test**: regresses squared residuals on regressors; tests $H_0$: homoskedasticity.
- **White test**: a more general heteroskedasticity test that includes squares and cross-products of regressors.
- **Durbin-Watson (DW) statistic**: a quick check for first-order autocorrelation (values near 2 suggest no autocorrelation).
- **Breusch-Godfrey (BG) test**: tests for higher-order serial correlation in the error term.
- **RESET test**: Ramsey's Regression Equation Specification Error Test; adds powers of fitted values to detect functional-form misspecification.
- **Chow test**: tests for a structural break at a known date by comparing full-sample vs split-sample regressions.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Residual plots (residuals vs fitted, residuals vs regressors, Q-Q plot)
- Complete notebook section: Breusch-Pagan and White tests for heteroskedasticity
- Complete notebook section: Durbin-Watson statistic and Breusch-Godfrey test for serial correlation
- Complete notebook section: Ramsey RESET test
- Complete notebook section: Chow test for structural break
- Produce at least one residual diagnostic plot and describe the pattern.
- Run at least two formal tests and state your conclusion about the OLS assumptions.

### Alternative Example (Not the Notebook Solution)
```python
# Toy heteroskedasticity test (not the notebook data):
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

rng = np.random.default_rng(0)
n = 300
x = rng.uniform(1, 10, n)
y = 2 + 3 * x + rng.normal(0, 0.5 * x, n)  # variance grows with x
X = sm.add_constant(x)
res = sm.OLS(y, X).fit()

lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(res.resid, X)
print(f"BP LM stat: {lm_stat:.2f}, p-value: {lm_pval:.4f}")
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: Breusch-Pagan and White tests for heteroskedasticity

Heteroskedasticity does not bias OLS coefficients, but it invalidates classical standard errors and test statistics. These two tests give you a formal verdict.

#### 1) Intuition (plain English)

If the spread of residuals changes systematically with one or more regressors, the errors are heteroskedastic. The Breusch-Pagan test asks: "Can I predict squared residuals from the regressors?" If yes, the error variance is not constant.

#### 2) Breusch-Pagan test: mechanics

1. Estimate the OLS model and obtain residuals $\hat\varepsilon_i$.
2. Compute $\hat\varepsilon_i^2$.
3. Regress $\hat\varepsilon_i^2$ on the original regressors $X$.
4. Under $H_0$ (homoskedasticity), the test statistic:

$$
LM = n \cdot R^2_{\text{aux}}
$$

follows a $\chi^2(K)$ distribution, where $K$ is the number of regressors (excluding the constant) in the auxiliary regression.

#### 3) White test: mechanics

The White test is a more general version. Instead of regressing $\hat\varepsilon_i^2$ on $X$ alone, it regresses on:

$$
\hat\varepsilon_i^2 \text{ on } [X,\; X^2,\; X_j \cdot X_k \text{ for } j < k]
$$

This catches nonlinear patterns in the variance. The cost is more degrees of freedom consumed, so it has lower power in small samples.

#### 4) Python implementation

```python
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Breusch-Pagan
bp_stat, bp_pval, _, _ = het_breuschpagan(res.resid, res.model.exog)

# White
white_stat, white_pval, _, _ = het_white(res.resid, res.model.exog)

print(f"BP test: stat={bp_stat:.2f}, p={bp_pval:.4f}")
print(f"White test: stat={white_stat:.2f}, p={white_pval:.4f}")
```

#### 5) What to do when the test rejects

- **Quick fix**: use heteroskedasticity-robust standard errors (HC3). OLS coefficients are fine; only inference changes.
- **Better fix**: investigate *why* the variance changes. A log transformation of $y$ often stabilizes variance. Alternatively, use WLS (see guide 07).
- **Do not ignore it**: classical SE will be too small in high-variance regions, producing misleadingly small p-values.

#### Exercises

- [ ] Run the BP test on an OLS model. If it rejects, re-estimate with HC3 SE and compare confidence intervals.
- [ ] Run the White test and compare its conclusion with the BP test. When might they disagree?
- [ ] Create a residuals-vs-fitted plot and visually confirm the heteroskedasticity pattern that the test detected.
- [ ] Show that taking $\ln(y)$ can stabilize the variance and repeat the BP test.

---

### Deep Dive: Serial correlation tests (DW, BG)

Serial correlation is the time-series counterpart of heteroskedasticity: it does not bias OLS coefficients (under strict exogeneity), but it destroys inference if you use standard errors that assume independence.

#### 1) Intuition (plain English)

If today's error is predictable from yesterday's error, the errors are serially correlated. Positive autocorrelation (the most common case) makes naive standard errors too small, leading to over-rejection of null hypotheses.

#### 2) Durbin-Watson statistic

The DW statistic tests for first-order autocorrelation:

$$
DW = \frac{\sum_{t=2}^{n}(\hat\varepsilon_t - \hat\varepsilon_{t-1})^2}{\sum_{t=1}^{n}\hat\varepsilon_t^2}.
$$

Rules of thumb:
- $DW \approx 2$: no first-order autocorrelation.
- $DW \ll 2$ (e.g., $DW < 1.5$): positive autocorrelation.
- $DW \gg 2$ (e.g., $DW > 2.5$): negative autocorrelation (rare in practice).
- $DW \approx 2(1 - \hat\rho)$ where $\hat\rho$ is the first-order autocorrelation of residuals.

**Limitation**: DW only tests AR(1) and has an inconclusive region. Also, it is biased toward 2 when a lagged dependent variable is included.

#### 3) Breusch-Godfrey test: higher-order serial correlation

The BG test generalizes to AR($p$) serial correlation:

1. Estimate OLS and obtain $\hat\varepsilon_t$.
2. Regress $\hat\varepsilon_t$ on $X_t$ and $\hat\varepsilon_{t-1}, \dots, \hat\varepsilon_{t-p}$.
3. Under $H_0$ (no serial correlation up to order $p$):

$$
LM = n \cdot R^2_{\text{aux}} \sim \chi^2(p).
$$

The BG test works even when the model includes lagged dependent variables, making it strictly superior to DW in that setting.

#### 4) Python implementation

```python
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

# Durbin-Watson (available directly from OLS results)
dw = sm.stats.durbin_watson(res.resid)
print(f"Durbin-Watson: {dw:.3f}")

# Breusch-Godfrey for up to 4 lags
bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(res, nlags=4)
print(f"BG test: stat={bg_stat:.2f}, p={bg_pval:.4f}")
```

#### 5) What to do when the test rejects

- **Quick fix**: use HAC (Newey-West) standard errors. Choose `maxlags` using a rule like $\lfloor 0.75 \cdot n^{1/3} \rfloor$.
- **Model fix**: if serial correlation is strong, consider adding lagged dependent variables, ARMA errors, or switching to a time-series model (see module 08).
- **Think about the source**: serial correlation often signals an omitted dynamic or trending variable.

#### Exercises

- [ ] Compute the Durbin-Watson statistic for a time-series OLS model and interpret the result.
- [ ] Run the Breusch-Godfrey test with $p = 1, 4, 12$ and compare the results. What does rejection at $p = 4$ but not $p = 1$ suggest?
- [ ] Compare naive SE vs HAC SE when the BG test rejects. Which coefficients lose significance?
- [ ] Add a lagged dependent variable and re-run the BG test. Does the autocorrelation disappear?

---

### Deep Dive: RESET test and structural breaks (Chow)

The RESET test checks whether your linear specification is adequate. The Chow test checks whether the relationship is stable across sub-samples.

#### 1) Ramsey RESET test: functional form misspecification

The RESET test asks: "After fitting $y = X\beta$, do *powers* of $\hat{y}$ have additional explanatory power?"

Procedure:
1. Fit OLS and compute fitted values $\hat{y}_i$.
2. Augment the model with $\hat{y}_i^2$ (and optionally $\hat{y}_i^3$).
3. Test joint significance of the added terms using an F-test.

Under $H_0$ (correct specification), the added terms should have no explanatory power.

$$
F = \frac{(SSR_R - SSR_U)/q}{SSR_U / (n - k - q - 1)}
$$

where $q$ is the number of added powers.

#### 2) RESET in Python

```python
from statsmodels.stats.diagnostic import linear_reset

reset_result = linear_reset(res, power=3, use_f=True)
print(f"RESET F-stat: {reset_result.fvalue:.2f}, p-value: {reset_result.pvalue:.4f}")
```

If the RESET test rejects, consider:
- adding polynomial terms ($x^2$, $x^3$),
- log-transforming variables,
- adding interaction terms,
- using a nonparametric or flexible specification.

#### 3) Chow test for structural breaks

The Chow test checks whether the regression coefficients are the same across two sub-samples (e.g., before and after a policy change at time $t^*$).

Procedure:
1. Estimate the full-sample model: $SSR_{\text{full}}$ with $n$ observations and $K+1$ parameters.
2. Estimate separate models for each sub-sample: $SSR_1$ ($n_1$ obs) and $SSR_2$ ($n_2$ obs).
3. The F-statistic:

$$
F = \frac{[SSR_{\text{full}} - (SSR_1 + SSR_2)] / (K+1)}{(SSR_1 + SSR_2) / (n - 2(K+1))}
\sim F(K+1,\; n - 2(K+1)).
$$

#### 4) Chow test in Python

```python
import statsmodels.api as sm

# Split at observation t_star
df_pre = df.loc[:t_star]
df_post = df.loc[t_star+1:]

res_full = sm.OLS(y, X).fit()
res_pre = sm.OLS(y_pre, X_pre).fit()
res_post = sm.OLS(y_post, X_post).fit()

ssr_full = res_full.ssr
ssr_split = res_pre.ssr + res_post.ssr
k = X.shape[1]
n = len(y)

F_chow = ((ssr_full - ssr_split) / k) / (ssr_split / (n - 2 * k))
```

**Important**: the Chow test requires you to specify the break date *a priori*. If you search over dates and pick the one with the largest F-statistic, the critical values are wrong -- use supremum-Wald tests (Andrews, 1993) instead.

#### Exercises

- [ ] Run the RESET test on a level-level model. If it rejects, add a quadratic term and re-run. Does the rejection disappear?
- [ ] Perform a Chow test at a known structural break date (e.g., 2008 financial crisis). Report the F-statistic and your conclusion.
- [ ] Explain why searching over break dates invalidates the Chow test's p-value.
- [ ] Combine all diagnostic tests (BP, BG, RESET, Chow) into a single diagnostic report table for one model.

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`), VIF (`vif_table`)
- `src/evaluation.py`: regression metrics helpers (`regression_metrics`)
- `src/features.py`: feature helpers (`add_lag_features`, `add_rolling_features`)
- `src/plots.py`: plotting utilities
- `statsmodels.stats.diagnostic`: `het_breuschpagan`, `het_white`, `acorr_breusch_godfrey`, `linear_reset`

### Common Mistakes
- Running diagnostic tests on the wrong residuals (e.g., using standardized residuals when the test expects raw residuals).
- Interpreting a "pass" on the BP test as proof of homoskedasticity; the test has low power in small samples.
- Using the Durbin-Watson statistic when the model includes a lagged dependent variable (use BG instead).
- Searching over break dates with the Chow test and treating the resulting p-value as valid.
- Treating diagnostic test failure as a reason to abandon OLS entirely, when robust SE or re-specification may suffice.

<a id="summary"></a>
## Summary + Suggested Readings

Residual diagnostics bridge the gap between estimation and valid inference. You should now be able to:
- detect heteroskedasticity with BP/White tests and respond with robust SE or WLS,
- detect serial correlation with DW/BG tests and respond with HAC SE or dynamic models,
- test for functional-form misspecification with RESET, and
- test for structural breaks with the Chow test.

Suggested readings:
- Wooldridge, *Introductory Econometrics*, Ch. 8 (heteroskedasticity) and Ch. 12 (serial correlation)
- Greene, *Econometric Analysis*, Ch. 9 (specification tests)
- Ramsey (1969), "Tests for Specification Errors in Classical Linear Least-Squares Regression Analysis" (*JRSS-B*)
- Andrews (1993), "Tests for Parameter Instability and Structural Change With Unknown Change Point" (*Econometrica*)
