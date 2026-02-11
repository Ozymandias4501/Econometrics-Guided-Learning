# Guide: 04_inference_time_series_hac

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/04_inference_time_series_hac.ipynb`.

This is the **definitive reference** for hypothesis testing and HAC (Newey-West) standard errors in this project. Other guides cross-reference this one rather than duplicating the content.

### Key Terms (defined)
- **Hypothesis test**: a structured procedure for assessing whether observed data are compatible with a specific claim (the null hypothesis).
- **p-value**: the probability, under the null hypothesis and model assumptions, of observing a test statistic at least as extreme as the one computed from data.
- **Confidence interval (CI)**: a range of parameter values consistent with the data at a given confidence level (e.g., 95%).
- **Type I error**: rejecting the null when it is true (false positive). Controlled by significance level $\alpha$.
- **Type II error**: failing to reject the null when it is false (false negative). Related to statistical power.
- **Standard error (SE)**: estimated standard deviation of a coefficient estimator; determines the width of CI and the magnitude of t-stats.
- **HAC (Heteroskedasticity-and-Autocorrelation Consistent)**: a class of SE estimators valid when errors are both heteroskedastic and serially correlated.
- **Newey-West**: the most common HAC estimator; uses Bartlett (triangular) kernel weights to downweight higher-lag autocovariances.
- **Kernel bandwidth (maxlags)**: the maximum lag $L$ included in the HAC estimator; a tuning parameter with a bias-variance tradeoff.
- **Bartlett weights**: the linearly declining weights $w_k = 1 - k/(L+1)$ used in Newey-West to ensure a positive semi-definite covariance matrix.

### How To Read This Guide
- Use **Step-by-Step** for the notebook checklist and a simulation-based alternative example.
- Use **Technical Explanations** for the full treatment of hypothesis testing, HAC mechanics, and the SE comparison table.
- For OLS foundations (derivation, assumptions, coefficient interpretation), see [Guide 00](00_single_factor_regression_micro.md#technical).
- For macro-specific regression issues (stationarity, spurious regression, structural breaks), see [Guide 02](02_single_factor_regression_macro.md#technical).

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- [ ] Fit an OLS regression on the notebook's time-series data.
- [ ] Plot residuals and inspect the ACF/PACF for evidence of serial correlation.
- [ ] Compute naive OLS standard errors and note the coefficient p-values.
- [ ] Re-fit with HAC SE using at least two different `maxlags` values (e.g., 2 and 8).
- [ ] Compare naive vs HAC standard errors side by side; note which coefficients change significance.
- [ ] Write a short interpretation: what does the sensitivity to `maxlags` tell you about the reliability of your inference?

### Alternative Example: Simulating AR(1) Errors to Show Naive SE Bias

This example generates data where the true coefficient is known, errors are AR(1), and naive SE systematically understate uncertainty.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)
T = 200
beta_true = 0.5

x = rng.normal(size=T)

# Generate AR(1) errors: u_t = 0.7 * u_{t-1} + e_t
e = rng.normal(size=T)
u = np.zeros(T)
u[0] = e[0]
for t in range(1, T):
    u[t] = 0.7 * u[t-1] + e[t]

y = 1.0 + beta_true * x + u
X = sm.add_constant(pd.DataFrame({'x': x}))
res = sm.OLS(y, X).fit()

# Naive SE vs HAC SE at multiple bandwidths
print(f"True beta: {beta_true}")
print(f"Estimated beta: {res.params['x']:.4f}")
print(f"Naive SE:       {res.bse['x']:.4f}")

for L in [2, 4, 8]:
    res_hac = res.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': L})
    print(f"HAC SE (L={L}):   {res_hac.bse['x']:.4f}")
```

You should see that naive SE are smaller than HAC SE, because the AR(1) errors create positive serial correlation that naive SE ignore. The HAC SE grow somewhat as `maxlags` increases, reflecting the bias-variance tradeoff in bandwidth selection.

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Cross-Reference: OLS Foundations

The OLS estimator, its derivation, assumptions for unbiasedness, coefficient interpretation, and basic diagnostics are covered in [Guide 00: Single-Factor Regression (Micro)](00_single_factor_regression_micro.md#technical). This guide assumes you have read that material and focuses on **inference** -- the machinery that tells you how uncertain your estimates are.

---

### Hypothesis Testing: How to Read p-values Without Fooling Yourself

Hypothesis tests show up everywhere in econometrics output. The goal of this section is not to worship p-values, but to understand what they *are* and what they *are not*.

#### 1) Intuition (plain English)

A hypothesis test is a structured way to ask:
- "If the true effect were zero, how surprising is my estimate?"

It is **not** a direct answer to:
- "What is the probability the effect is real?"
- "Is my model correct?"

**Story example:** You regress unemployment on an interest-rate spread and get a small p-value.
That might mean:
- the relationship is real in-sample,
- or your SE are wrong (autocorrelation),
- or you tried many specs (multiple testing),
- or the effect is tiny but precisely estimated.

#### 2) Notation + setup (define symbols)

We usually test a claim about a population parameter $\theta$ (mean, regression coefficient, difference in means, ...).

Define:
- $H_0$: the **null hypothesis** (default claim),
- $H_1$: the **alternative hypothesis** (what you consider if evidence contradicts $H_0$),
- $T$: a **test statistic** computed from data,
- $\alpha$: a pre-chosen significance level (e.g., 0.05).

Example in regression:
- $H_0: \beta_j = 0$
- $H_1: \beta_j \neq 0$ (two-sided)

#### 3) Assumptions (why tests are conditional statements)

Every p-value is conditional on:
- the statistical model (e.g., OLS assumptions),
- the standard error estimator you use (naive vs robust vs HAC vs clustered),
- the sample and selection process.

If those assumptions fail, the p-value may be meaningless.

#### 4) Estimation mechanics in OLS: where t-stats come from

OLS estimates coefficients:

$$
\hat\beta = (X'X)^{-1}X'y.
$$

For coefficient $\beta_j$, you compute an estimated standard error $\widehat{SE}(\hat\beta_j)$.

The t-statistic for testing $H_0: \beta_j = 0$ is:

$$
t_j = \frac{\hat\beta_j - 0}{\widehat{SE}(\hat\beta_j)}.
$$

**What each term means**
- numerator: your estimated effect.
- denominator: your uncertainty estimate.
- large |t| means "many standard errors away from 0."

Under suitable assumptions, $t_j$ is compared to a t distribution (or asymptotic normal), producing a p-value.

#### 5) What the p-value actually means

> **Definition:** The **p-value** is the probability (under the null and model assumptions) of observing a test statistic at least as extreme as what you observed.

So:
- p-value is about the *data under the null model*,
- not about the probability the null is true.

Also: p-values do not measure effect size.

#### 6) Confidence intervals (often more informative than p-values)

A 95% confidence interval is approximately:

$$
\hat\beta_j \pm t_{0.975} \cdot \widehat{SE}(\hat\beta_j).
$$

Interpretation:
- it is a range of values consistent with the data under assumptions,
- it shows both sign and magnitude uncertainty.

If the 95% CI excludes 0, the two-sided p-value is typically < 0.05.

#### 7) Robust SE change p-values (without changing coefficients)

Different SE estimators correspond to different assumptions about errors:
- **Naive OLS SE:** homoskedastic, uncorrelated errors.
- **HC3:** heteroskedasticity-robust (cross-section).
- **HAC/Newey-West:** autocorrelation + heteroskedasticity (time series).
- **Clustered SE:** within-cluster correlated errors (panels/DiD).

**Key idea:** changing SE changes $\widehat{SE}(\hat\beta_j)$ which changes the t-stat and p-value, even when $\hat\beta_j$ is identical.

#### 8) Diagnostics: how hypothesis testing goes wrong (minimum set)

1) **Multiple testing**
- If you try many features/specs, some will "work" by chance.
- A few p-values < 0.05 are expected even if all true effects are 0.

2) **P-hacking / specification search**
- Tweaking the model until p-values look good invalidates the usual interpretation.

3) **Wrong SE (dependence)**
- Autocorrelation or clustering can make naive SE far too small.

4) **Confounding**
- A "significant" association is not a causal effect without identification.

Practical rule: interpret p-values as one piece of evidence, not a conclusion.

#### 9) Interpretation + reporting (how to write results responsibly)

Good reporting includes:
- effect size (coefficient) in meaningful units,
- uncertainty (CI preferred),
- correct SE choice for the data structure,
- a note about model limitations and identification.

**What this does NOT mean**
- "Significant" is not "important."
- "Not significant" is not "no effect" (could be low power).

#### 10) Small Python demo

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

rng = np.random.default_rng(0)

# 1) One-sample t-test
x = rng.normal(loc=0.2, scale=1.0, size=200)
t_stat, p_val = stats.ttest_1samp(x, popmean=0.0)
print('t-test t:', t_stat, 'p:', p_val)

# 2) Regression t-test
n = 300
x2 = rng.normal(size=n)
y = 1.0 + 0.5 * x2 + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({'y': y, 'x': x2})
X = sm.add_constant(df[['x']])
res = sm.OLS(df['y'], X).fit()
print(res.summary())
```

#### Exercises

- [ ] Take one regression output and rewrite it in words: coefficient, CI, and what assumptions the p-value relies on.
- [ ] Show how p-values change when you switch from naive SE to HC3 or HAC (same coefficient, different uncertainty).
- [ ] Create a multiple-testing demonstration: test 50 random predictors against random noise and count how many p-values < 0.05.
- [ ] Write 6 sentences explaining why "statistically significant" is not the same as "economically meaningful."

---

### HAC / Newey-West Standard Errors (Time-Series Inference)

HAC (heteroskedasticity-and-autocorrelation consistent) standard errors are the minimum correction for many time-series regressions.

#### 1) Intuition (plain English)

Time series residuals are rarely independent:
- shocks persist (serial correlation),
- variance changes across regimes (heteroskedasticity).

If you use naive OLS SE, you often understate uncertainty and overstate "significance."

#### 2) Notation + setup (define symbols)

Consider a time-series regression:

$$
y_t = x_t'\beta + u_t, \quad t = 1,\dots,T.
$$

Stacking:
$$
\mathbf{y} = \mathbf{X}\beta + \mathbf{u}.
$$

Classical OLS SE assume:
$$
\mathrm{Var}(\mathbf{u} \mid \mathbf{X}) = \sigma^2 I_T.
$$

But in time series, we often have:
- $\mathrm{Var}(u_t)$ changes over time (heteroskedasticity),
- $\mathrm{Cov}(u_t, u_{t-k}) \neq 0$ for some lags $k$ (autocorrelation).

#### 3) Estimation mechanics: coefficients vs uncertainty

OLS coefficients still equal:
$$
\hat\beta = (X'X)^{-1}X'y.
$$

HAC changes only the variance estimate:

$$
\widehat{\mathrm{Var}}_{HAC}(\hat\beta)
= (X'X)^{-1} \left(X'\hat\Omega X\right) (X'X)^{-1}.
$$

**What each term means**
- $\hat\Omega$ estimates the error covariance across time (including lagged autocovariances).
- The "sandwich" structure $(X'X)^{-1} (\cdot) (X'X)^{-1}$ replaces the classical $\sigma^2(X'X)^{-1}$.

#### 4) Newey-West: a common HAC choice

Newey-West constructs $\hat\Omega$ by combining residual autocovariances up to a maximum lag $L$:

$$
\hat\Omega_{NW} = \hat\Gamma_0 + \sum_{k=1}^{L} w_k (\hat\Gamma_k + \hat\Gamma_k'),
\quad w_k = 1 - \frac{k}{L+1}.
$$

**What each term means**
- $\hat\Gamma_0$: contemporaneous covariance term (analogous to HC).
- $\hat\Gamma_k = \frac{1}{T}\sum_{t=k+1}^{T} \hat{u}_t \hat{u}_{t-k} x_t x_{t-k}'$: lag-$k$ autocovariance contribution.
- $w_k$: **Bartlett weights** -- linearly declining from 1 toward 0. These ensure $\hat\Omega_{NW}$ is positive semi-definite.
- $L$: maximum lag included (the tuning parameter).

#### 5) Choosing `maxlags` (why it is a sensitivity parameter)

There is no universally correct $L$. The choice involves a **bias-variance tradeoff**: more lags capture more of the autocorrelation structure (reducing bias in the SE estimate), but each additional lag adds estimation noise (increasing variance). Too few lags leave autocorrelation unaccounted for; too many lags produce an erratic, imprecise SE estimate.

**Rule-of-thumb (Newey-West):** A common automatic choice is:

$$
L = \lfloor 0.75 \cdot T^{1/3} \rfloor
$$

where $T$ is the sample size. For example, with $T = 200$ quarterly observations, this gives $L = \lfloor 0.75 \times 5.85 \rfloor = 4$. This is a starting point, not gospel.

Reasonable habits:
- quarterly data: try 1, 2, 4,
- monthly data: try 3, 6, 12,
- report sensitivity if inference changes.

If results flip sign/significance dramatically across plausible $L$, treat inference as fragile -- it likely means the autocorrelation structure is strong and sample size is too small for confident conclusions.

#### 6) Mapping to code (statsmodels)

In `statsmodels`:
- fit OLS normally,
- request HAC covariance:

```python
res_hac = res.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': 4})
```

Or equivalently at fit time:

```python
res_hac = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
```

#### 7) Diagnostics + robustness (minimum set)

1) **Residual autocorrelation**
- Inspect ACF/PACF or Durbin-Watson style diagnostics.

2) **HAC sensitivity**
- Try several `maxlags` values; report if conclusions change.

3) **Stationarity checks**
- Nonstationarity can create spurious inference even with HAC.

4) **Stability**
- Fit on subperiods or rolling windows; do coefficients drift?

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
- [ ] Write 5 sentences: "When HAC is appropriate" vs "when HAC is not enough."

---

### SE Comparison Table

Use this table to choose the right standard error for your data structure. The coefficient $\hat\beta$ is **identical** across all four; only the uncertainty estimate changes.

| SE Type | Handles | Best For | Code |
|---------|---------|----------|------|
| Classical (naive) | Nothing -- assumes homoskedastic, uncorrelated errors | Textbook exercises only | `sm.OLS(y, X).fit()` |
| HC3 (robust) | Heteroskedasticity | Cross-section data (wages, health expenditures) | `.fit(cov_type='HC3')` |
| HAC / Newey-West | Heteroskedasticity + autocorrelation | Time-series data (macro, finance) | `.fit(cov_type='HAC', cov_kwds={'maxlags': L})` |
| Clustered | Heteroskedasticity + within-group correlation | Panel data, grouped cross-sections, DiD | See [Guide 06_causal/01](../06_causal/01_diff_in_diff.md) |

**When in doubt:**
- Cross-section with unknown error structure: use HC3 (it is never worse than naive, and only slightly conservative).
- Time series: use HAC. Check sensitivity to `maxlags`.
- Panel / grouped data: use clustered SE at the level of the group (e.g., state, firm, individual).

---

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`) + multicollinearity (`vif_table`)
- `src/macro.py`: GDP + labels (`gdp_growth_*`, `technical_recession_label`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`)

### Common Mistakes
- Using naive SE on time-series data and treating the resulting p-values as reliable.
- Choosing `maxlags` to make results "look significant" rather than reporting sensitivity.
- Confusing "statistically significant" with "economically important" (large sample + tiny effect = small p-value).
- Treating p > 0.05 as proof of no effect (could be low power).
- Forgetting that HAC fixes inference only -- it does not fix endogeneity, nonstationarity, or model misspecification.

<a id="summary"></a>
## Summary + Suggested Readings

This guide covers the two core inference tools for time-series econometrics: hypothesis testing and HAC standard errors. You should now be able to:
- state what a p-value is and is not,
- construct and interpret confidence intervals,
- apply Newey-West HAC SE and understand the maxlags tradeoff,
- choose among classical, HC3, HAC, and clustered SE based on data structure, and
- recognize when hypothesis testing goes wrong (multiple testing, wrong SE, p-hacking).

**Companion guides:**
- [Guide 00](00_single_factor_regression_micro.md): OLS foundations (derivation, assumptions, diagnostics)
- [Guide 02](02_single_factor_regression_macro.md): Macro-specific regression issues (stationarity, spurious regression, structural breaks)

Suggested readings:
- Wooldridge: *Introductory Econometrics*, Ch. 4 (inference), Ch. 12 (serial correlation, HAC)
- Newey & West (1987): "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix" -- the original paper
- Angrist & Pischke: *Mostly Harmless Econometrics*, Ch. 8 (standard errors)
- statsmodels docs: `get_robustcov_results` and `cov_type` options
