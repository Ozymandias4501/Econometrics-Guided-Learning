# Guide: 04_inference_time_series_hac

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/04_inference_time_series_hac.ipynb`.

This regression module covers both prediction and inference, with a strong emphasis on interpretation.

### Key Terms (defined)
- **OLS (Ordinary Least Squares)**: chooses coefficients that minimize squared prediction errors.
- **Coefficient**: expected change in the target per unit change in a feature (holding others fixed).
- **Standard error (SE)**: uncertainty estimate for a coefficient.
- **p-value**: probability of observing an effect at least as extreme if the true effect were zero (under assumptions).
- **Confidence interval (CI)**: a range of plausible coefficient values under assumptions.
- **Heteroskedasticity**: non-constant error variance; common in cross-section.
- **Autocorrelation**: errors correlated over time; common in time series.
- **HAC/Newey-West**: robust SE for time-series autocorrelation/heteroskedasticity.


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Assumptions
- Complete notebook section: Autocorrelation
- Complete notebook section: HAC SE
- Fit at least one plain OLS model and one robust-SE variant (HC3 or HAC).
- Interpret coefficients in units (or standardized units) and explain what they do *not* mean.
- Run at least one diagnostic: residual plot, VIF table, or rolling coefficient stability plot.

### Alternative Example (Not the Notebook Solution)
```python
# Toy OLS with robust SE (not the notebook data):
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)
x = rng.normal(size=200)
y = 2.0 + 0.5*x + rng.normal(scale=1 + 0.5*np.abs(x), size=200)  # heteroskedastic errors
X = sm.add_constant(pd.DataFrame({'x': x}))
res = sm.OLS(y, X).fit()
res_hc3 = res.get_robustcov_results(cov_type='HC3')
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Core Regression: mechanics, assumptions, and interpretation (OLS as the baseline)

Linear regression is the baseline model for both econometrics and ML. Even when you use nonlinear models, the regression mindset (assumptions → estimation → inference → diagnostics) remains essential.

#### 1) Intuition (plain English)

Regression answers questions like:
- “How does $Y$ vary with $X$ on average?”
- “Holding other observed controls fixed, what is the association between one feature and the outcome?”

In economics we care about two different uses:
- **prediction:** does a model forecast well out-of-sample?
- **inference:** what is the estimated relationship and its uncertainty?

#### 2) Notation + setup (define symbols)

Scalar form (observation $i=1,\\dots,n$):

$$
y_i = \\beta_0 + \\beta_1 x_{i1} + \\cdots + \\beta_K x_{iK} + \\varepsilon_i.
$$

Matrix form:

$$
\\mathbf{y} = \\mathbf{X}\\beta + \\varepsilon.
$$

**What each term means**
- $\\mathbf{y}$: $n\\times 1$ vector of outcomes.
- $\\mathbf{X}$: $n\\times (K+1)$ design matrix (includes an intercept column).
- $\\beta$: $(K+1)\\times 1$ vector of coefficients.
- $\\varepsilon$: $n\\times 1$ vector of errors (unobserved determinants).

#### 3) Assumptions (what you need for unbiasedness and for inference)

For interpretation and inference, it helps to separate:

**(A) Assumptions for unbiased coefficients**

1) **Linearity in parameters**
- $y$ is linear in $\\beta$ (you can still include nonlinear transformations of $x$).

2) **No perfect multicollinearity**
- columns of $X$ are not perfectly linearly dependent.

3) **Exogeneity (key!)**
$$
\\mathbb{E}[\\varepsilon \\mid X] = 0.
$$

This rules out:
- omitted variable bias,
- reverse causality,
- many forms of measurement error problems.

**(B) Assumptions for classical standard errors**

4) **Homoskedasticity**
$$
\\mathrm{Var}(\\varepsilon \\mid X) = \\sigma^2 I.
$$

5) **No autocorrelation (time series)**
$$
\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_{t-k}) = 0 \\text{ for } k \\neq 0.
$$

When (4)–(5) fail, OLS coefficients can remain valid under (A), but naive SE are wrong → robust/HAC/clustered SE.

#### 4) Estimation mechanics: deriving OLS

OLS chooses coefficients to minimize the sum of squared residuals:

$$
\\hat\\beta = \\arg\\min_{\\beta} \\sum_{i=1}^{n} (y_i - x_i'\\beta)^2
= \\arg\\min_{\\beta} (\\mathbf{y} - \\mathbf{X}\\beta)'(\\mathbf{y} - \\mathbf{X}\\beta).
$$

Take derivatives (the “normal equations”):

$$
\\frac{\\partial}{\\partial \\beta} (\\mathbf{y}-\\mathbf{X}\\beta)'(\\mathbf{y}-\\mathbf{X}\\beta)
= -2\\mathbf{X}'(\\mathbf{y}-\\mathbf{X}\\beta) = 0.
$$

Solve:
$$
\\mathbf{X}'\\mathbf{X}\\hat\\beta = \\mathbf{X}'\\mathbf{y}
\\quad \\Rightarrow \\quad
\\hat\\beta = (\\mathbf{X}'\\mathbf{X})^{-1}\\mathbf{X}'\\mathbf{y}.
$$

**What each term means**
- $(X'X)^{-1}$ exists only if there is no perfect multicollinearity.
- OLS is a projection of $y$ onto the column space of $X$.

#### 5) Coefficient interpretation (and why “holding fixed” is tricky)

In the model, $\\beta_j$ means:

> the expected change in $y$ when $x_j$ increases by one unit, holding other regressors fixed (within the model).

In economics, “holding fixed” can be unrealistic if regressors move together (multicollinearity).
That is why:
- coefficient signs can flip,
- SE can inflate,
- interpretation must be cautious.

#### 6) Inference: standard errors, t-stats, confidence intervals

Under classical assumptions:

$$
\\mathrm{Var}(\\hat\\beta \\mid X) = \\sigma^2 (X'X)^{-1}.
$$

In practice we estimate $\\sigma^2$ and compute standard errors:
- $\\widehat{SE}(\\hat\\beta_j)$
- t-stat: $t_j = \\hat\\beta_j / \\widehat{SE}(\\hat\\beta_j)$
- 95% CI: $\\hat\\beta_j \\pm 1.96\\,\\widehat{SE}(\\hat\\beta_j)$ (approx.)

When assumptions fail, use robust SE:
- **HC3** for cross-section heteroskedasticity,
- **HAC/Newey–West** for time-series autocorrelation + heteroskedasticity,
- **clustered SE** for grouped dependence (panels/DiD).

#### 7) Diagnostics + robustness (minimum set)

1) **Residual checks**
- plot residuals vs fitted values; look for heteroskedasticity/nonlinearity.

2) **Multicollinearity**
- compute VIF; large VIF → unstable coefficients.

3) **Time-series dependence**
- check residual autocorrelation; use HAC when needed.

4) **Stability**
- rolling regressions or sub-sample splits; do coefficients drift?

#### 8) Interpretation + reporting

Always report:
- coefficient in units (or standardized units),
- robust SE appropriate to data structure,
- a short causal warning unless you have a causal design.

**What this does NOT mean**
- Regression does not “control away” all confounding automatically.
- A small p-value does not imply economic importance.
- A high $R^2$ does not imply good forecasting out-of-sample.

#### Exercises

- [ ] Derive the normal equations and explain each step in words.
- [ ] Fit OLS and HC3 (or HAC) and compare SE; explain why they differ.
- [ ] Create two correlated regressors and show how multicollinearity affects coefficient stability.
- [ ] Write a 6-sentence interpretation of one regression output, including what you can and cannot claim.

### Deep Dive: Hypothesis Testing — how to read p-values without fooling yourself

Hypothesis tests show up everywhere in econometrics output. The goal of this section is not to worship p-values, but to understand what they *are* and what they *are not*.

#### 1) Intuition (plain English)

A hypothesis test is a structured way to ask:
- “If the true effect were zero, how surprising is my estimate?”

It is **not** a direct answer to:
- “What is the probability the effect is real?”
- “Is my model correct?”

**Story example:** You regress unemployment on an interest-rate spread and get a small p-value.
That might mean:
- the relationship is real in-sample,
- or your SE are wrong (autocorrelation),
- or you tried many specs (multiple testing),
- or the effect is tiny but precisely estimated.

#### 2) Notation + setup (define symbols)

We usually test a claim about a population parameter $\\theta$ (mean, regression coefficient, difference in means, …).

Define:
- $H_0$: the **null hypothesis** (default claim),
- $H_1$: the **alternative hypothesis** (what you consider if evidence contradicts $H_0$),
- $T$: a **test statistic** computed from data,
- $\\alpha$: a pre-chosen significance level (e.g., 0.05).

Example in regression:
- $H_0: \\beta_j = 0$
- $H_1: \\beta_j \\neq 0$ (two-sided)

#### 3) Assumptions (why tests are conditional statements)

Every p-value is conditional on:
- the statistical model (e.g., OLS assumptions),
- the standard error estimator you use (naive vs robust vs HAC vs clustered),
- the sample and selection process.

If those assumptions fail, the p-value may be meaningless.

#### 4) Estimation mechanics in OLS: where t-stats come from

OLS estimates coefficients:

$$
\\hat\\beta = (X'X)^{-1}X'y.
$$

For coefficient $\\beta_j$, you compute an estimated standard error $\\widehat{SE}(\\hat\\beta_j)$.

The t-statistic for testing $H_0: \\beta_j = 0$ is:

$$
t_j = \\frac{\\hat\\beta_j - 0}{\\widehat{SE}(\\hat\\beta_j)}.
$$

**What each term means**
- numerator: your estimated effect.
- denominator: your uncertainty estimate.
- large |t| means “many standard errors away from 0.”

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
\\hat\\beta_j \\pm t_{0.975} \\cdot \\widehat{SE}(\\hat\\beta_j).
$$

Interpretation:
- it is a range of values consistent with the data under assumptions,
- it shows both sign and magnitude uncertainty.

If the 95% CI excludes 0, the two-sided p-value is typically < 0.05.

#### 7) Robust SE change p-values (without changing coefficients)

Different SE estimators correspond to different assumptions about errors:
- **Naive OLS SE:** homoskedastic, uncorrelated errors.
- **HC3:** heteroskedasticity-robust (cross-section).
- **HAC/Newey–West:** autocorrelation + heteroskedasticity (time series).
- **Clustered SE:** within-cluster correlated errors (panels/DiD).

**Key idea:** changing SE changes $\\widehat{SE}(\\hat\\beta_j)$ → changes t-stat and p-value, even when $\\hat\\beta_j$ is identical.

#### 8) Diagnostics: how hypothesis testing goes wrong (minimum set)

1) **Multiple testing**
- If you try many features/specs, some will “work” by chance.
- A few p-values < 0.05 are expected even if all true effects are 0.

2) **P-hacking / specification search**
- tweaking the model until p-values look good invalidates the usual interpretation.

3) **Wrong SE (dependence)**
- autocorrelation or clustering can make naive SE far too small.

4) **Confounding**
- a “significant” association is not a causal effect without identification.

Practical rule:
- interpret p-values as one piece of evidence, not a conclusion.

#### 9) Interpretation + reporting (how to write results responsibly)

Good reporting includes:
- effect size (coefficient) in meaningful units,
- uncertainty (CI preferred),
- correct SE choice for the data structure,
- a note about model limitations and identification.

**What this does NOT mean**
- “Significant” is not “important.”
- “Not significant” is not “no effect” (could be low power).

#### 10) Small Python demo (optional)

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
- [ ] Write 6 sentences explaining why “statistically significant” is not the same as “economically meaningful.”

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

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`) + multicollinearity (`vif_table`)
- `src/macro.py`: GDP + labels (`gdp_growth_*`, `technical_recession_label`)
- `src/evaluation.py`: regression metrics helpers
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Interpreting a coefficient as causal without a causal design.
- Ignoring multicollinearity (high VIF) and over-trusting coefficient signs.
- Using naive SE on time series and over-trusting p-values.

<a id="summary"></a>
## Summary + Suggested Readings

Regression is the core bridge between statistics and ML. You should now be able to:
- fit interpretable linear models,
- quantify uncertainty (robust SE), and
- diagnose when coefficients are unstable.


Suggested readings:
- Wooldridge: Introductory Econometrics (OLS, robust SE, interpretation)
- Angrist & Pischke: Mostly Harmless Econometrics (causal thinking)
- statsmodels docs: robust covariance (HCx, HAC)
