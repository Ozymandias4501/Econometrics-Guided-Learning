# Guide: 06_rolling_regressions_stability

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/06_rolling_regressions_stability.ipynb`.

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
- Complete notebook section: Rolling regression
- Complete notebook section: Coefficient drift
- Complete notebook section: Regime interpretation
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

### Deep Dive: Rolling regressions — coefficient stability over time

Rolling regressions are a simple tool to see whether relationships drift across macro regimes.

#### 1) Intuition (plain English)

In macro data, relationships change:
- policy regimes shift,
- financial structure changes,
- measurement changes.

A single “full-sample” regression can hide instability.
Rolling regressions estimate coefficients repeatedly on moving windows to reveal drift.

#### 2) Notation + setup (define symbols)

Let window length be $W$ (in periods).
For each end time $t \\ge W$, define the window:
$$
\\{t-W+1, \\dots, t\\}.
$$

Estimate:
$$
\\hat\\beta_t = \\arg\\min_{\\beta} \\sum_{s=t-W+1}^{t} (y_s - x_s'\\beta)^2.
$$

**What each term means**
- $\\hat\\beta_t$ is the coefficient estimate using only the most recent $W$ observations ending at $t$.

#### 3) Assumptions and caveats

Rolling regressions assume:
- the relationship is approximately stable within each window,
- $W$ is large enough for estimation but small enough to detect changes.

Inference caveat:
- consecutive windows overlap heavily, so estimates are correlated.

#### 4) Mechanics (practical use)

Typical workflow:
1) choose a window (e.g., 40 quarters ≈ 10 years),
2) estimate $\\hat\\beta_t$ for each $t$,
3) plot $\\hat\\beta_t$ with CI bands,
4) compare drift to recession shading or known events.

#### 5) Diagnostics + robustness (minimum set)

1) **Window sensitivity**
- try multiple $W$; does the drift pattern persist?

2) **Residual diagnostics**
- within each window, check autocorrelation; HAC may still be needed.

3) **Regime interpretation**
- do coefficient changes align with known macro events? (recessions, policy changes)

#### 6) Interpretation + reporting

Rolling coefficients suggest instability but do not identify why.
Report:
- window length,
- SE choice,
- and a narrative linking drift to plausible regime changes.

#### Exercises

- [ ] Run a rolling regression and plot the coefficient path with CI.
- [ ] Compare two window lengths and explain the bias–variance trade-off.
- [ ] Identify one period where the coefficient changes sign and propose a macro explanation.

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
