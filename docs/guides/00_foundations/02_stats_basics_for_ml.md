# Guide: 02_stats_basics_for_ml

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_foundations/02_stats_basics_for_ml.ipynb`.

This foundations module builds core intuition you will reuse in every later notebook.

### Key Terms (defined)
- **Time series**: data indexed by time; ordering is meaningful and must be respected.
- **Leakage**: using future information in features/labels, producing unrealistically good results.
- **Train/test split**: separating data for model fitting vs evaluation.
- **Multicollinearity**: predictors are highly correlated; coefficients can become unstable.


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Correlation vs causation
- Complete notebook section: Multicollinearity (VIF)
- Complete notebook section: Bias/variance
- Complete notebook section: Hypothesis testing
- Run the bootstrap cell and confirm `PROJECT_ROOT` points to the repo root.
- Complete all TODOs (no `...` left).
- Write a short paragraph explaining a leakage example you created.

### Alternative Example (Not the Notebook Solution)
```python
# Toy leakage example (not the notebook data):
import numpy as np
import pandas as pd

idx = pd.date_range('2020-01-01', periods=200, freq='D')
y = pd.Series(np.sin(np.linspace(0, 12, len(idx))) + 0.1*np.random.randn(len(idx)), index=idx)

# Correct feature: yesterday
x_lag1 = y.shift(1)

# Leakage feature: tomorrow (do NOT do this)
x_leak = y.shift(-1)
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

This notebook introduces the core statistical vocabulary used throughout the project.

### Core Foundations: Time, Evaluation, and Leakage (the rules you never stop using)

This project treats “foundations” as more than warm-up material. They are the rules that keep every later result honest.

#### 1) Intuition (plain English): what problem are we solving?

Most mistakes in applied econometrics + ML come from confusing these three questions:

1) **What did we know at the time of prediction/decision?** (timing)
2) **What are we trying to predict/estimate?** (target/estimand)
3) **How do we know the result would hold in the future or in another sample?** (evaluation/generalization)

**Story example:** You build a recession probability model using macro indicators.
- If you accidentally include information from the future (even indirectly), the model will look amazing on paper and fail in reality.
- If you evaluate with random splits, you are testing a different problem (IID classification) than the one you actually face (time-ordered forecasting).

#### 2) Notation + setup (define symbols)

Time index:
- $t = 1,\\dots,T$ indexes time (months/quarters).

Forecast horizon:
- $h \\ge 1$ is how far ahead you predict.

Features and target:
- $X_t$ is the feature vector available at time $t$.
- $y_{t+h}$ is the future value you want to predict (or a label defined using future data).

The forecasting problem is:

$$
\\text{learn a function } f \\text{ such that } \\hat y_{t+h} = f(X_t).
$$

**What each term means**
- $X_t$: information available at time $t$ (must be “past-only”).
- $y_{t+h}$: the thing you want to know in the future.
- $f$: your model (linear regression, logistic regression, random forest, …).

#### 3) Assumptions (and why time breaks ML defaults)

Many ML defaults assume **IID** data: independent and identically distributed samples.
Time series often violate both:
- observations are correlated over time,
- the data-generating process can drift (regime changes, structural breaks).

Practical implications:
- random train/test splits are usually invalid for forecasting,
- you must use time-aware splits and leakage checks.

#### 4) Estimation mechanics: evaluation is part of the method

When you evaluate a model, you are estimating its future performance.
If the evaluation scheme does not match the real timing of the task, the estimate is biased.

**Time-aware evaluation patterns**
- **Holdout (time split):** train on early period, test on later period.
- **Walk-forward / rolling origin:** re-train as time advances and evaluate sequentially.

If you use a walk-forward scheme, conceptually you are estimating:

$$
\\text{future error} \\approx \\frac{1}{M} \\sum_{m=1}^{M} \\ell(\\hat y_{t_m+h}, y_{t_m+h})
$$

where:
- $\\ell$ is a loss function (squared error, log loss, …),
- $t_m$ are evaluation times in the future relative to training.

#### 5) Leakage (the #1 silent killer)

> **Definition:** **Leakage** happens when features contain information that would not have been available at the time of prediction.

Leakage examples:
- using $y_{t+h}$ (or a function of it) in $X_t$,
- using “centered” rolling windows that include future values,
- merging datasets with mismatched timestamps so the future leaks into the past,
- standardizing using full-sample mean/variance before splitting.

**Why leakage is so dangerous**
- it makes models look much better than they are,
- it leads to confident but wrong decisions,
- it is often subtle and not caught by unit tests.

#### 6) Diagnostics + robustness (minimum set)

1) **Timing statement (write it down)**
- “At time $t$ we know $X_t$ and we predict $y_{t+h}$.”
- If you cannot say this clearly, leakage risk is high.

2) **Index sanity**
- check index type (DatetimeIndex), sorting, monotonicity.

3) **Shift sanity**
- after creating lags/targets, confirm the direction:
  - lags use `.shift(+k)` (past),
  - targets for forecasting are often `.shift(-h)` (future).

4) **Train/test boundary check**
- print the last train date and first test date; confirm no overlap.

#### 7) Interpretation + reporting

When you report results in this repo, always include:
- the prediction horizon $h$,
- the split scheme (time split or walk-forward),
- at least one metric appropriate for the task,
- a leakage check (what you did to prevent it).

**What this does NOT mean**
- A high in-sample $R^2$ is not evidence of real forecasting power.
- Random-split accuracy is not forecasting accuracy.

<details>
<summary>Optional: why time series “generalization” is harder</summary>

In forecasting, you train on one historical regime and test on another.
If the economy changes (policy regime, technology, measurement), relationships can shift.
That is why stability checks and walk-forward evaluation are emphasized in this project.

</details>

#### Exercises

- [ ] Write the timing statement for one notebook: “At time $t$ I know __ and predict __ at $t+h$.”
- [ ] Create one intentional leakage feature (future shift) and show how it inflates test performance.
- [ ] Compare random-split vs time-split evaluation on the same dataset; explain the difference.
- [ ] List 3 places leakage can enter during feature engineering (lags, rolling windows, scaling, joins).

### Deep Dive: Correlation vs causation — the question you must not confuse

Economics is full of correlated variables. Causal inference is about deciding when a relationship is more than correlation.

#### 1) Intuition (plain English)

Correlation answers:
- “Do $X$ and $Y$ move together?”

Causation answers:
- “If I intervene and change $X$, does $Y$ change?”

**Story example (macro):** Interest rates and inflation are correlated.  
That does not imply raising rates increases inflation; the correlation can reflect policy responses to expected inflation.

#### 2) Notation + setup (define symbols)

Correlation between random variables $X$ and $Y$:

$$
\\rho_{XY} = \\frac{\\mathrm{Cov}(X,Y)}{\\sqrt{\\mathrm{Var}(X)\\mathrm{Var}(Y)}}.
$$

**What each term means**
- $\\mathrm{Cov}(X,Y)$: whether $X$ and $Y$ co-move.
- correlation is unit-free and lies in $[-1,1]$.

Correlation is symmetric:
$$
\\rho_{XY} = \\rho_{YX}.
$$
But causal effects are directional.

#### 3) Assumptions (what you need for a causal claim)

A causal claim requires an identification strategy:
- randomization,
- a natural experiment,
- a credible quasi-experimental design (DiD/IV/RD),
- or a structural model with defensible assumptions.

Without identification, regression coefficients are best interpreted as conditional associations.

#### 4) Estimation mechanics: how confounding creates correlation without causation

Consider a simple confounding structure:
- $Z$ causes both $X$ and $Y$.

One possible DGP:
$$
X = aZ + \\eta, \\qquad Y = bZ + \\varepsilon.
$$

Even if $X$ does not cause $Y$, $X$ and $Y$ will be correlated because they share the common cause $Z$.

Regression “controls” can help if you measure the confounder, but:
- you rarely observe all confounders,
- controlling for the wrong variables (colliders/mediators) can introduce bias.

#### 5) Inference: significance is not causality

A small p-value means “incompatible with $\\beta=0$ under the model assumptions.”
It does not mean:
- the model is correct,
- the effect is causal,
- the effect is economically large.

#### 6) Diagnostics + robustness (minimum set)

1) **Timing sanity**
- can $X$ plausibly affect $Y$ given publication/decision timing?

2) **Confounder list**
- write down plausible common causes of $X$ and $Y$ (before running regressions).

3) **Placebos**
- test outcomes that should not respond to $X$; “effects” there suggest confounding.

4) **Design upgrade**
- if you need causality, move from “controls” to FE/DiD/IV where appropriate.

#### 7) Interpretation + reporting

Be explicit about the claim type:
- predictive association vs causal effect.

**What this does NOT mean**
- “Controlling for some variables” is not a guarantee of causality.

#### Exercises

- [ ] Write one example where correlation is expected but causality is ambiguous (macro or micro).
- [ ] Draw a simple confounding story in words (Z→X and Z→Y).
- [ ] Simulate confounding and show correlation without causation.
- [ ] Rewrite a regression interpretation paragraph to remove causal language unless justified.

### Deep Dive: Bias–variance tradeoff and overfitting (why train ≠ test)

Understanding overfitting is essential for both ML and econometrics—especially in small-sample macro settings.

#### 1) Intuition (plain English)

Models can fail in two ways:
- **too simple:** cannot capture real patterns (high bias),
- **too flexible:** fits noise that does not repeat (high variance).

Overfitting is when performance looks great on training data but poor on new data.

#### 2) Notation + setup (define symbols)

Let:
- true outcome be $y = f(x) + \\varepsilon$,
- model prediction be $\\hat f(x)$,
- loss be squared error.

For a fixed $x$, the expected prediction error decomposes as:

$$
\\mathbb{E}[(\\hat f(x) - y)^2]
= \\underbrace{(\\mathbb{E}[\\hat f(x)] - f(x))^2}_{\\text{bias}^2}
\\; + \\;
\\underbrace{\\mathbb{E}[(\\hat f(x) - \\mathbb{E}[\\hat f(x)])^2]}_{\\text{variance}}
\\; + \\;
\\underbrace{\\mathrm{Var}(\\varepsilon)}_{\\text{noise}}.
$$

**What each term means**
- bias: systematic error from model misspecification/underfitting,
- variance: sensitivity to sample fluctuations (overfitting risk),
- noise: irreducible uncertainty in outcomes.

#### 3) Assumptions

This decomposition assumes:
- a stable data-generating process for the evaluation period,
- meaningful train/test separation (no leakage),
- loss function matches the task.

In time series, regime changes can dominate this story; walk-forward evaluation helps reveal it.

#### 4) Estimation mechanics: why complexity increases variance

More flexible models can fit training data better (lower bias) but often:
- increase variance,
- require more data to generalize,
- need regularization/constraints.

In regression, adding more correlated predictors can:
- increase coefficient variance,
- create unstable interpretations,
- improve in-sample fit without improving out-of-sample performance.

#### 5) Inference connection

Overfitting is not only an ML problem:
- specification search (trying many models) is a form of overfitting,
- p-values become misleading under heavy model selection.

#### 6) Diagnostics + robustness (minimum set)

1) **Train vs test gap**
- large gap suggests overfitting or leakage.

2) **Learning curves**
- performance as a function of sample size can reveal high variance.

3) **Cross-validation (time-aware)**
- use walk-forward folds for time series.

4) **Regularization sensitivity**
- ridge/lasso strength vs performance; look for a stable region.

#### 7) Interpretation + reporting

Report:
- out-of-sample metrics (not just in-sample),
- evaluation scheme (time split / walk-forward),
- and a simple overfitting check (train/test comparison).

#### Exercises

- [ ] Fit a simple model and a complex model; compare train vs test performance.
- [ ] Increase feature count and watch the train/test gap change.
- [ ] Plot a learning curve (even crude) by training on increasing time windows.
- [ ] Explain in 6 sentences how overfitting relates to specification search in econometrics.

### Deep Dive: Multicollinearity and VIF — why coefficients become unstable

Multicollinearity is common in economic data and is one of the main reasons coefficient interpretation becomes fragile.

#### 1) Intuition (plain English)

If two predictors contain almost the same information, the regression struggles to decide “which variable deserves the credit.”

**Story example (macro):**
- many indicators co-move with the business cycle,
- a multifactor regression may assign unstable signs to “similar” indicators depending on sample period.

Prediction may still be fine, but coefficient stories become unreliable.

#### 2) Notation + setup (define symbols)

Regression in matrix form:

$$
\\mathbf{y} = \\mathbf{X}\\beta + \\varepsilon.
$$

OLS estimator:
$$
\\hat\\beta = (X'X)^{-1}X'y.
$$

Under classical assumptions:
$$
\\mathrm{Var}(\\hat\\beta \\mid X) = \\sigma^2 (X'X)^{-1}.
$$

**What each term means**
- When columns of $X$ are highly correlated, $X'X$ is close to singular.
- Then $(X'X)^{-1}$ has large entries → coefficient variance inflates.

#### 3) What multicollinearity does (and does not do)

- Often does **not** hurt prediction much.
- **Does** inflate standard errors (coefficients become noisy).
- **Does** make coefficients sensitive to small data changes (unstable signs/magnitudes).
- **Does not** automatically bias coefficients if exogeneity holds; it mostly increases variance.

#### 4) VIF (Variance Inflation Factor): what it measures

To compute VIF for feature $j$:
- regress $x_j$ on all other predictors,
- record the $R_j^2$ from that auxiliary regression.

Then:

$$
\\mathrm{VIF}_j = \\frac{1}{1 - R_j^2}.
$$

**Interpretation**
- If $R_j^2$ is near 1, $x_j$ is almost perfectly explained by other predictors.
- Then $\\mathrm{VIF}_j$ is large → coefficient uncertainty for $\\beta_j$ is inflated.

Rules of thumb (not laws):
- VIF > 5 suggests notable collinearity.
- VIF > 10 suggests serious collinearity.

#### 5) Estimation mechanics: “holding others fixed” becomes unrealistic

Coefficient interpretation relies on the counterfactual:
- “Increase $x_j$ by 1 while holding other predictors fixed.”

If predictors are tightly linked economically, that counterfactual can be meaningless (you cannot change one indicator while freezing another).

So multicollinearity is both:
- a statistical issue (variance inflation),
- and an economic interpretation issue (counterfactuals).

#### 6) Diagnostics + robustness (minimum set)

1) **Correlation matrix**
- identify groups of highly correlated features.

2) **VIF table**
- quantify redundancy; large VIF → unstable coefficient.

3) **Coefficient stability**
- fit the regression on different subperiods or bootstrap samples; do signs flip?

4) **Condition number**
- large condition number of $X$ suggests numerical instability.

#### 7) What to do about multicollinearity

Options (choose based on goals):

- **If you care about interpretation**
  - drop redundant variables,
  - combine variables into an index,
  - use domain-driven composites.

- **If you care about prediction**
  - use regularization (ridge/lasso),
  - use dimension reduction (PCA/factors),
  - use nonlinear models (trees) with care about leakage and evaluation.

#### 8) Interpretation + reporting

When multicollinearity is present, report:
- VIF or correlation evidence,
- coefficient instability (if observed),
- and avoid strong stories about individual coefficients.

#### Exercises

- [ ] Construct two highly correlated predictors and show VIF > 10.
- [ ] Fit OLS with both predictors; observe coefficient instability vs the true DGP.
- [ ] Drop one predictor and compare interpretability and fit.
- [ ] Fit ridge regression and compare coefficient stability to OLS.

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

### Project Code Map
- `scripts/scaffold_curriculum.py`: how this curriculum is generated (for curiosity)
- `src/evaluation.py`: time splits and metrics used later
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Using `train_test_split(shuffle=True)` on time-indexed data.
- Looking at the test set repeatedly while tuning ("test leakage").
- Assuming a significant p-value implies causation.
- Running many tests/specs and treating a small p-value as proof (multiple testing / p-hacking).

<a id="summary"></a>
## Summary + Suggested Readings

You now have the tooling to avoid the two most common beginner mistakes in economic ML:
1) leaking future information, and
2) over-interpreting correlated coefficients.


Suggested readings:
- Hyndman & Athanasopoulos: Forecasting: Principles and Practice (time series basics)
- Wooldridge: Introductory Econometrics (interpretation + pitfalls)
- scikit-learn docs: model evaluation and cross-validation
