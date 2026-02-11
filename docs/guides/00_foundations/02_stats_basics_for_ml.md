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
# Toy confounding example (not the notebook data):
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500

# Z is a common cause (confounder)
z = rng.normal(size=n)
x = 0.8 * z + rng.normal(scale=0.5, size=n)  # Z -> X
y = 1.2 * z + rng.normal(scale=0.5, size=n)  # Z -> Y (X does NOT cause Y)

# Naive correlation is strong, but it's not causal
print(f"Correlation(x, y): {np.corrcoef(x, y)[0, 1]:.3f}")

# Controlling for Z removes the spurious association
import statsmodels.api as sm
X_naive = sm.add_constant(pd.DataFrame({'x': x}))
X_ctrl  = sm.add_constant(pd.DataFrame({'x': x, 'z': z}))
print("Naive coef on x:", sm.OLS(y, X_naive).fit().params['x'].round(3))
print("Controlled coef on x:", sm.OLS(y, X_ctrl).fit().params['x'].round(3))
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
- $t = 1,\dots,T$ indexes time (months/quarters).

Forecast horizon:
- $h \ge 1$ is how far ahead you predict.

Features and target:
- $X_t$ is the feature vector available at time $t$.
- $y_{t+h}$ is the future value you want to predict (or a label defined using future data).

The forecasting problem is:

$$
\text{learn a function } f \text{ such that } \hat y_{t+h} = f(X_t).
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
\text{future error} \approx \frac{1}{M} \sum_{m=1}^{M} \ell(\hat y_{t_m+h}, y_{t_m+h})
$$

where:
- $\ell$ is a loss function (squared error, log loss, …),
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
\rho_{XY} = \frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X)\mathrm{Var}(Y)}}.
$$

**What each term means**
- $\mathrm{Cov}(X,Y)$: whether $X$ and $Y$ co-move.
- correlation is unit-free and lies in $[-1,1]$.

Correlation is symmetric:
$$
\rho_{XY} = \rho_{YX}.
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
X = aZ + \eta, \qquad Y = bZ + \varepsilon.
$$

Even if $X$ does not cause $Y$, $X$ and $Y$ will be correlated because they share the common cause $Z$.

Regression “controls” can help if you measure the confounder, but:
- you rarely observe all confounders,
- controlling for the wrong variables (colliders/mediators) can introduce bias.

#### 5) Inference: significance is not causality

A small p-value means “incompatible with $\beta=0$ under the model assumptions.”
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
- true outcome be $y = f(x) + \varepsilon$,
- model prediction be $\hat f(x)$,
- loss be squared error.

For a fixed $x$, the expected prediction error decomposes as:

$$
\mathbb{E}[(\hat f(x) - y)^2]
= \underbrace{(\mathbb{E}[\hat f(x)] - f(x))^2}_{\text{bias}^2}
\; + \;
\underbrace{\mathbb{E}[(\hat f(x) - \mathbb{E}[\hat f(x)])^2]}_{\text{variance}}
\; + \;
\underbrace{\mathrm{Var}(\varepsilon)}_{\text{noise}}.
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

### Multicollinearity and VIF — preview

When predictors are highly correlated, regression coefficients become unstable. The key diagnostic is the **Variance Inflation Factor (VIF)**:

$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^2}
$$

where $R_j^2$ comes from regressing feature $j$ on all other features. VIF > 5 suggests notable collinearity; VIF > 10 suggests serious collinearity.

Key points:
- Multicollinearity inflates SE but does **not** bias coefficients (if exogeneity holds).
- It makes "holding others fixed" interpretations unrealistic when predictors move together.
- Solutions: drop redundant variables, use regularization (ridge/lasso), or PCA.

> **Full treatment**: See the [regression guide: Multicollinearity and VIF](../02_regression/00_single_factor_regression_micro.md#deep-dive-multicollinearity-and-vif--why-coefficients-become-unstable) for the complete derivation, diagnostics, and exercises.

### Hypothesis Testing — preview

Hypothesis tests appear in every regression output. The essentials:

- **p-value**: probability of data at least as extreme as observed, **under the null and model assumptions**. It is not the probability the null is true.
- **t-statistic**: $t_j = \hat\beta_j / \widehat{SE}(\hat\beta_j)$ — how many SEs the coefficient is from zero.
- **Confidence interval**: $\hat\beta_j \pm t_{0.975} \cdot \widehat{SE}(\hat\beta_j)$ — shows sign and magnitude uncertainty together. Often more informative than p-values.
- Changing SE estimator (naive → HC3 → HAC → clustered) changes the p-value **without** changing the coefficient.

Common pitfalls:
- Multiple testing / spec search inflates false positives.
- "Significant" is not "important" and "not significant" is not "no effect."
- A small p-value is not causal evidence without an identification strategy.

> **Full treatment**: See the [regression guide: Hypothesis Testing](../02_regression/00_single_factor_regression_micro.md#deep-dive-hypothesis-testing--how-to-read-p-values-without-fooling-yourself) for the complete derivation, demos, and exercises.

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
