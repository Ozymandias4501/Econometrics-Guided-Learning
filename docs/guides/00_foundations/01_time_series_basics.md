# Guide: 01_time_series_basics

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_foundations/01_time_series_basics.ipynb`.

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
- Complete notebook section: Toy series
- Complete notebook section: Resampling
- Complete notebook section: Lag and rolling features
- Complete notebook section: Leakage demo
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

This notebook introduces the two most important ideas for economic ML:
1) time-aware evaluation, and
2) leakage prevention.

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

### Deep Dive: Time splits — evaluation that matches forecasting reality

Time-aware splitting is not optional in forecasting tasks; it defines what “generalization” means.

#### 1) Intuition (plain English)

If you predict the future, you must train on the past and test on the future.
Random splits answer a different question: “Can I interpolate within a mixed pool of time periods?”

**Story example:** If you train on 2008 and test on 2006 (random split), you are letting crisis-era patterns help predict pre-crisis data—an unrealistic advantage.

#### 2) Notation + setup (define symbols)

Let:
- data be ordered by time $t=1,\\dots,T$,
- training window be $t \\le t_{train}$,
- test window be $t > t_{train}$.

A basic time split is:
- Train: $\\{1,\\dots,t_{train}\\}$
- Test: $\\{t_{train}+1,\\dots,T\\}$

#### 3) Assumptions (what time splits assume)

Time splits assume:
- you can use historical data to learn relationships relevant for the future,
- the feature/label timing is correctly defined (no leakage),
- you accept that regimes can change (so performance can vary).

#### 4) Estimation mechanics: why random splits overestimate performance

Random splits mix early and late periods in both train and test.
That creates two problems:
- **information leakage via time correlation** (nearby periods are similar),
- **regime mixing** (train sees future regimes).

So the test metric can be biased upward relative to true forecasting performance.

#### 5) Inference: splits affect uncertainty

Even if you do inference (p-values), time dependence matters:
- serial correlation inflates effective sample size if ignored,
- time splits help reveal whether relationships are stable across eras.

#### 6) Diagnostics + robustness (minimum set)

1) **Report split dates**
- always print the last train date and first test date.

2) **Try multiple cut points**
- if performance depends heavily on one boundary, results are unstable.

3) **Plot train vs test distributions**
- shifts in feature distributions indicate regime drift.

4) **Compare to walk-forward**
- walk-forward validation often gives a more realistic error estimate.

#### 7) Interpretation + reporting

Report:
- split scheme (single holdout vs multiple folds),
- dates and horizon,
- metrics on the test period (and ideally multiple periods).

**What this does NOT mean**
- One lucky split is not proof of generalization.

#### Exercises

- [ ] Evaluate the same model with a random split and a time split; compare and explain the gap.
- [ ] Move the split boundary forward/backward by a few years and report stability.
- [ ] Plot feature distributions in train vs test; identify at least one shifted feature.

### Deep Dive: Leakage — what it is, how it happens, how to detect it

Leakage is the fastest way to get “amazing” results that do not survive reality.

#### 1) Intuition (plain English)

If the model sees information that would not have been available at the time of prediction, it is not learning; it is cheating.

**Story example:** You “predict” next quarter’s recession using an indicator that is published with a delay, but you accidentally align it as if it were known in real time. Your backtest looks great, then fails when you try to use it live.

#### 2) Notation + setup (define symbols)

Let:
- $t$ be the time you make a prediction,
- $h$ be the forecast horizon,
- $X_t$ be features available at time $t$,
- $y_{t+h}$ be the target you want to predict.

The core question for every feature is:

$$
\\text{Was this feature value knowable at time } t \\text{ when predicting } y_{t+h}?
$$

If “no,” it is leakage.

#### 3) Common leakage types (defined)

> **Target leakage:** a feature directly/indirectly encodes the target (or future information about it).

> **Temporal leakage:** a feature uses future values (wrong shift direction, centered rolling windows, forward-filled joins).

> **Split leakage:** your train/test strategy allows future information into training (random splits for forecasting).

> **Preprocessing leakage:** preprocessing is fit on the full dataset (test set influences scaling/imputation).

#### 4) Estimation mechanics: how leakage inflates performance

Leakage typically:
- increases in-sample fit,
- increases test fit *under the wrong evaluation scheme*,
- collapses under true time-ordered evaluation or live deployment.

The reason is simple: the model has access to information correlated with the future target that would not exist at prediction time.

#### 5) Inference: leakage also breaks “statistical significance”

If leakage is present:
- coefficients, p-values, and CI are not meaningful,
- you are no longer analyzing the intended prediction problem.

So leakage is a first-order validity issue, not a minor bug.

#### 6) Practical code patterns (and anti-patterns)

**(a) The classic `shift(-1)` bug**

```python
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.default_rng(0)
idx = pd.date_range('2020-01-01', periods=200, freq='D')
s = pd.Series(np.cumsum(rng.normal(size=len(idx))), index=idx, name='y')

# Goal: predict tomorrow (t+1)
target = s.shift(-1)

# Legit: yesterday (t-1)
x_lag1 = s.shift(1)

# LEAK: tomorrow (t+1) — this equals the target
x_leak = s.shift(-1)

df = pd.DataFrame({'target': target, 'x_lag1': x_lag1, 'x_leak': x_leak}).dropna()

# Time split
split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

m_ok = LinearRegression().fit(train[['x_lag1']], train['target'])
m_leak = LinearRegression().fit(train[['x_leak']], train['target'])

print('R2 legit:', r2_score(test['target'], m_ok.predict(test[['x_lag1']])))
print('R2 leak :', r2_score(test['target'], m_leak.predict(test[['x_leak']])))
```

**(b) Rolling-window leakage**

Centered rolling windows use future values:

```python
feature_leaky = s.rolling(window=7, center=True).mean()   # BAD for forecasting
feature_ok = s.rolling(window=7, center=False).mean()     # past-only
```

**(c) Preprocessing leakage**

Fit scalers/imputers on training only (use pipelines):

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
  ("scaler", StandardScaler()),
  ("clf", LogisticRegression(max_iter=5000)),
])
```

#### 7) Diagnostics + robustness (minimum set)

1) **Random split vs time split gap**
- if random split performance is much higher than time split, suspect leakage or regime drift.

2) **Feature audit**
- audit every `.shift()` direction and every rolling window.

3) **Timestamp audit**
- inspect the meaning of timestamps after merges (month-end vs quarter-end, publication lags).

4) **“Too good to be true” check**
- if one feature predicts nearly perfectly, investigate it as a leak candidate.

#### 8) Interpretation + reporting

When you present results, state:
- the forecast horizon,
- the evaluation scheme (time split / walk-forward),
- at least one concrete leakage prevention step you took.

**What this does NOT mean**
- A model that looks great with leakage is not “close”; it is solving a different problem.

#### Exercises

- [ ] Create an intentional leakage feature and show how it inflates performance under a random split.
- [ ] Fix the leakage and re-evaluate with a time split; write 5 sentences explaining what changed.
- [ ] List 5 places leakage can enter (shifts, rolls, merges, scaling, target construction).
- [ ] For one notebook dataset, manually verify (by printing rows) that features are past-only relative to the label.

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
