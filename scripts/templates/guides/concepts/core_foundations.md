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
