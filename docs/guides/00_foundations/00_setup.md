# Guide: 00_setup

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_foundations/00_setup.ipynb`.

This foundations module builds core intuition you will reuse in every later notebook.

### Key Terms (defined)
- **Time series**: data indexed by time; ordering is meaningful and must be respected.
- **Leakage**: using future information in features/labels, producing unrealistically good results.
- **Train/test split**: separating data for model fitting vs evaluation.
- **Multicollinearity**: predictors are highly correlated; coefficients can become unstable.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Environment bootstrap
- Complete notebook section: Verify API keys
- Complete notebook section: Load sample data
- Complete notebook section: Checkpoints
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

### Core Foundations: The Ideas You Will Reuse Everywhere

This project is built around a simple principle:

> **Definition:** A good model result is one that would still hold up if you had to use the model in the real world.

To get there, you need correct evaluation and correct data timing.

#### Time series vs cross-sectional (defined)
> **Definition:** A **time series** is indexed by time; ordering matters.

> **Definition:** **Cross-sectional** data compares many units at one time; ordering is not temporal.

Many ML defaults assume IID (independent and identically distributed) samples. Time series data is rarely IID.

#### What "leakage" really means
Leakage is not just a bug. It is a violation of the prediction setting.
If you are predicting the future, your features must be available in the past.

#### What "generalization" means in time series
In forecasting, generalization means:
- you train on one historical period
- you perform well in a later period

That is much harder than random-split generalization because the data generating process can change.

#### A practical habit
For every dataset row at time t, write a one-line statement:
- "At time t, we know X, and we are trying to predict Y at time t+h."

If you cannot state that clearly, it is very easy to leak information.

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
