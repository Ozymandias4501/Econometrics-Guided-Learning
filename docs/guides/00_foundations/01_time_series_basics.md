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

### Deep Dive: Train/Test Splits for Time Series

**Train/test split** means you fit your model on one subset (train) and evaluate on a later, untouched subset (test).
In time series, the split must respect chronology.

**Random split vs time split**
- Random split: mixes past and future in both train and test.
- Time split: train is earlier, test is later.

**Why time split matters**
- The future can look statistically different (regimes).
- The model must operate in real time: train on the past, predict the future.

**Python demo: random vs time split**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
idx = pd.date_range('2010-01-01', periods=400, freq='D')

# A simple AR(1)-like process
y = np.zeros(len(idx))
for t in range(1, len(y)):
    y[t] = 0.9*y[t-1] + rng.normal(scale=1.0)
s = pd.Series(y, index=idx)

df = pd.DataFrame({'y': s, 'y_lag1': s.shift(1)}).dropna()
X = df[['y_lag1']].to_numpy()
y_arr = df['y'].to_numpy()

# Random split
X_tr, X_te, y_tr, y_te = train_test_split(X, y_arr, test_size=0.2, shuffle=True, random_state=0)
m = LinearRegression().fit(X_tr, y_tr)
rmse_rand = mean_squared_error(y_te, m.predict(X_te), squared=False)

# Time split
split = int(len(df) * 0.8)
X_tr2, X_te2 = X[:split], X[split:]
y_tr2, y_te2 = y_arr[:split], y_arr[split:]
m2 = LinearRegression().fit(X_tr2, y_tr2)
rmse_time = mean_squared_error(y_te2, m2.predict(X_te2), squared=False)

print('RMSE random:', rmse_rand)
print('RMSE time  :', rmse_time)
```

**Walk-forward validation (preview)**
- Instead of one split, you evaluate across multiple chronological folds.
- This reveals stability: the model can look good in one era and fail in another.


### Deep Dive: Leakage (What It Is, How It Happens, How To Detect It)

**Leakage** means your model (or your features) accidentally uses information that would not be available at prediction time.
In time series, leakage is especially easy because the index *looks* like just another column, but it encodes causality constraints.

**Common leakage types**
- **Target leakage**: a feature is derived from the target (directly or indirectly).
- **Temporal leakage**: a feature uses future values (wrong shift direction, centered rolling windows, etc.).
- **Split leakage**: random splits mix future and past, letting the model learn patterns it wouldn't have in production.
- **Preprocessing leakage**: scaling/imputation fitted on all data (train + test) instead of train only.

**How to spot leakage (symptoms)**
- Test metrics that look "too good to be true" for the problem.
- A single feature dominates and seems to predict perfectly.
- Performance collapses when you switch from random split to time split.

**Debug checklist (time-series)**
1. For each feature, ask: *would I know this value at time t when making a prediction for t+1?*
2. Verify every shift direction: `shift(+k)` uses the past; `shift(-k)` leaks future.
3. Verify rolling window alignment: `rolling(..., center=True)` leaks future.
4. Ensure preprocessing (scalers, imputers) are fitted on training data only.

**Python demo: the classic `shift(-1)` leak**
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.default_rng(0)
idx = pd.date_range('2020-01-01', periods=200, freq='D')
y = pd.Series(np.cumsum(rng.normal(size=len(idx))), index=idx)

# Predict tomorrow (t+1)
target = y.shift(-1)

# Legit feature: yesterday (t-1)
x_lag1 = y.shift(1)

# LEAK feature: tomorrow (t+1)
x_leak = y.shift(-1)

df = pd.DataFrame({'target': target, 'x_lag1': x_lag1, 'x_leak': x_leak}).dropna()

X_ok = df[['x_lag1']].to_numpy()
X_leak = df[['x_leak']].to_numpy()
y_arr = df['target'].to_numpy()

# Time split
split = int(len(df) * 0.8)
X_ok_tr, X_ok_te = X_ok[:split], X_ok[split:]
X_leak_tr, X_leak_te = X_leak[:split], X_leak[split:]
y_tr, y_te = y_arr[:split], y_arr[split:]

m_ok = LinearRegression().fit(X_ok_tr, y_tr)
m_leak = LinearRegression().fit(X_leak_tr, y_tr)

print('R2 legit:', r2_score(y_te, m_ok.predict(X_ok_te)))
print('R2 leak :', r2_score(y_te, m_leak.predict(X_leak_te)))
```

**Python demo: rolling-window leakage (centered windows)**
```python
import pandas as pd

# This uses future values because the window is centered.
feature_leaky = y.rolling(window=7, center=True).mean()
```

**Practical interpretation (economics)**
- A recession model with leakage will appear to "predict" recessions, but it is usually just reading signals from the future.
- The real goal is to predict with only information available *at the time*.


### Common Mistakes
- Using `train_test_split(shuffle=True)` on time-indexed data.
- Looking at the test set repeatedly while tuning ("test leakage").
- Assuming a significant p-value implies causation.

<a id="summary"></a>
## Summary + Suggested Readings

You now have the tooling to avoid the two most common beginner mistakes in economic ML:
1) leaking future information, and
2) over-interpreting correlated coefficients.


Suggested readings:
- Hyndman & Athanasopoulos: Forecasting: Principles and Practice (time series basics)
- Wooldridge: Introductory Econometrics (interpretation + pitfalls)
- scikit-learn docs: model evaluation and cross-validation
