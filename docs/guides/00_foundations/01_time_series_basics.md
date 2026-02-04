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

### Deep Dive: Train/Test Splits for Time Series

> **Definition:** A **train/test split** separates data into (1) a training set used to fit the model and (2) a test set used only for evaluation.

> **Definition:** A **time split** is a train/test split that respects chronology: training uses earlier time periods, testing uses later time periods.

> **Definition:** A **random split** mixes past and future in both train and test. For forecasting problems, this usually creates overly optimistic results.

#### Why time splits matter (intuition)
Economic data is time-ordered. Many things can change over time:
- policy regimes
- measurement definitions and revisions
- structural breaks (relationships change)

A model that looks good under random splits can fail when deployed because deployment always looks like: train on the past, predict the future.

#### The forecasting question you are actually answering
If your target is at time $t+1$ and your features are at time $t$, the real question is:

$$
\text{How well can we predict } y_{t+1} \text{ using information available at time } t?
$$

A random split answers a different question: "How well can we interpolate across mixed time periods?" That is not what you want.

#### Python demo: random split vs time split (commented)
```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Make a toy time series where time ordering matters.
# We simulate something like an AR(1) process.

rng = np.random.default_rng(0)
idx = pd.date_range('2010-01-01', periods=400, freq='D')

y = np.zeros(len(idx))
for t in range(1, len(y)):
    y[t] = 0.9 * y[t-1] + rng.normal(scale=1.0)

s = pd.Series(y, index=idx, name='y')

# Build a "legit" lag feature: yesterday's value.
df = pd.DataFrame({'y': s, 'y_lag1': s.shift(1)}).dropna()
X = df[['y_lag1']].to_numpy()
y_arr = df['y'].to_numpy()

# 1) Random split (NOT recommended for time series)
X_tr, X_te, y_tr, y_te = train_test_split(X, y_arr, test_size=0.2, shuffle=True, random_state=0)
m = LinearRegression().fit(X_tr, y_tr)
rmse_rand = mean_squared_error(y_te, m.predict(X_te), squared=False)

# 2) Time split (recommended)
split = int(len(df) * 0.8)
X_tr2, X_te2 = X[:split], X[split:]
y_tr2, y_te2 = y_arr[:split], y_arr[split:]

m2 = LinearRegression().fit(X_tr2, y_tr2)
rmse_time = mean_squared_error(y_te2, m2.predict(X_te2), squared=False)

print('RMSE random:', rmse_rand)
print('RMSE time  :', rmse_time)
```

#### What to look for in results
- If random split is much better than time split, suspect that time structure matters (or that leakage exists elsewhere).
- If both are similar, the series may be close to stationary and the feature set may be simple.

#### Walk-forward validation (stronger than a single split)
> **Definition:** **Walk-forward validation** evaluates a model across multiple chronological folds. It answers: "Does this model work across multiple eras or only in one?"

Typical pattern:
- train on early period
- test on next block
- move forward and repeat

**Python demo: walk-forward split indices (project-adjacent)**
```python
from src.evaluation import walk_forward_splits

# n = number of time points (quarters or months)
n = 120

splits = list(walk_forward_splits(n, initial_train_size=60, test_size=12))
print('num folds:', len(splits))
print('first split:', splits[0])
print('last split :', splits[-1])
```

#### Debug checklist
1. Confirm your index is sorted and unique.
2. Confirm no feature uses future information (see leakage guide section).
3. Confirm your test period is strictly after your training period.
4. If tuning hyperparameters, do not tune on the final test period (use a validation scheme).

#### Project touchpoints (where this shows up in code)
- In notebooks: you should split time series chronologically before fitting any model.
- In code: `src/evaluation.py` includes `time_train_test_split_index` and `walk_forward_splits`.

#### Economics interpretation
A time split is not just a technical detail. It is the difference between:
- "This model can describe patterns in the full dataset"
- "This model can predict the future using only information available at the time"

### Deep Dive: Leakage (What It Is, How It Happens, How To Detect It)

> **Definition:** **Leakage** happens when your model uses information that would not be available at prediction time.

Leakage is one of the fastest ways to get "amazing" results that do not survive contact with reality.

#### Common leakage types (defined)
> **Definition:** **Target leakage** occurs when a feature is derived from the target (directly or indirectly).

> **Definition:** **Temporal leakage** occurs when a feature uses future values (wrong shift direction, centered rolling windows, etc.).

> **Definition:** **Split leakage** occurs when your split strategy allows future information into training (random splits for forecasting).

> **Definition:** **Preprocessing leakage** occurs when you fit preprocessing (scaling, imputation) on all data instead of training data only.

#### The core question to ask for every feature
For each feature $x_t$, ask:

$$
\text{Would I know this value at time } t \text{ when making a prediction for } t+1?
$$

If the answer is "no", it is leakage.

#### Python demo: the classic `shift(-1)` bug (commented)
```python
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.default_rng(0)
idx = pd.date_range('2020-01-01', periods=200, freq='D')

# Random-walk-like series
s = pd.Series(np.cumsum(rng.normal(size=len(idx))), index=idx, name='y')

# Goal: predict tomorrow (t+1)
target = s.shift(-1)

# Legit feature: yesterday (t-1)
x_lag1 = s.shift(1)

# LEAK feature: tomorrow (t+1) - this is literally the target!
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

#### Python demo: rolling-window leakage (centered windows)
> **Definition:** A **rolling window** summarizes the recent past (e.g., last 12 months). If you center the window, it includes future values.

```python
# Centered rolling windows leak future information.
# If you are predicting at time t, a centered window uses values after t.

feature_leaky = s.rolling(window=7, center=True).mean()

# Safer default: center=False (uses past values ending at t)
feature_ok = s.rolling(window=7, center=False).mean()
```

#### Python demo: preprocessing leakage (scalers)
If you standardize features using the whole dataset, the test set influences the mean/variance.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Imagine X_train and X_test are separated by time.
# WRONG: fit scaler on all data
# scaler = StandardScaler().fit(np.vstack([X_train, X_test]))

# RIGHT: fit scaler on training only
# scaler = StandardScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
```

#### Symptoms of leakage (how it looks)
- Test metrics that are "too good to be true" for the difficulty of the problem.
- Huge performance gap: random split looks great, time split collapses.
- A single feature appears to predict perfectly.

#### Debug checklist (practical)
1. Audit every `shift(...)` direction.
   - `shift(+k)` uses the past.
   - `shift(-k)` uses the future.
2. Audit rolling windows.
   - Avoid `center=True`.
3. Audit preprocessing.
   - Fit scalers/imputers on training only.
   - Use sklearn `Pipeline` to enforce this.
4. Validate timestamp meaning.
   - Are your features "known as of" the prediction date?

#### Project touchpoints (where leakage is prevented or easy to introduce)
- `src/features.py` explicitly forbids non-positive lags in `add_lag_features` to prevent accidental future leakage.
- `src/evaluation.py` contains time-aware split helpers (`time_train_test_split_index`, `walk_forward_splits`).
- Classification notebooks use sklearn `Pipeline` to prevent preprocessing leakage.

#### Economics interpretation
A recession model with leakage will appear to "predict" recessions, but it is usually just reading signals from the future.
The goal is to predict with only information available at the time.

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
