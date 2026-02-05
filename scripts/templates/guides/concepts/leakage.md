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
