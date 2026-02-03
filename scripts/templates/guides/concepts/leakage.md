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

#### Economics interpretation
A recession model with leakage will appear to "predict" recessions, but it is usually just reading signals from the future.
The goal is to predict with only information available at the time.
