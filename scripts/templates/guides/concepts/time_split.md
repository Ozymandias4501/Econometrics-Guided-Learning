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

#### Debug checklist
1. Confirm your index is sorted and unique.
2. Confirm no feature uses future information (see leakage guide section).
3. Confirm your test period is strictly after your training period.
4. If tuning hyperparameters, do not tune on the final test period (use a validation scheme).

#### Economics interpretation
A time split is not just a technical detail. It is the difference between:
- "This model can describe patterns in the full dataset"
- "This model can predict the future using only information available at the time"
