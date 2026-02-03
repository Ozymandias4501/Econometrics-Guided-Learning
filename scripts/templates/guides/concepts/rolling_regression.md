### Deep Dive: Rolling Regressions (Stability and Structural Breaks)

A rolling regression re-fits a model on a moving window of past data.

#### Key terms (defined)
> **Definition:** A **rolling window** is a fixed-size window that moves forward through time.

> **Definition:** A **structural break** is when the relationship between X and Y changes.

#### Why rolling regressions matter in macro
Relationships can change across eras.
A single coefficient can hide that instability.

#### Python demo: coefficient changes over time (commented)
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)

n = 200
x = rng.normal(size=n)

# Coefficient changes halfway
beta = np.r_[np.repeat(0.2, n//2), np.repeat(-0.2, n - n//2)]
y = 1.0 + beta * x + rng.normal(scale=1.0, size=n)

idx = pd.date_range('1970-03-31', periods=n, freq='QE')
df = pd.DataFrame({'y': y, 'x': x}, index=idx)

window = 60
betas = []
dates = []

for end in range(window, len(df)+1):
    sub = df.iloc[end-window:end]
    res = sm.OLS(sub['y'], sm.add_constant(sub[['x']])).fit()
    betas.append(res.params['x'])
    dates.append(sub.index[-1])

beta_series = pd.Series(betas, index=dates)
print(beta_series.head())
```

#### Interpretation
If coefficients drift:
- the model is not describing a single stable mechanism
- for prediction, you may want to weight recent history more
- for inference, be cautious about a single "effect" claim
