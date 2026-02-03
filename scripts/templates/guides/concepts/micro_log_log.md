### Deep Dive: Log-Log Regression (Elasticity-Style Interpretation)

In microeconomic cross-sectional data, log transforms are common because many variables (income, population, rent) are heavy-tailed.

#### Key terms (defined)
> **Definition:** **Cross-sectional data** observes many units (counties) at one time.

> **Definition:** A **log transform** uses $\log(x)$ to compress scale and turn multiplicative relationships into additive ones.

#### Log-log model and interpretation
A log-log regression is:

$$
\log(y) = \alpha + \beta \log(x) + \varepsilon
$$

Interpretation (rule of thumb):
- a 1% increase in $x$ is associated with about a $\beta$% increase in $y$.

Why? Because for small changes:

$$
\Delta \log(x) \approx \frac{\Delta x}{x}
$$

#### Pitfall: zeros and missing values
- $\log(0)$ is undefined.
- Filter out non-positive values or use a different transform (`log1p`) if justified.

#### Python demo: simulated log-log relationship (commented)
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)

n = 800
x = np.exp(rng.normal(size=n))

# y grows with x^0.3 (multiplicative)
y = 2.0 * (x ** 0.3) * np.exp(rng.normal(scale=0.2, size=n))

df = pd.DataFrame({'x': x, 'y': y})
df['lx'] = np.log(df['x'])
df['ly'] = np.log(df['y'])

X = sm.add_constant(df[['lx']])
res = sm.OLS(df['ly'], X).fit()
print(res.params)
```
