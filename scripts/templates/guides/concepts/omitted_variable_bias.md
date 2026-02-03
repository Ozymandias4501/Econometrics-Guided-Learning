### Deep Dive: Omitted Variable Bias (Why Adding Controls Changes Coefficients)

> **Definition:** **Omitted variable bias (OVB)** happens when a missing variable influences the outcome and is correlated with an included predictor.

If you omit that variable, your coefficient can absorb its effect.

#### Setup
Suppose the true model is:

$$
 y = \beta x + \gamma z + \varepsilon
$$

If you regress $y$ on $x$ but omit $z$, the estimated effect of $x$ can be biased if $x$ and $z$ are correlated.

#### Python demo: confounder makes x look important (commented)
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)

n = 2000

# z affects y, and z also affects x
z = rng.normal(size=n)
x = 0.8*z + rng.normal(scale=1.0, size=n)
y = 2.0*z + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({'y': y, 'x': x, 'z': z})

# Omitted z: biased coefficient on x
res_omit = sm.OLS(df['y'], sm.add_constant(df[['x']])).fit()

# Include z: coefficient on x shrinks
res_full = sm.OLS(df['y'], sm.add_constant(df[['x', 'z']])).fit()

print('omit z:', res_omit.params)
print('full  :', res_full.params)
```

#### Practical rule
If a coefficient flips sign or changes drastically when adding plausible controls,
be cautious about interpreting the original coefficient.
