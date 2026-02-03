### Deep Dive: Robust Standard Errors (HC3) for Cross-Sectional Data

Cross-sectional data often has heteroskedasticity.

> **Definition:** **Heteroskedasticity** means the variance of errors changes with predictors.

If you ignore it, OLS coefficients can be the same, but standard errors can be wrong.

#### What changes and what does not
- Coefficients $\hat\beta$ do not change.
- Standard errors, t-stats, p-values, and confidence intervals change.

#### Python demo: heteroskedastic errors and robust SE (commented)
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)

n = 400
x = rng.normal(size=n)

# Error variance increases with |x|
eps = rng.normal(scale=1 + 2*np.abs(x), size=n)
y = 1.0 + 0.5*x + eps

X = sm.add_constant(pd.DataFrame({'x': x}))
res = sm.OLS(y, X).fit()
res_hc3 = res.get_robustcov_results(cov_type='HC3')

print('naive SE:', res.bse)
print('HC3 SE  :', res_hc3.bse)
```

#### Interpretation warning
Robust SE improves uncertainty estimates under heteroskedasticity.
It does not fix confounding or make a coefficient causal.
