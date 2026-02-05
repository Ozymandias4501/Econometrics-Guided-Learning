## Primer: Panel and IV regression with `linearmodels`

`statsmodels` is great for OLS inference, but panel and IV workflows are often cleaner with `linearmodels`.

This project uses `linearmodels` for:
- **PanelOLS** (fixed effects / TWFE)
- **IV2SLS** (instrumental variables)

### Panel data shape
Most panel estimators expect a **MultiIndex**:
- level 0: entity (e.g., county `fips`)
- level 1: time (e.g., `year`)

In pandas:
```python
# df is a DataFrame with columns fips, year, y, x1, x2
# df = df.set_index(['fips', 'year']).sort_index()
```

### Minimal PanelOLS (two-way fixed effects)
```python
from linearmodels.panel import PanelOLS

# y: Series with MultiIndex
# X: DataFrame with MultiIndex

# model: y_it = beta'X_it + alpha_i + gamma_t + eps_it
# res = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(cov_type='robust')
# print(res.summary)
```

### Clustered standard errors (common in applied work)
If errors are correlated within clusters (e.g., state), use clustered SE:
```python
# clusters must align with y/X index (same rows)
# res = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
#     cov_type='clustered',
#     clusters=df['state'],  # e.g., state-level clustering
# )
```

### Minimal IV2SLS (one endogenous regressor)
```python
from linearmodels.iv import IV2SLS
import statsmodels.api as sm

# y: Series
# endog: DataFrame with endogenous regressor(s)
# exog: DataFrame with controls (include a constant)
# instr: DataFrame with instruments

# exog = sm.add_constant(exog, has_constant='add')
# res = IV2SLS(y, exog, endog, instr).fit(cov_type='robust')
# print(res.summary)
```

### Practical rule
- If the goal is **causal identification**, always write down the assumptions first (parallel trends, exclusion restriction, etc.).
- Then treat the model output as conditional on those assumptions, not as “truth”.

