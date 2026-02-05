## Primer: Panel + IV regression with `linearmodels` (FE, clustered SE, 2SLS)

This repo uses:
- `statsmodels` for classic OLS inference patterns, and
- `linearmodels` for **panel fixed effects** and **instrumental variables** (IV/2SLS).

The goal of this primer is to make you productive quickly (with the *minimum* theory needed to use the tools correctly). Deep math lives in the guides.

### Why `linearmodels`?

`linearmodels` provides clean APIs for:
- `PanelOLS`: fixed effects / TWFE
- `IV2SLS`: two-stage least squares

and it handles some panel-specific details (like absorbing FE) more naturally than `statsmodels`.

### Panel data shape (the #1 requirement)

Most panel estimators expect a **MultiIndex**:
- level 0: entity (e.g., county `fips`)
- level 1: time (e.g., `year`)

```python
# df has columns: fips, year, y, x1, x2, state, ...
df = df.copy()
df["fips"] = df["fips"].astype(str)
df["year"] = df["year"].astype(int)
df = df.set_index(["fips", "year"]).sort_index()
```

**Expected output / sanity check**
- `df.index.nlevels == 2`
- `df.index.is_monotonic_increasing` is `True`
- no duplicate index pairs: `df.index.duplicated().any()` is `False`

### TWFE model (PanelOLS)

Econometric form:

$$
Y_{it} = X_{it}'\\beta + \\alpha_i + \\gamma_t + \\varepsilon_{it}
$$

In code:

```python
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

y = df["y"].astype(float)
X = df[["x1", "x2"]].astype(float)
X = sm.add_constant(X, has_constant="add")

res = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(cov_type="robust")
print(res.summary)
```

### Clustered SE (common in applied panel/DiD work)

If errors are correlated within clusters (e.g., state-level shocks), use clustered SE:

```python
clusters = df["state"]  # must align row-for-row with y/X index

res_cl = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
  cov_type="clustered",
  clusters=clusters,
)
```

**Expected output / sanity check**
- clustered SE are often larger than robust SE (not guaranteed, but common)
- always report the number of clusters: `clusters.nunique()`

### IV / 2SLS (IV2SLS)

Structural equation (endogeneity motivation):
$$
Y = \\beta X + W'\\delta + u, \\quad \\mathrm{Cov}(X,u)\\neq 0
$$

In code (one endogenous regressor):

```python
from linearmodels.iv import IV2SLS
import statsmodels.api as sm

y = df["y"].astype(float)
endog = df[["x_endog"]].astype(float)
exog = sm.add_constant(df[["x_exog1", "x_exog2"]].astype(float), has_constant="add")
instr = df[["z1", "z2"]].astype(float)

res_iv = IV2SLS(y, exog, endog, instr).fit(cov_type="robust")
print(res_iv.summary)
```

**Expected output / sanity check**
- `res_iv.params` contains coefficients for exog + endogenous variables
- `res_iv.first_stage` (if printed) shows instrument relevance diagnostics

### Common pitfalls (and quick fixes)

- **MultiIndex mismatch:** if `clusters` is not aligned to the same index as `y/X`, you’ll get errors or wrong results.
  - Fix: construct clusters from the same `df` after indexing/sorting.
- **Non-numeric dtypes:** strings in `X` silently break models.
  - Fix: `astype(float)` on model columns.
- **Missing data:** panels often have missing rows after merges/transforms.
  - Fix: build a modeling table with `.dropna()` for required columns.
- **Too few clusters:** cluster-robust inference is fragile with very small cluster counts.
  - Fix: treat p-values as fragile; report cluster count; consider alternative designs.
