## Primer: `statsmodels` vs `scikit-learn` (inference vs prediction)

This repo uses both libraries because they serve different goals:

- **Prediction (ML):** optimize out-of-sample accuracy → `scikit-learn`
- **Inference (econometrics):** interpret coefficients + quantify uncertainty → `statsmodels`

### Minimal `statsmodels` OLS pattern

```python
import statsmodels.api as sm

# X: DataFrame of features, y: Series target
Xc = sm.add_constant(X, has_constant="add")  # add intercept
res = sm.OLS(y, Xc).fit()
print(res.summary())
```

**Expected output / sanity check**
- a table with `coef`, `std err`, `t`, `P>|t|`, and a CI column
- coefficient names match your column names

### What you are looking at in `res.summary()`

- **coef**: $\\hat\\beta$ (estimated effect in the model)
- **std err**: estimated uncertainty $\\widehat{SE}(\\hat\\beta)$
- **t**: $\\hat\\beta / \\widehat{SE}(\\hat\\beta)$
- **P>|t|**: p-value for $H_0: \\beta=0$ (conditional on assumptions)
- **[0.025, 0.975]**: 95% confidence interval

### Robust standard errors (change uncertainty, not coefficients)

```python
# Cross-section heteroskedasticity
res_hc3 = res.get_robustcov_results(cov_type="HC3")

# Time series autocorrelation + heteroskedasticity
res_hac = res.get_robustcov_results(cov_type="HAC", cov_kwds={"maxlags": 4})
```

### Common pitfalls (and quick fixes)

- **Forgetting the intercept**
  - Fix: always `add_constant`.
- **Wrong SE for time series**
  - Fix: use HAC when residuals are autocorrelated.
- **Treating p-values as causal proof**
  - Fix: write the identification assumption; otherwise interpret as association.
- **Mixing prediction and inference**
  - Fix: use `sklearn` pipelines + time splits for prediction; use `statsmodels` for coefficient uncertainty.
