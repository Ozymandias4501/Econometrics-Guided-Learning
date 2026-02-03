## Primer: statsmodels vs scikit-learn (Inference vs Prediction)

### Two different goals
- **Prediction (ML):** optimize out-of-sample accuracy.
- **Inference (econometrics):** interpret coefficients + quantify uncertainty.

scikit-learn is mostly prediction-focused.
statsmodels is built for inference (standard errors, p-values, confidence intervals).

### Minimal statsmodels OLS pattern
```python
import statsmodels.api as sm

# X: DataFrame of features
# y: Series target

# Add intercept column
# Xc = sm.add_constant(X, has_constant='add')

# Fit OLS
# res = sm.OLS(y, Xc).fit()
# print(res.summary())
```

### What you're looking at in `res.summary()`
Typical regression output includes:
- **coef**: estimated coefficients $\hat\beta$
- **std err**: estimated standard errors $\widehat{SE}(\hat\beta)$
- **t**: t-statistic $\hat\beta / \widehat{SE}(\hat\beta)$
- **P>|t|**: p-value for $H_0: \beta=0$ (under assumptions)
- **[0.025, 0.975]**: 95% confidence interval for $\beta$

You can access these programmatically:

```python
# res.params        # coefficients (pandas Series)
# res.bse           # standard errors
# res.pvalues       # p-values
# res.conf_int()    # confidence intervals
```

### Robust standard errors
Robust SE change uncertainty estimates, not coefficients.

```python
# HC3 (cross-section)
# res_hc3 = res.get_robustcov_results(cov_type='HC3')

# HAC/Newey-West (time series)
# res_hac = res.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': 4})
```

Why this matters:
- Cross-sectional data often has **heteroskedasticity** (error variance changes across observations).
- Time series often has **autocorrelation** (errors correlated over time).

### Practical rule
- Use sklearn for predictive pipelines and cross-validation.
- Use statsmodels when you need inference and careful interpretation.
