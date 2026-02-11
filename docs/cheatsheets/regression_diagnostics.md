# Cheatsheet: Regression Diagnostics

## Post-Estimation Checklist

After fitting any regression, check these before interpreting results:

```
1. Residual plots → patterns indicate model misspecification
2. Heteroskedasticity → use robust SE if present
3. Serial correlation (time series) → use HAC SE if present
4. Multicollinearity → VIF check for multiple regression
5. Influential observations → leverage and Cook's distance
6. Normality of residuals → matters for small-sample inference
```

## Residual Diagnostics

| Plot | What to look for | Problem indicated |
|---|---|---|
| Residuals vs. fitted values | Random scatter around zero | Non-random pattern → misspecification |
| Residuals vs. each predictor | Random scatter | Fan shape → heteroskedasticity; curve → nonlinearity |
| Q-Q plot of residuals | Points on the 45-degree line | Deviations → non-normality |
| ACF/PACF of residuals | No significant spikes | Significant spikes → serial correlation |

## Heteroskedasticity

**What it is**: Error variance changes with $X$ (e.g., larger residuals for larger $X$).

**Why it matters**: OLS coefficients are still unbiased, but standard errors are wrong (usually too small), making t-tests unreliable.

**Detection**:
- Visual: fan-shaped residual plot
- Formal: Breusch-Pagan test (`statsmodels.stats.diagnostic.het_breuschpagan`)

**Fix**: Use robust standard errors (HC3 for cross-section, HAC for time series).

```python
# Cross-section
res_robust = res.get_robustcov_results(cov_type='HC3')

# Time series
res_hac = res.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': 4})
```

## Serial Correlation (Time Series)

**What it is**: $Corr(\varepsilon_t, \varepsilon_{t-k}) \neq 0$ — today's error is correlated with past errors.

**Why it matters**: Naive SE are too small, inflating t-stats and producing false significance.

**Detection**:
- Visual: ACF/PACF plot of residuals
- Formal: Durbin-Watson statistic (DW $\approx$ 2 means no autocorrelation; DW $\ll$ 2 means positive autocorrelation)
- Formal: Ljung-Box test

**Fix**: HAC / Newey-West standard errors with appropriate `maxlags`.

## Multicollinearity

**What it is**: Predictors are highly correlated with each other.

**Why it matters**: OLS coefficients are still unbiased, but standard errors blow up. Individual coefficients become unreliable even though the overall model fits well.

**Detection**: Variance Inflation Factor (VIF)

$$
VIF_j = \frac{1}{1 - R^2_j}
$$

where $R^2_j$ is from regressing predictor $j$ on all other predictors.

| VIF | Interpretation |
|---|---|
| 1 | No collinearity |
| 1-5 | Moderate (usually fine) |
| 5-10 | Concerning |
| > 10 | Severe ($> 90$% of variance explained by other predictors) |

```python
from src.econometrics import vif_table
vif_table(df, ['x1', 'x2', 'x3'])
```

**Fixes**: Drop one of the correlated predictors, combine them (PCA), or use regularization (Ridge).

## Influential Observations

| Measure | What it detects | Rule of thumb |
|---|---|---|
| **Leverage** ($h_{ii}$) | Points far from the mean of $X$ | $h_{ii} > 2k/n$ is high leverage |
| **Studentized residuals** | Points far from the regression line | $|t_i| > 3$ is an outlier |
| **Cook's distance** | Combined leverage + residual influence | $D_i > 4/n$ warrants investigation |

**High leverage $\neq$ influential**: A point can be far from the mean of $X$ but still on the regression line.

**Influential** = high leverage AND large residual. One influential point can substantially change $\hat\beta$.

## $R^2$ and Adjusted $R^2$

| Metric | Formula | Property |
|---|---|---|
| $R^2$ | $1 - \frac{SSR}{SST}$ | Always increases when you add predictors |
| Adjusted $R^2$ | $1 - \frac{SSR/(n-k)}{SST/(n-1)}$ | Penalizes for number of predictors $k$ |

**Common misconception**: High $R^2$ does not mean the model is "good." A regression of one trending variable on another can have $R^2 > 0.9$ and be completely spurious.

## Functional Form

| Symptom | Possible fix |
|---|---|
| Curved residual pattern | Add $X^2$ term or use log transform |
| Right-skewed dependent variable | Use $\ln(Y)$ |
| Multiplicative relationship | Log-log specification: $\ln Y = \beta_0 + \beta_1 \ln X$ |
| Diminishing returns | Log-level: $\ln Y = \beta_0 + \beta_1 X$ |

## Quick Reference: What Breaks What

| Problem | $\hat\beta$ biased? | SE wrong? | Fix |
|---|---|---|---|
| Heteroskedasticity | No | Yes (too small) | HC3 robust SE |
| Serial correlation | No | Yes (too small) | HAC / Newey-West SE |
| Multicollinearity | No | Yes (too large) | Drop/combine predictors, Ridge |
| Omitted variable (correlated) | **Yes** | Yes | Add the variable, use IV, or panel FE |
| Non-stationarity | **Yes** (spurious) | Yes | Difference or use cointegration |
| Measurement error in $X$ | **Yes** (attenuation) | Yes | Use IV |
| Outliers | Possibly | Possibly | Investigate, robust regression |
