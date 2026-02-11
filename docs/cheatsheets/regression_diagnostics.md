# Cheatsheet: Regression Diagnostics

## Post-Estimation Checklist

After fitting any regression, work through these diagnostics before interpreting coefficients or reporting results:

```
1. Residual plots       → Does the model capture the relationship, or is there structure left over?
2. Heteroskedasticity   → Is the error variance constant? If not, SE and p-values are wrong.
3. Serial correlation   → (Time series only) Are errors correlated over time?
4. Multicollinearity    → Are predictors too correlated to isolate individual effects?
5. Influential points   → Are a few observations driving the entire result?
6. Functional form      → Should you use logs, polynomials, or interactions?
7. Normality            → (Small samples only) Are residuals approximately normal?
```

## Residual Diagnostics

Residuals ($\hat\varepsilon_i = y_i - \hat y_i$) are the leftover variation your model didn't explain. If the model is well-specified, residuals should look like random noise — no patterns, no structure.

| Plot | What you're looking for | What a problem looks like | What it means |
|---|---|---|---|
| **Residuals vs. fitted values** | Random scatter centered on zero, constant spread | U-shape or curve: the relationship isn't linear. Fan or funnel shape: variance isn't constant | Curve → add polynomial terms or log-transform. Funnel → heteroskedasticity, use robust SE |
| **Residuals vs. each predictor** | Random scatter | Systematic pattern (curve, clusters, fan) | The functional form for this predictor may be wrong |
| **Q-Q plot** | Points lying on the 45-degree diagonal line | S-curve (heavy tails), departure at extremes | Non-normal residuals. For large samples, this is usually fine (CLT). For small samples, it affects CI coverage |
| **ACF/PACF of residuals** | All bars within the blue confidence bands (no significant spikes) | Significant spike at lag 1, 2, etc. | Serial correlation — errors carry information from one period to the next. Use HAC SE |

**Python**:
```python
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Residuals vs fitted
plt.scatter(res.fittedvalues, res.resid)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')

# Q-Q plot
sm.qqplot(res.resid, line='45')

# ACF/PACF (time series)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(res.resid, lags=20)
plot_pacf(res.resid, lags=20)
```

## Heteroskedasticity

**What it is**: The variance of the error term changes across observations. In cross-section data, this often means that larger or wealthier units have more variable outcomes. In a scatter plot, you see a "fan" or "funnel" shape — residuals spread out as $X$ increases.

**Concrete example**: Regressing household medical spending on income. Low-income households all spend roughly the same (limited budgets), but high-income households vary widely (some choose expensive care, others don't). The residual variance increases with income.

**What it does to your regression**:
- OLS coefficients ($\hat\beta$) are still unbiased and consistent — the point estimates are fine
- Standard errors are wrong. Classical SE assume constant variance; when variance is non-constant, SE are typically too small, which makes t-statistics too large and p-values too small. You reject $H_0$ more often than you should.

**Detection**:
- Visual: fan-shaped residual-vs-fitted plot
- Formal: Breusch-Pagan test — regresses squared residuals on the predictors. Significant result = heteroskedasticity detected
- Formal: White's test — more general but less powerful

```python
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_pval, _, _ = het_breuschpagan(res.resid, res.model.exog)
# bp_pval < 0.05 → evidence of heteroskedasticity
```

**Fix**: Use HC3 robust standard errors for cross-section data. These adjust the SE formula to account for non-constant variance without changing the coefficient estimates.

```python
from src.econometrics import fit_ols_hc3
res_robust = fit_ols_hc3(df, y_col='y', x_cols=['x1', 'x2'])
```

## Serial Correlation (Time Series)

**What it is**: Today's error is correlated with yesterday's error: $Corr(\varepsilon_t, \varepsilon_{t-k}) \neq 0$. In macro data, this is the norm, not the exception — a recession quarter is followed by more recession quarters, not by a random draw from the error distribution.

**Concrete example**: Regressing GDP growth on the yield spread. If GDP growth is below-trend this quarter (negative residual), it's likely to be below-trend next quarter too. The residuals cluster in runs of positive and negative values rather than bouncing randomly.

**What it does to your regression**:
- OLS coefficients are still unbiased (if the model is otherwise correctly specified)
- Standard errors are too small, often dramatically so. With positive autocorrelation (the most common case), naive SE understate the true uncertainty because the "effective" sample size is smaller than the actual $n$

**Detection**:
- Visual: ACF plot of residuals showing significant spikes beyond lag 0
- Durbin-Watson statistic: $DW \approx 2$ means no autocorrelation; $DW \ll 2$ (near 0) means strong positive autocorrelation; $DW \gg 2$ (near 4) means negative autocorrelation
- Ljung-Box test: formal test for significant autocorrelation up to $k$ lags

```python
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(res.resid)  # should be near 2

from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(res.resid, lags=[4, 8, 12])
# significant p-values → serial correlation present
```

**Fix**: HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors, also called Newey-West SE. These allow for both heteroskedasticity and autocorrelation up to `maxlags` lags.

```python
from src.econometrics import fit_ols_hac
res_hac = fit_ols_hac(df, y_col='y', x_cols=['spread'], maxlags=4)
```

**Choosing maxlags**: Use the Newey-West rule of thumb: $L = \lfloor 0.75 \cdot T^{1/3} \rfloor$. For $T = 200$, $L \approx 4$. Too few lags = remaining bias in SE; too many lags = noisy SE estimates.

## Multicollinearity

**What it is**: Two or more predictors are highly correlated with each other. Regression tries to isolate each predictor's independent contribution — "what happens when $X_1$ changes but $X_2$ stays the same?" If $X_1$ and $X_2$ always move together, this counterfactual has almost no empirical support, and the estimates become unstable.

**Concrete example**: Including both "number of hospital beds" and "number of nurses" in a regression predicting patient outcomes. Larger hospitals have more of both, so the variables are highly correlated. Regression can estimate the *combined* effect well but struggles to separate the individual contributions. Adding or removing a few observations can cause the coefficients to swing wildly.

**What it does to your regression**:
- OLS coefficients are still unbiased (multicollinearity is not a bias problem)
- Standard errors are inflated — coefficients are estimated imprecisely
- Individual t-tests may show "not significant" even when the predictors jointly matter (high F-stat, low individual t-stats)
- Small changes in the data can cause large changes in coefficient estimates

**Detection**: Variance Inflation Factor (VIF)

$$
VIF_j = \frac{1}{1 - R^2_j}
$$

$R^2_j$ is the R-squared from regressing predictor $j$ on all other predictors. VIF measures how much the variance of $\hat\beta_j$ is inflated due to correlation with other predictors.

| VIF | Interpretation | Action |
|---|---|---|
| 1 | No correlation with other predictors | No issue |
| 1-5 | Moderate correlation | Usually acceptable |
| 5-10 | High correlation | Investigate; consider dropping or combining |
| > 10 | Severe (>90% of variance explained by other predictors) | Definitely address: drop one variable, combine via PCA, or use Ridge |

```python
from src.econometrics import vif_table
vif_table(df, ['beds', 'nurses', 'income', 'population'])
```

**Fixes**:
- Drop one of the correlated predictors (the one with weaker theoretical motivation)
- Combine correlated predictors (e.g., beds-per-nurse ratio instead of both separately)
- Use PCA to extract orthogonal factors from correlated inputs
- Use Ridge regression, which handles multicollinearity by shrinking correlated coefficients toward each other

## Influential Observations

A single outlier can pull the entire regression line. Diagnostics help you find these points and decide what to do about them.

| Measure | What it identifies | How to interpret | Rule of thumb |
|---|---|---|---|
| **Leverage** ($h_{ii}$) | Observations with unusual $X$ values (far from the center of the data) | High leverage means the observation *could* be influential — it's in a position to pull the line. But it might still be on the line | $h_{ii} > \frac{2k}{n}$ is high leverage |
| **Studentized residuals** | Observations with large residuals relative to their expected variability | Large studentized residual = the observation doesn't fit the pattern. Combined with high leverage, this is influential | $|t_i| > 3$ is an outlier |
| **Cook's distance** | Combines leverage and residual size into a single influence measure. Asks: "how much would all the coefficients change if I deleted this observation?" | High Cook's D means the observation has outsized influence on $\hat\beta$ | $D_i > \frac{4}{n}$ warrants investigation |

**Key distinction**: High leverage $\neq$ influential. A point can be far from the center of $X$ but perfectly on the regression line (high leverage, low residual, not influential). Influence requires *both* unusual position and deviation from the fitted line.

**What to do about influential points**:
1. Investigate: is it a data error (typo, coding mistake)? If so, fix the data
2. Is it a legitimate but unusual observation? Understand why it's unusual
3. Report results with and without the influential point. If conclusions change, discuss why
4. Never silently delete observations just because they're influential — that introduces bias

```python
from statsmodels.stats.outliers_influence import OLSInfluence
influence = OLSInfluence(res)
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag
```

## $R^2$ and Adjusted $R^2$

| Metric | Formula | Key property |
|---|---|---|
| $R^2$ | $1 - \frac{SSR}{SST}$ | Fraction of variance in $Y$ explained by the model. Always increases (or stays the same) when you add predictors, even useless ones |
| Adjusted $R^2$ | $1 - \frac{SSR/(n-k)}{SST/(n-1)}$ | Penalizes each additional predictor. Can decrease if a new predictor doesn't improve fit enough to justify the added complexity |

**Common misconceptions**:
- High $R^2$ does not mean the model is correct or useful. A regression of one trending variable on another can have $R^2 > 0.95$ and be completely spurious.
- Low $R^2$ does not mean the model is useless. In noisy data (individual health outcomes, daily stock returns), $R^2$ of 0.05 can still represent an economically meaningful relationship.
- $R^2$ does not measure whether you've found a causal effect. A confounded regression can have a high $R^2$.

## Functional Form

Choosing the right functional form can matter as much as choosing the right variables.

| Specification | Equation | Coefficient interpretation | When to use |
|---|---|---|---|
| **Level-level** | $Y = \beta_0 + \beta_1 X$ | A 1-unit increase in $X$ is associated with a $\beta_1$-unit change in $Y$ | Default; use when both variables are in natural units |
| **Log-level** | $\ln Y = \beta_0 + \beta_1 X$ | A 1-unit increase in $X$ is associated with a $\beta_1 \times 100$% change in $Y$ | When $Y$ is right-skewed or has diminishing returns (e.g., income, spending) |
| **Level-log** | $Y = \beta_0 + \beta_1 \ln X$ | A 1% increase in $X$ is associated with a $\beta_1 / 100$-unit change in $Y$ | When the effect of $X$ diminishes at higher levels |
| **Log-log** | $\ln Y = \beta_0 + \beta_1 \ln X$ | A 1% increase in $X$ is associated with a $\beta_1$% change in $Y$ (elasticity) | When you want to estimate an elasticity; multiplicative relationships |

**How to diagnose**: If the residual-vs-fitted plot shows a curve, the current functional form is wrong. Try logging the dependent variable (for right-skew) or adding polynomial terms (for nonlinearity in $X$).

## Quick Reference: What Breaks What

| Problem | $\hat\beta$ biased? | SE wrong? | What breaks | Fix |
|---|---|---|---|---|
| **Heteroskedasticity** | No | Yes — usually too small | p-values too small, false significance | HC3 robust SE |
| **Serial correlation** | No | Yes — usually too small | p-values too small, false significance | HAC / Newey-West SE |
| **Multicollinearity** | No | Yes — too large | Individual coefficients imprecise, unstable | Drop/combine predictors, Ridge |
| **Omitted variable (correlated with $X$)** | **Yes** | Yes | Everything — coefficient, SE, and inference all wrong | Add the variable, use IV, panel FE, or DiD |
| **Non-stationarity** | **Yes** (spurious) | Yes | $R^2$ and t-stats are meaningless | First-difference, log-difference, or cointegration |
| **Measurement error in $X$** | **Yes** (attenuation bias — coefficient biased toward zero) | Yes | Underestimates the true effect | Use IV with a correctly measured instrument |
| **Influential outliers** | Possibly | Possibly | Results driven by a handful of observations | Investigate, report with/without, consider robust regression |
| **Wrong functional form** | **Yes** | Yes | Coefficient doesn't correspond to the true relationship | Try logs, polynomials, or interactions based on residual plots |
