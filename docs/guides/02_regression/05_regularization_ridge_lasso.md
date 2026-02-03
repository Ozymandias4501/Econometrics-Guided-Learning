# Guide: 05_regularization_ridge_lasso

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/05_regularization_ridge_lasso.ipynb`.

This regression module covers both prediction and inference, with a strong emphasis on interpretation.

### Key Terms (defined)
- **OLS (Ordinary Least Squares)**: chooses coefficients that minimize squared prediction errors.
- **Coefficient**: expected change in the target per unit change in a feature (holding others fixed).
- **Standard error (SE)**: uncertainty estimate for a coefficient.
- **p-value**: probability of observing an effect at least as extreme if the true effect were zero (under assumptions).
- **Confidence interval (CI)**: a range of plausible coefficient values under assumptions.
- **Heteroskedasticity**: non-constant error variance; common in cross-section.
- **Autocorrelation**: errors correlated over time; common in time series.
- **HAC/Newey-West**: robust SE for time-series autocorrelation/heteroskedasticity.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Build feature matrix
- Complete notebook section: Fit ridge/lasso
- Complete notebook section: Coefficient paths
- Fit at least one plain OLS model and one robust-SE variant (HC3 or HAC).
- Interpret coefficients in units (or standardized units) and explain what they do *not* mean.
- Run at least one diagnostic: residual plot, VIF table, or rolling coefficient stability plot.

### Alternative Example (Not the Notebook Solution)
```python
# Toy OLS with robust SE (not the notebook data):
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)
x = rng.normal(size=200)
y = 2.0 + 0.5*x + rng.normal(scale=1 + 0.5*np.abs(x), size=200)  # heteroskedastic errors
X = sm.add_constant(pd.DataFrame({'x': x}))
res = sm.OLS(y, X).fit()
res_hc3 = res.get_robustcov_results(cov_type='HC3')
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: Regularization (Ridge vs Lasso)

Regularization adds a penalty to the loss function to reduce overfitting and stabilize coefficients.
In macro data with correlated indicators, it is often essential.

#### Key Terms (defined)
- **Regularization**: adding a penalty term to discourage large coefficients.
- **L2 penalty (ridge)**: penalizes squared coefficients.
- **L1 penalty (lasso)**: penalizes absolute coefficients; can drive some to exactly 0.
- **Alpha (lambda)**: strength of the penalty (hyperparameter).
- **Coefficient path**: coefficients as a function of alpha.

#### Objectives (math)
- OLS: minimize `||y - Xβ||^2`
- Ridge: minimize `||y - Xβ||^2 + α * ||β||_2^2`
- Lasso: minimize `||y - Xβ||^2 + α * ||β||_1`

#### Why standardization matters
- Penalties depend on coefficient magnitudes.
- If features are on different scales (percent vs index points), the penalty is uneven.
- Standardize (`StandardScaler`) before ridge/lasso.

#### Ridge vs lasso when predictors are correlated
- Ridge tends to shrink correlated predictors together (grouping effect).
- Lasso often picks one feature from a correlated group (can be unstable across samples).

#### Python demo: coefficient instability vs stabilization
```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(0)
n = 300
x1 = rng.normal(size=n)
x2 = x1 * 0.98 + rng.normal(scale=0.2, size=n)  # highly correlated
y = 1.0 + 2.0*x1 + rng.normal(scale=1.0, size=n)
X = np.column_stack([x1, x2])

ols = LinearRegression().fit(X, y)
ridge = Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=5.0))]).fit(X, y)
lasso = Pipeline([('sc', StandardScaler()), ('m', Lasso(alpha=0.1, max_iter=10000))]).fit(X, y)

print('ols  coef:', ols.coef_)
print('ridge coef:', ridge.named_steps['m'].coef_)
print('lasso coef:', lasso.named_steps['m'].coef_)
```

#### Interpretation warning
- Regularized coefficients are biased by design.
- They can be excellent for prediction, but do not treat them as classical OLS inference objects.
- Prefer out-of-sample evaluation and stability checks.


### OLS Objective
- Model: `y = Xβ + ε`
- OLS chooses `β` to minimize `Σ (y_i - ŷ_i)^2`.

### Interpreting Coefficients
- In a simple regression with one feature, the slope is the expected change in `y` for a +1 change in `x`.
- In a multi-factor regression, the slope is the expected change in `y` for a +1 change in `x_j` *holding other X fixed*.
- If features are correlated (multicollinearity), "holding others fixed" can be a fragile, unrealistic counterfactual.

### Inference vs Prediction
- Inference: emphasize coefficient uncertainty and assumptions.
- Prediction: emphasize out-of-sample performance.
- You can have strong prediction with weak/unstable coefficients.

### Robust Standard Errors
- **HC3** addresses heteroskedasticity (common for cross-sectional county data).
- **HAC/Newey-West** addresses autocorrelation + heteroskedasticity (common for quarterly macro time series).


### Common Mistakes
- Interpreting a coefficient as causal without a causal design.
- Ignoring multicollinearity (high VIF) and over-trusting coefficient signs.
- Using naive SE on time series and over-trusting p-values.

<a id="summary"></a>
## Summary + Suggested Readings

Regression is the core bridge between statistics and ML. You should now be able to:
- fit interpretable linear models,
- quantify uncertainty (robust SE), and
- diagnose when coefficients are unstable.


Suggested readings:
- Wooldridge: Introductory Econometrics (OLS, robust SE, interpretation)
- Angrist & Pischke: Mostly Harmless Econometrics (causal thinking)
- statsmodels docs: robust covariance (HCx, HAC)
