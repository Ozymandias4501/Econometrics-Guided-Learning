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

### Core Regression: Mechanics, Interpretation, and Uncertainty

Regression is used for both prediction and inference.

#### The model
We write a linear regression as:

$$
\mathbf{y} = \mathbf{X}\beta + \varepsilon
$$

- $\mathbf{y}$: outcomes
- $\mathbf{X}$: predictors (features)
- $\beta$: coefficients
- $\varepsilon$: error term (everything not modeled)

OLS estimates:

$$
\hat\beta = (X'X)^{-1}X'y
$$

#### Coefficient interpretation
> **Definition:** A **coefficient** $\beta_j$ is the expected change in $y$ for a one-unit change in $x_j$, holding other features fixed (within the model).

Key interpretation cautions:
- "Holding others fixed" can be unrealistic when predictors are correlated.
- A coefficient is not automatically causal.

#### Standard errors and confidence intervals
> **Definition:** A **standard error** measures uncertainty in an estimated coefficient.

A 95% confidence interval is roughly:

$$
\hat\beta_j \pm 1.96 \cdot \widehat{SE}(\hat\beta_j)
$$

(Exact multipliers depend on the t distribution and sample size.)

#### Robust standard errors
Robust SE do not change coefficients, but they change uncertainty estimates.
- HC3: common for cross-section (heteroskedasticity)
- HAC/Newey-West: common for time series (autocorrelation + heteroskedasticity)

#### Prediction vs inference
- For prediction, use time-aware evaluation and report out-of-sample metrics.
- For inference, report uncertainty and diagnose assumptions.

### Deep Dive: Regularization (Ridge vs Lasso)

Regularization adds a penalty to reduce overfitting and stabilize coefficients.

> **Definition:** **Regularization** modifies the training objective to discourage overly complex models (often large coefficients).

#### The bias/variance tradeoff (why regularization can help)
- OLS can have low bias but high variance (coefficients jump around across samples), especially with correlated predictors.
- Regularization intentionally introduces some bias to reduce variance.

#### Objectives (math)
Let $y$ be your target and $X$ your feature matrix.

OLS minimizes:

$$
\min_{\beta} \; ||y - X\beta||_2^2
$$

Ridge (L2) minimizes:

$$
\min_{\beta} \; ||y - X\beta||_2^2 + \alpha ||\beta||_2^2
$$

Lasso (L1) minimizes:

$$
\min_{\beta} \; ||y - X\beta||_2^2 + \alpha ||\beta||_1
$$

- $||\beta||_2^2 = \sum_j \beta_j^2$
- $||\beta||_1 = \sum_j |\beta_j|$

#### What changes and what does not
- As $\alpha$ increases, coefficients shrink.
- Ridge typically shrinks all coefficients toward 0.
- Lasso can drive some coefficients exactly to 0 (feature selection).

#### Why standardization matters
> **Definition:** **Standardization** rescales features to mean 0 and standard deviation 1.

Regularization penalties depend on coefficient size. If features are on different scales, the penalty is applied unevenly.
Always use `StandardScaler` before ridge/lasso.

#### Python demo: ridge vs lasso (commented)
```python
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(0)

# Create correlated predictors
n = 300
x1 = rng.normal(size=n)
x2 = 0.98 * x1 + rng.normal(scale=0.2, size=n)
X = np.column_stack([x1, x2])

# Target depends on x1

y = 1.0 + 2.0 * x1 + rng.normal(scale=1.0, size=n)

# OLS (no penalty)
ols = LinearRegression().fit(X, y)

# Ridge + scaling
ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=5.0)),
]).fit(X, y)

# Lasso + scaling
lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso(alpha=0.1, max_iter=10000)),
]).fit(X, y)

print('ols  coef:', ols.coef_)
print('ridge coef:', ridge.named_steps['model'].coef_)
print('lasso coef:', lasso.named_steps['model'].coef_)
```

#### Interpretation cautions
- Regularized coefficients are biased by design.
- They can be excellent for prediction and stability.
- Do not interpret lasso-selected features as "the true causes".
- In correlated macro data, lasso may pick one variable from a group and ignore equally good substitutes.

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`) + multicollinearity (`vif_table`)
- `src/macro.py`: GDP + labels (`gdp_growth_*`, `technical_recession_label`)
- `src/evaluation.py`: regression metrics helpers
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

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
