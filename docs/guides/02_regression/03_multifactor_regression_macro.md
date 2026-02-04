# Guide: 03_multifactor_regression_macro

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/03_multifactor_regression_macro.ipynb`.

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
- Complete notebook section: Choose features
- Complete notebook section: Fit model
- Complete notebook section: VIF + stability
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

### Deep Dive: Multicollinearity and VIF (Why Coefficients Become Unstable)

> **Definition:** **Multicollinearity** means two or more predictors contain overlapping information (they are highly correlated).

Multicollinearity is especially common in macro data because many indicators move together (business cycle, policy regimes).

#### What multicollinearity does (and does not do)
- It often does **not** hurt prediction much.
- It **does** make individual coefficients unstable.
- It inflates standard errors, making p-values fragile.

#### Regression notation
We write a linear regression as:

$$
\mathbf{y} = \mathbf{X}\beta + \varepsilon
$$

- $\mathbf{y}$ is an $n \times 1$ vector of outcomes.
- $\mathbf{X}$ is an $n \times p$ matrix of predictors.
- $\beta$ is a $p \times 1$ vector of coefficients.
- $\varepsilon$ is an $n \times 1$ vector of errors.

When columns of $\mathbf{X}$ are nearly linearly dependent, $(\mathbf{X}'\mathbf{X})$ is close to singular, and coefficient estimates become unstable.

#### VIF (Variance Inflation Factor)
> **Definition:** The **variance inflation factor** for feature $j$ is:

$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^2}
$$

Where $R_j^2$ is from regressing $x_j$ on all the other predictors.

Interpretation:
- If $R_j^2$ is high, $x_j$ is well-explained by other predictors.
- Then $\mathrm{VIF}_j$ is high, meaning the variance of $\hat\beta_j$ is inflated.

Rules of thumb (not laws):
- VIF > 5 suggests notable collinearity.
- VIF > 10 suggests serious collinearity.

#### Python demo: correlated predictors -> unstable coefficients (commented)
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.econometrics import vif_table

rng = np.random.default_rng(0)

# Create two highly correlated predictors
n = 600
x1 = rng.normal(size=n)
x2 = 0.95 * x1 + rng.normal(scale=0.2, size=n)  # mostly the same information

# True outcome depends only on x1
# In the presence of collinearity, the model may "split" credit unpredictably.
y = 1.0 + 2.0 * x1 + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

# VIF quantifies how redundant each predictor is
print(vif_table(df, ['x1', 'x2']))

# Fit OLS with both predictors
X = sm.add_constant(df[['x1', 'x2']])
res = sm.OLS(df['y'], X).fit()

print('params:', res.params.to_dict())
print('std err:', res.bse.to_dict())
```

#### What to do about multicollinearity
> **Definition:** **Regularization** (like ridge) adds a penalty that stabilizes coefficients.

Practical options:
- Drop one variable from a correlated group.
- Combine variables (domain composite, PCA/factors).
- Use ridge regression to stabilize.
- Focus on prediction rather than coefficient interpretation.

#### Economics interpretation warning
In macro data, "holding other indicators fixed" can be an unrealistic counterfactual.
If two indicators are tightly linked, the idea of changing one while freezing the other is not economically meaningful.
Treat coefficients as conditional correlations unless you have a causal design.

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
