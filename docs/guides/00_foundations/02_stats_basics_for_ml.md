# Guide: 02_stats_basics_for_ml

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_foundations/02_stats_basics_for_ml.ipynb`.

This foundations module builds core intuition you will reuse in every later notebook.

### Key Terms (defined)
- **Time series**: data indexed by time; ordering is meaningful and must be respected.
- **Leakage**: using future information in features/labels, producing unrealistically good results.
- **Train/test split**: separating data for model fitting vs evaluation.
- **Multicollinearity**: predictors are highly correlated; coefficients can become unstable.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Correlation vs causation
- Complete notebook section: Multicollinearity (VIF)
- Complete notebook section: Bias/variance
- Complete notebook section: Hypothesis testing
- Run the bootstrap cell and confirm `PROJECT_ROOT` points to the repo root.
- Complete all TODOs (no `...` left).
- Write a short paragraph explaining a leakage example you created.

### Alternative Example (Not the Notebook Solution)
```python
# Toy leakage example (not the notebook data):
import numpy as np
import pandas as pd

idx = pd.date_range('2020-01-01', periods=200, freq='D')
y = pd.Series(np.sin(np.linspace(0, 12, len(idx))) + 0.1*np.random.randn(len(idx)), index=idx)

# Correct feature: yesterday
x_lag1 = y.shift(1)

# Leakage feature: tomorrow (do NOT do this)
x_leak = y.shift(-1)
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

This notebook introduces the core statistical vocabulary used throughout the project.

### Core Foundations: The Ideas You Will Reuse Everywhere

This project is built around a simple principle:

> **Definition:** A good model result is one that would still hold up if you had to use the model in the real world.

To get there, you need correct evaluation and correct data timing.

#### Time series vs cross-sectional (defined)
> **Definition:** A **time series** is indexed by time; ordering matters.

> **Definition:** **Cross-sectional** data compares many units at one time; ordering is not temporal.

Many ML defaults assume IID (independent and identically distributed) samples. Time series data is rarely IID.

#### What "leakage" really means
Leakage is not just a bug. It is a violation of the prediction setting.
If you are predicting the future, your features must be available in the past.

#### What "generalization" means in time series
In forecasting, generalization means:
- you train on one historical period
- you perform well in a later period

That is much harder than random-split generalization because the data generating process can change.

#### A practical habit
For every dataset row at time t, write a one-line statement:
- "At time t, we know X, and we are trying to predict Y at time t+h."

If you cannot state that clearly, it is very easy to leak information.

### Deep Dive: Correlation vs Causation (What You Can and Cannot Claim)

> **Definition:** **Correlation** means two variables move together.

> **Definition:** **Causation** means changing X would change Y (an intervention claim).

Correlation is a descriptive property of the observed data.
Causation is a claim about a data-generating mechanism.

#### Why this matters in economics
Most economic datasets are observational.
That means you usually have correlations, not controlled experiments.

If you interpret a coefficient as causal without a causal design, you can be confidently wrong.

#### Python demo: confounding creates correlation (commented)
```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

n = 1000

# z is a confounder that affects both x and y
z = rng.normal(size=n)
x = 0.8 * z + rng.normal(scale=1.0, size=n)
y = 2.0 * z + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({'x': x, 'y': y, 'z': z})

# x and y are correlated, but the relationship is driven by z
print(df.corr())
```

#### Practical interpretation
- "x is correlated with y" is a safe claim.
- "x causes y" requires identification assumptions (experiments, instruments, diff-in-diff, etc.).

#### In this project
We focus on prediction and careful interpretation.
We do not claim causal effects unless explicitly designed.

### Deep Dive: Bias/Variance and Overfitting (Why Test Performance Matters)

> **Definition:** **Overfitting** happens when a model learns patterns specific to the training data noise rather than general structure.

> **Definition:** **Bias** is systematic error from an overly simple model.

> **Definition:** **Variance** is error from sensitivity to the particular training sample.

High-level relationship:
- simple models: higher bias, lower variance
- flexible models: lower bias, higher variance

#### How it looks in practice
- Training error decreases as model complexity increases.
- Test error often decreases at first, then increases (the "U-shape" idea).

#### Python demo: simple vs flexible model (commented)
```python
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)

# Synthetic data: y is a smooth function of x + noise
n = 300
x = rng.uniform(-3, 3, size=n)
y = np.sin(x) + 0.3 * rng.normal(size=n)

# Feature matrix
X = x.reshape(-1, 1)

# Time-like split (just to keep the habit)
split = int(n * 0.8)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

lin = LinearRegression().fit(X_tr, y_tr)
tree = DecisionTreeRegressor(random_state=0).fit(X_tr, y_tr)

rmse_lin_tr = mean_squared_error(y_tr, lin.predict(X_tr), squared=False)
rmse_lin_te = mean_squared_error(y_te, lin.predict(X_te), squared=False)
rmse_tree_tr = mean_squared_error(y_tr, tree.predict(X_tr), squared=False)
rmse_tree_te = mean_squared_error(y_te, tree.predict(X_te), squared=False)

print({'lin_train': rmse_lin_tr, 'lin_test': rmse_lin_te})
print({'tree_train': rmse_tree_tr, 'tree_test': rmse_tree_te})
```

#### Interpretation
- A model with extremely low training error but high test error is overfitting.
- Regularization and simpler models can reduce variance.

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

### Deep Dive: Hypothesis Testing (How To Read p-values Without Fooling Yourself)

Hypothesis testing shows up everywhere in statistics and econometrics, especially in regression output.

#### The basic setup
> **Definition:** A **hypothesis** is a claim about a population parameter (like a mean or a regression coefficient).

> **Definition:** The **null hypothesis** $H_0$ is the default claim (often "no effect" or "no difference").

> **Definition:** The **alternative hypothesis** $H_1$ is what you consider if the null looks inconsistent with the data.

Example in regression:
- $H_0: \beta_j = 0$ (feature $x_j$ has no linear association with $y$ after controlling for other X)
- $H_1: \beta_j \ne 0$ (two-sided)

#### Test statistics, p-values, and alpha
> **Definition:** A **test statistic** is a number computed from the data that measures how incompatible the data is with $H_0$.

> **Definition:** A **p-value** is the probability (under the null model assumptions) of seeing a test statistic at least as extreme as what you observed.

> **Definition:** The **significance level** $\alpha$ is a chosen cutoff (like 0.05) for rejecting $H_0$.

Important: the p-value is **not**:
- the probability that $H_0$ is true
- the probability your model is correct
- a measure of economic importance

#### Type I / Type II errors and power
> **Definition:** A **Type I error** is rejecting $H_0$ when it is true (false positive). Probability = $\alpha$ (approximately, under assumptions).

> **Definition:** A **Type II error** is failing to reject $H_0$ when $H_1$ is true (false negative).

> **Definition:** **Power** is $1 - P(\text{Type II error})$: the probability you detect an effect when it exists.

Power increases with:
- larger sample size
- lower noise
- larger true effect size

#### Hypothesis testing in OLS regression
OLS coefficient estimates:

$$
\hat\beta = (X'X)^{-1}X'y
$$

A typical coefficient test uses a t-statistic:

$$
 t_j = \frac{\hat\beta_j - 0}{\widehat{SE}(\hat\beta_j)}
$$

- If model assumptions hold, $t_j$ is compared to a t distribution.
- The p-value is derived from that distribution.

> **Key idea:** Changing the standard error estimator changes the t-statistic and p-value, even when the coefficient stays the same.

#### Robust standard errors and hypothesis testing
- **Plain OLS SE** assume homoskedastic, uncorrelated errors.
- **HC3 SE** relax heteroskedasticity (common in cross-section).
- **HAC/Newey-West SE** relax autocorrelation + heteroskedasticity (common in time series).

This project uses robust SE to avoid overly confident inference.

#### Confidence intervals and hypothesis tests (relationship)
A 95% confidence interval for $\beta_j$ is roughly:

$$
\hat\beta_j \pm t_{0.975} \cdot \widehat{SE}(\hat\beta_j)
$$

If the interval does not include 0, the two-sided p-value is typically < 0.05.

#### Python demo: a simple t-test vs a regression coefficient test
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

rng = np.random.default_rng(0)

# 1) One-sample t-test: is the mean of x equal to 0?
x = rng.normal(loc=0.2, scale=1.0, size=200)
t_stat, p_val = stats.ttest_1samp(x, popmean=0.0)
print('t-test t:', t_stat, 'p:', p_val)

# 2) Regression t-test: is slope on x equal to 0?
# Create y that depends on x
n = 300
x2 = rng.normal(size=n)
y = 1.0 + 0.5 * x2 + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({'y': y, 'x': x2})
X = sm.add_constant(df[['x']])
res = sm.OLS(df['y'], X).fit()
print(res.summary())

# Manual t-stat for slope (matches summary output)
beta_hat = res.params['x']
se_hat = res.bse['x']
print('manual t:', beta_hat / se_hat)
```

#### How hypothesis tests go wrong in macro/ML workflows
Common failure modes:
- **Multiple testing**: trying many features/specifications inflates false positives.
- **P-hacking**: changing the spec until p-values look good.
- **Autocorrelation/nonstationarity**: time series violate assumptions; naive SE can be wildly wrong.
- **Confounding**: significance does not imply causation.

> **Definition:** **Multiple testing** means running many hypothesis tests; even if all nulls are true, some p-values will be small by chance.

Practical rule: if you searched over 50 features/specs, a few p-values < 0.05 are expected even with no real signal.

#### How to use p-values responsibly in this project
- Prefer robust SE (HC3 / HAC) when appropriate.
- Treat p-values as one piece of evidence, not the goal.
- Report effect sizes and uncertainty (confidence intervals), not just "significant".
- Use out-of-sample evaluation for predictive tasks.

#### Project touchpoints (where hypothesis testing shows up)
- Regression notebooks use `statsmodels` summaries and ask you to interpret:
  - coefficients, standard errors, t-stats, p-values, and confidence intervals
- `src/econometrics.py` provides convenience wrappers:
  - `fit_ols_hc3` for cross-sectional robust SE
  - `fit_ols_hac` for time-series robust SE

### Project Code Map
- `scripts/scaffold_curriculum.py`: how this curriculum is generated (for curiosity)
- `src/evaluation.py`: time splits and metrics used later
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Using `train_test_split(shuffle=True)` on time-indexed data.
- Looking at the test set repeatedly while tuning ("test leakage").
- Assuming a significant p-value implies causation.
- Running many tests/specs and treating a small p-value as proof (multiple testing / p-hacking).

<a id="summary"></a>
## Summary + Suggested Readings

You now have the tooling to avoid the two most common beginner mistakes in economic ML:
1) leaking future information, and
2) over-interpreting correlated coefficients.


Suggested readings:
- Hyndman & Athanasopoulos: Forecasting: Principles and Practice (time series basics)
- Wooldridge: Introductory Econometrics (interpretation + pitfalls)
- scikit-learn docs: model evaluation and cross-validation
