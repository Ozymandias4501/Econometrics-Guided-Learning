# Guide: 00_single_factor_regression_micro

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/00_single_factor_regression_micro.ipynb`.

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
- Complete notebook section: Load census data
- Complete notebook section: Build log variables
- Complete notebook section: Fit OLS + HC3
- Complete notebook section: Interpretation
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

### Deep Dive: Log-Log Regression (Elasticity-Style Interpretation)

In microeconomic cross-sectional data, log transforms are common because many variables (income, population, rent) are heavy-tailed.

#### Key terms (defined)
> **Definition:** **Cross-sectional data** observes many units (counties) at one time.

> **Definition:** A **log transform** uses $\log(x)$ to compress scale and turn multiplicative relationships into additive ones.

#### Log-log model and interpretation
A log-log regression is:

$$
\log(y) = \alpha + \beta \log(x) + \varepsilon
$$

Interpretation (rule of thumb):
- a 1% increase in $x$ is associated with about a $\beta$% increase in $y$.

Why? Because for small changes:

$$
\Delta \log(x) \approx \frac{\Delta x}{x}
$$

#### Pitfall: zeros and missing values
- $\log(0)$ is undefined.
- Filter out non-positive values or use a different transform (`log1p`) if justified.

#### Python demo: simulated log-log relationship (commented)
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)

n = 800
x = np.exp(rng.normal(size=n))

# y grows with x^0.3 (multiplicative)
y = 2.0 * (x ** 0.3) * np.exp(rng.normal(scale=0.2, size=n))

df = pd.DataFrame({'x': x, 'y': y})
df['lx'] = np.log(df['x'])
df['ly'] = np.log(df['y'])

X = sm.add_constant(df[['lx']])
res = sm.OLS(df['ly'], X).fit()
print(res.params)
```

### Deep Dive: Robust Standard Errors (HC3) for Cross-Sectional Data

Cross-sectional data often has heteroskedasticity.

> **Definition:** **Heteroskedasticity** means the variance of errors changes with predictors.

If you ignore it, OLS coefficients can be the same, but standard errors can be wrong.

#### What changes and what does not
- Coefficients $\hat\beta$ do not change.
- Standard errors, t-stats, p-values, and confidence intervals change.

#### Python demo: heteroskedastic errors and robust SE (commented)
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)

n = 400
x = rng.normal(size=n)

# Error variance increases with |x|
eps = rng.normal(scale=1 + 2*np.abs(x), size=n)
y = 1.0 + 0.5*x + eps

X = sm.add_constant(pd.DataFrame({'x': x}))
res = sm.OLS(y, X).fit()
res_hc3 = res.get_robustcov_results(cov_type='HC3')

print('naive SE:', res.bse)
print('HC3 SE  :', res_hc3.bse)
```

#### Interpretation warning
Robust SE improves uncertainty estimates under heteroskedasticity.
It does not fix confounding or make a coefficient causal.

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
