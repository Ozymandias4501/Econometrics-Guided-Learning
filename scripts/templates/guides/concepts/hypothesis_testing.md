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
