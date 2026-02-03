## Primer: Hypothesis Testing (p-values, t-tests, and Confidence Intervals)

You will see p-values, t-statistics, and confidence intervals in regression output (especially `statsmodels`).
This primer gives you the minimum you need to avoid the most common misunderstandings.

### Definitions (in plain language)
- **Hypothesis**: a claim about an unknown population quantity (a parameter).
- **Null hypothesis** $H_0$: the "default" claim (often “no effect”).
- **Alternative hypothesis** $H_1$: the claim you consider if the data looks inconsistent with $H_0$.
- **Test statistic**: a number computed from data that measures how surprising the data is under $H_0$.
- **p-value**: the probability (under the assumptions of the null model) of seeing a test statistic at least as extreme as you observed.
- **Significance level** $\alpha$: a pre-chosen cutoff (commonly 0.05) used to decide whether to reject $H_0$.

### What a p-value is NOT
- It is **not** the probability $H_0$ is true.
- It is **not** the probability your model is correct.
- It is **not** a measure of economic importance.

### Type I / Type II errors and power
- **Type I error (false positive)**: rejecting $H_0$ when $H_0$ is true. Rough probability $\approx \alpha$ under assumptions.
- **Type II error (false negative)**: failing to reject $H_0$ when $H_1$ is true.
- **Power**: $1 - P(\text{Type II error})$. Power increases with larger samples and larger true effects.

### Regression t-test intuition
In OLS regression we estimate coefficients $\hat\beta$. A common test is:
- $H_0: \beta_j = 0$ (no linear association between $x_j$ and $y$ holding other features fixed)

The **t-statistic** is:

$$
t_j = \frac{\hat\beta_j - 0}{\widehat{SE}(\hat\beta_j)}
$$

Interpretation (roughly):
- if $|t_j|$ is large, $\hat\beta_j$ is far from 0 relative to its uncertainty estimate
- if $|t_j|$ is small, the data is compatible with $\beta_j$ being near 0 (given assumptions)

### Confidence intervals (CI) connect to hypothesis tests
A 95% CI is usually reported as:

$$
\hat\beta_j \pm t_{0.975}\cdot \widehat{SE}(\hat\beta_j)
$$

If the 95% CI does not include 0, the two-sided p-value is typically < 0.05.

### Python demo (toy): one-sample t-test and a regression coefficient test
This is not your project data; it is purely to make the objects concrete.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

rng = np.random.default_rng(0)

# 1) One-sample t-test: is the mean of x equal to 0?
x = rng.normal(loc=0.2, scale=1.0, size=200)
t_stat, p_val = stats.ttest_1samp(x, popmean=0.0)
print("t-test t:", t_stat, "p:", p_val)

# 2) Regression t-test: is the slope on x equal to 0?
n = 300
x2 = rng.normal(size=n)
eps = rng.normal(scale=1.0, size=n)
y = 1.0 + 0.5 * x2 + eps

df = pd.DataFrame({"y": y, "x": x2})
X = sm.add_constant(df[["x"]])
res = sm.OLS(df["y"], X).fit()

# The summary includes coef, SE, t, p, and CI
print(res.summary())

# Manual t-stat for slope (matches summary output)
beta_hat = res.params["x"]
se_hat = res.bse["x"]
print("manual t:", beta_hat / se_hat)
```

### Common ways hypothesis testing goes wrong in ML + macro
- **Multiple testing**: you try many features/specifications; some will look “significant” by chance.
- **Violating assumptions**: autocorrelation and heteroskedasticity can make naive SE too small.
- **Confusing predictive success with causal claims**: a coefficient can predict well without being causal.

Practical guidance for this project:
- Report effect sizes + uncertainty, not just “significant / not significant.”
- For macro time series, prefer robust SE (HAC/Newey-West) when interpreting p-values.
- For predictive tasks, always complement p-values with out-of-sample evaluation.
