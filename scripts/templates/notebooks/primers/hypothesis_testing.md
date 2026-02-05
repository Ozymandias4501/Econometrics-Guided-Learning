## Primer: Hypothesis testing (p-values, t-stats, confidence intervals)

You will see p-values, t-statistics, and confidence intervals in regression output (especially `statsmodels`). This primer gives you the minimum to interpret them correctly.

### The objects (plain language)

- **Null hypothesis** $H_0$: the default claim (often “no effect”).
- **Alternative** $H_1$: the claim you consider if the data looks inconsistent with $H_0$.
- **Test statistic**: “how far” your estimate is from the null, in uncertainty units.
- **p-value**: probability (under the null *and model assumptions*) of seeing a test statistic at least as extreme as observed.
- **Confidence interval (CI)**: a range of parameter values consistent with the data under assumptions.

### What a p-value is NOT

- Not the probability $H_0$ is true.
- Not the probability the model is correct.
- Not a measure of economic importance.

### Regression t-test intuition

In OLS, a common test is $H_0: \\beta_j = 0$.

$$
t_j = \\frac{\\hat\\beta_j}{\\widehat{SE}(\\hat\\beta_j)}
$$

If you change your SE estimator (HC3/HAC/cluster), you change $\\widehat{SE}$ and therefore the p-value, even if the coefficient stays the same.

### Expected output / what you should look at in `res.summary()`

- `coef`: effect size (in model units)
- `std err`: uncertainty
- CI columns: magnitude + uncertainty together

### Common pitfalls in this project

- Macro time series often have autocorrelation → naive SE too small → use HAC when interpreting p-values.
- Multiple testing/spec-search can produce small p-values by chance.
- Predictive success ≠ causal interpretation.

### Tiny demo (toy; not project data)

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)
n = 300
x = rng.normal(size=n)
y = 1.0 + 0.5 * x + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({"y": y, "x": x})
X = sm.add_constant(df[["x"]])
res = sm.OLS(df["y"], X).fit()
print(res.summary())
```
