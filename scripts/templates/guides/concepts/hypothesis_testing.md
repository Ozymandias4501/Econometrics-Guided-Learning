### Deep Dive: Hypothesis Testing — how to read p-values without fooling yourself

Hypothesis tests show up everywhere in econometrics output. The goal of this section is not to worship p-values, but to understand what they *are* and what they *are not*.

#### 1) Intuition (plain English)

A hypothesis test is a structured way to ask:
- “If the true effect were zero, how surprising is my estimate?”

It is **not** a direct answer to:
- “What is the probability the effect is real?”
- “Is my model correct?”

**Story example:** You regress unemployment on an interest-rate spread and get a small p-value.
That might mean:
- the relationship is real in-sample,
- or your SE are wrong (autocorrelation),
- or you tried many specs (multiple testing),
- or the effect is tiny but precisely estimated.

#### 2) Notation + setup (define symbols)

We usually test a claim about a population parameter $\\theta$ (mean, regression coefficient, difference in means, …).

Define:
- $H_0$: the **null hypothesis** (default claim),
- $H_1$: the **alternative hypothesis** (what you consider if evidence contradicts $H_0$),
- $T$: a **test statistic** computed from data,
- $\\alpha$: a pre-chosen significance level (e.g., 0.05).

Example in regression:
- $H_0: \\beta_j = 0$
- $H_1: \\beta_j \\neq 0$ (two-sided)

#### 3) Assumptions (why tests are conditional statements)

Every p-value is conditional on:
- the statistical model (e.g., OLS assumptions),
- the standard error estimator you use (naive vs robust vs HAC vs clustered),
- the sample and selection process.

If those assumptions fail, the p-value may be meaningless.

#### 4) Estimation mechanics in OLS: where t-stats come from

OLS estimates coefficients:

$$
\\hat\\beta = (X'X)^{-1}X'y.
$$

For coefficient $\\beta_j$, you compute an estimated standard error $\\widehat{SE}(\\hat\\beta_j)$.

The t-statistic for testing $H_0: \\beta_j = 0$ is:

$$
t_j = \\frac{\\hat\\beta_j - 0}{\\widehat{SE}(\\hat\\beta_j)}.
$$

**What each term means**
- numerator: your estimated effect.
- denominator: your uncertainty estimate.
- large |t| means “many standard errors away from 0.”

Under suitable assumptions, $t_j$ is compared to a t distribution (or asymptotic normal), producing a p-value.

#### 5) What the p-value actually means

> **Definition:** The **p-value** is the probability (under the null and model assumptions) of observing a test statistic at least as extreme as what you observed.

So:
- p-value is about the *data under the null model*,
- not about the probability the null is true.

Also: p-values do not measure effect size.

#### 6) Confidence intervals (often more informative than p-values)

A 95% confidence interval is approximately:

$$
\\hat\\beta_j \\pm t_{0.975} \\cdot \\widehat{SE}(\\hat\\beta_j).
$$

Interpretation:
- it is a range of values consistent with the data under assumptions,
- it shows both sign and magnitude uncertainty.

If the 95% CI excludes 0, the two-sided p-value is typically < 0.05.

#### 7) Robust SE change p-values (without changing coefficients)

Different SE estimators correspond to different assumptions about errors:
- **Naive OLS SE:** homoskedastic, uncorrelated errors.
- **HC3:** heteroskedasticity-robust (cross-section).
- **HAC/Newey–West:** autocorrelation + heteroskedasticity (time series).
- **Clustered SE:** within-cluster correlated errors (panels/DiD).

**Key idea:** changing SE changes $\\widehat{SE}(\\hat\\beta_j)$ → changes t-stat and p-value, even when $\\hat\\beta_j$ is identical.

#### 8) Diagnostics: how hypothesis testing goes wrong (minimum set)

1) **Multiple testing**
- If you try many features/specs, some will “work” by chance.
- A few p-values < 0.05 are expected even if all true effects are 0.

2) **P-hacking / specification search**
- tweaking the model until p-values look good invalidates the usual interpretation.

3) **Wrong SE (dependence)**
- autocorrelation or clustering can make naive SE far too small.

4) **Confounding**
- a “significant” association is not a causal effect without identification.

Practical rule:
- interpret p-values as one piece of evidence, not a conclusion.

#### 9) Interpretation + reporting (how to write results responsibly)

Good reporting includes:
- effect size (coefficient) in meaningful units,
- uncertainty (CI preferred),
- correct SE choice for the data structure,
- a note about model limitations and identification.

**What this does NOT mean**
- “Significant” is not “important.”
- “Not significant” is not “no effect” (could be low power).

#### 10) Small Python demo (optional)

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

rng = np.random.default_rng(0)

# 1) One-sample t-test
x = rng.normal(loc=0.2, scale=1.0, size=200)
t_stat, p_val = stats.ttest_1samp(x, popmean=0.0)
print('t-test t:', t_stat, 'p:', p_val)

# 2) Regression t-test
n = 300
x2 = rng.normal(size=n)
y = 1.0 + 0.5 * x2 + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({'y': y, 'x': x2})
X = sm.add_constant(df[['x']])
res = sm.OLS(df['y'], X).fit()
print(res.summary())
```

#### Exercises

- [ ] Take one regression output and rewrite it in words: coefficient, CI, and what assumptions the p-value relies on.
- [ ] Show how p-values change when you switch from naive SE to HC3 or HAC (same coefficient, different uncertainty).
- [ ] Create a multiple-testing demonstration: test 50 random predictors against random noise and count how many p-values < 0.05.
- [ ] Write 6 sentences explaining why “statistically significant” is not the same as “economically meaningful.”
