# Guide: 02_stats_basics_for_ml

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_foundations/02_stats_basics_for_ml.ipynb`.

This guide covers the statistical foundations that every applied analysis rests on: hypothesis testing, confidence intervals, the distinction between correlation and causation, the law of large numbers and CLT, multicollinearity, and the bias-variance tradeoff. These are not abstract topics -- they determine whether your regression output is trustworthy or misleading.

For a health economist working as a data analyst, these concepts appear daily: interpreting a clinical trial's $p$-value, deciding whether an observational association is causal, diagnosing why a regression coefficient flipped sign when you added a variable, or explaining to a stakeholder what a confidence interval actually means.

### Key Terms (defined)
- **p-value**: the probability of observing data at least as extreme as what you got, assuming the null hypothesis and all model assumptions are true. It is NOT the probability that the null is true.
- **Type I error ($\alpha$)**: rejecting the null when it is actually true (false positive). Conventional threshold: 0.05.
- **Type II error ($\beta$)**: failing to reject the null when it is actually false (false negative).
- **Statistical power ($1 - \beta$)**: the probability of correctly rejecting a false null. Depends on sample size, effect size, and $\alpha$.
- **Confidence interval**: a range constructed so that, across repeated samples, $(1-\alpha) \times 100\%$ of such intervals would contain the true parameter. It does NOT mean there is a 95% probability the parameter is inside this particular interval.
- **Multicollinearity**: predictors are highly correlated with each other; individual coefficients become unstable even though the overall model fit may be fine.
- **VIF (Variance Inflation Factor)**: a diagnostic that quantifies how much the variance of a coefficient is inflated by collinearity with other predictors.
- **Confounder**: a variable that causally affects both the treatment/exposure and the outcome, creating a spurious association if not controlled for.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Correlation vs causation
- Complete notebook section: Multicollinearity (VIF)
- Complete notebook section: Bias/variance
- Complete notebook section: Hypothesis testing
- Run the bootstrap cell and confirm `PROJECT_ROOT` points to the repo root.
- Complete all TODOs (no `...` left).
- Compute VIF for a set of predictors and write a 3-sentence interpretation.
- Run a hypothesis test, report the $p$-value and CI, and write one sentence stating what they do and do not tell you.

### Alternative Example (Not the Notebook Solution)
```python
# Toy confounding example (not the notebook data):
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500

# Z is a common cause (confounder)
z = rng.normal(size=n)
x = 0.8 * z + rng.normal(scale=0.5, size=n)  # Z -> X
y = 1.2 * z + rng.normal(scale=0.5, size=n)  # Z -> Y (X does NOT cause Y)

# Naive correlation is strong, but it's not causal
print(f"Correlation(x, y): {np.corrcoef(x, y)[0, 1]:.3f}")

# Controlling for Z removes the spurious association
import statsmodels.api as sm
X_naive = sm.add_constant(pd.DataFrame({'x': x}))
X_ctrl  = sm.add_constant(pd.DataFrame({'x': x, 'z': z}))
print("Naive coef on x:", sm.OLS(y, X_naive).fit().params['x'].round(3))
print("Controlled coef on x:", sm.OLS(y, X_ctrl).fit().params['x'].round(3))
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

This guide covers the statistical vocabulary and reasoning patterns that underlie every regression, every test, and every model comparison in this project.

### Deep Dive: Hypothesis testing -- p-values, Type I/II error, and power

#### 1) Intuition (plain English)

A hypothesis test asks: "Is the pattern I see in the data compatible with pure chance (the null hypothesis), or is it strong enough that chance alone is an unlikely explanation?"

**Story example (health econ):** A hospital implements a new discharge protocol. Average length of stay drops from 5.2 to 4.8 days. Is that a real effect or random variation? A hypothesis test formalizes this question. But the answer depends on sample size, variability, and what you define as "the null."

#### 2) Notation and setup

- $H_0$: null hypothesis (e.g., $\beta = 0$, "no effect of the protocol").
- $H_1$: alternative hypothesis (e.g., $\beta \neq 0$).
- Test statistic: $t = \hat\beta / \widehat{SE}(\hat\beta)$.
- $p$-value: $P(\text{data this extreme or more} \mid H_0 \text{ is true})$.

The decision rule at significance level $\alpha$:
- If $p < \alpha$, reject $H_0$.
- If $p \ge \alpha$, fail to reject $H_0$ (this is NOT the same as accepting $H_0$).

#### 3) Type I error, Type II error, and power

|  | $H_0$ true (no real effect) | $H_0$ false (real effect exists) |
|---|---|---|
| **Reject $H_0$** | Type I error (false positive), prob = $\alpha$ | Correct rejection, prob = $1-\beta$ (power) |
| **Fail to reject $H_0$** | Correct, prob = $1-\alpha$ | Type II error (false negative), prob = $\beta$ |

**Practical implications for a health economist:**
- **Type I error:** You conclude a drug reduces readmissions when it does not. Wasted resources, possible harm.
- **Type II error:** You conclude a drug has no effect when it actually does. Patients miss a beneficial treatment.
- **Power** increases with: larger sample size, larger true effect, larger $\alpha$, lower noise.

**Rule of thumb:** Clinical trials typically target 80% power at $\alpha = 0.05$. Observational studies in health econ often have lower power due to noisy data and small effect sizes.

#### 4) What p-values do and do not tell you

A $p$-value of 0.03 means: "If the null were true and all model assumptions held, there is a 3% chance of seeing a result this extreme."

It does **NOT** mean:
- "There is a 97% probability the effect is real."
- "The effect is large or clinically meaningful."
- "The model is correctly specified."

**Worked example:**

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(99)
n = 200

# True effect is small but nonzero
treatment = rng.binomial(1, 0.5, size=n)
outcome = 0.15 * treatment + rng.normal(scale=1.0, size=n)  # true beta = 0.15

X = sm.add_constant(treatment)
result = sm.OLS(outcome, X).fit()

print(f"Estimated beta: {result.params[1]:.3f}")
print(f"SE:             {result.bse[1]:.3f}")
print(f"t-stat:         {result.tvalues[1]:.3f}")
print(f"p-value:        {result.pvalues[1]:.4f}")
print(f"95% CI:         [{result.conf_int().iloc[1, 0]:.3f}, {result.conf_int().iloc[1, 1]:.3f}]")
# With n=200 and a small effect, you may or may not reject at 0.05.
# That's the power problem: small effects need large samples.
```

#### 5) The SE estimator changes the p-value (not the coefficient)

This is crucial for applied work. The same $\hat\beta$ can be "significant" or "not significant" depending on how you estimate the standard error:

- **Naive (homoskedastic) SE**: assumes constant error variance. Often too small.
- **HC3 (robust) SE**: allows heteroskedasticity. Standard in cross-sectional health econ.
- **HAC (Newey-West) SE**: allows heteroskedasticity and autocorrelation. Required for time-series data.
- **Clustered SE**: allows correlation within groups (e.g., patients within hospitals). Required for panel data.

Changing the SE estimator changes $t$, $p$, and the CI width without changing $\hat\beta$.

### Deep Dive: Confidence intervals -- what they actually mean

#### 1) The correct interpretation

A 95% confidence interval means: **if you repeated the sampling procedure many times, 95% of the resulting intervals would contain the true parameter.**

It does NOT mean: "There is a 95% probability that $\beta$ is between $a$ and $b$." The parameter $\beta$ is fixed (not random); it either is or is not in the interval. The randomness is in the interval itself (which changes with each sample).

#### 2) Why CIs are often more useful than p-values

A CI tells you:
- **Sign:** Is the entire CI positive? Then you can be fairly confident the effect is positive.
- **Magnitude:** Is the CI $[0.01, 0.03]$ or $[0.01, 3.50]$? The first is precise; the second means you know almost nothing about the size.
- **Practical significance:** A CI of $[0.001, 0.005]$ for a drug effect on mortality might be statistically significant but clinically negligible.

**Health econ example:** A study of Medicaid expansion finds a coefficient on ER visits of $-12.3$ with 95% CI $[-22.1, -2.5]$. This tells you the effect is likely negative (reduced ER visits) and somewhere between 2.5 and 22.1 fewer visits. That range matters for policy -- the low end may not justify the cost, the high end clearly does.

#### 3) Formula and connection to hypothesis testing

For a coefficient $\hat\beta_j$ with estimated SE:

$$
CI_{95\%} = \hat\beta_j \pm t_{0.975, \, n-k} \cdot \widehat{SE}(\hat\beta_j).
$$

A 95% CI excludes zero if and only if the $p$-value for $H_0: \beta_j = 0$ is below 0.05. They are dual representations of the same test.

### Deep Dive: Correlation vs causation -- the question you must not confuse

#### 1) Intuition (plain English)

Correlation answers: "Do $X$ and $Y$ move together?"

Causation answers: "If I intervene and change $X$, does $Y$ change?"

**Story example:** Counties with more hospital beds per capita have higher mortality rates. Does building hospitals kill people? No -- sicker, older populations attract more hospital capacity. The confounder (population health/age) drives both.

#### 2) Notation

Correlation between random variables $X$ and $Y$:

$$
\rho_{XY} = \frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X)\mathrm{Var}(Y)}}.
$$

Correlation is symmetric ($\rho_{XY} = \rho_{YX}$) and unit-free, lying in $[-1, 1]$. Causal effects are directional and have units.

#### 3) Confounding: how correlation arises without causation

A confounder $Z$ causes both $X$ and $Y$:

$$
X = aZ + \eta, \qquad Y = bZ + \varepsilon.
$$

Even if $X$ does not cause $Y$, they will be correlated because they share the common cause $Z$.

Regression "controls" can help if you measure the confounder, but:
- you rarely observe all confounders,
- controlling for the wrong variables (colliders, mediators) can introduce bias.

#### 4) What you need for a causal claim

A causal claim requires an **identification strategy**:
- Randomization (RCT -- the gold standard in clinical research).
- Natural experiment (e.g., a policy change that affects some groups but not others).
- Quasi-experimental design (DiD, IV, RD) with defensible assumptions.
- A structural model with explicit, testable assumptions.

Without identification, regression coefficients are best described as **conditional associations**, not causal effects.

### Deep Dive: Law of Large Numbers and CLT -- why sample means work

#### 1) Intuition (plain English)

The **Law of Large Numbers (LLN)** says: as your sample gets larger, the sample mean converges to the population mean. This is why surveys, clinical trials, and administrative data analyses work -- with enough observations, you get close to the truth.

The **Central Limit Theorem (CLT)** says: regardless of the original distribution, the sampling distribution of the sample mean becomes approximately normal as $n$ grows. This is why $t$-tests and confidence intervals work even when the underlying data is skewed.

#### 2) Formal statements (brief)

**LLN:** If $X_1, X_2, \dots, X_n$ are i.i.d. with mean $\mu$, then:

$$
\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \xrightarrow{p} \mu \quad \text{as } n \to \infty.
$$

**CLT:** Under the same conditions, with finite variance $\sigma^2$:

$$
\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2).
$$

Equivalently, $\bar{X}_n$ is approximately $N(\mu, \sigma^2/n)$ for large $n$.

#### 3) Why this matters practically

- The CLT justifies using normal-based confidence intervals and $t$-tests even when health outcomes (costs, lengths of stay) are heavily right-skewed.
- The LLN tells you that with a large enough administrative dataset, your sample averages are reliable estimates of population quantities.
- **Caveat:** Both require independence (or at least weak dependence). Clustered data (patients within hospitals) or time-series data violate this, which is why you need clustered SEs or HAC SEs.

**Health econ example:** Hospital cost data is notoriously right-skewed (a few very expensive cases). With $n = 50$, the sampling distribution of mean cost may not be well-approximated by a normal. With $n = 5{,}000$, the CLT kicks in and normal-based inference is reliable.

### Deep Dive: Multicollinearity and VIF

#### 1) Intuition (plain English)

Multicollinearity means your predictors are highly correlated with each other. This does **not** bias your coefficient estimates (if exogeneity holds), but it inflates their standard errors, making it hard to tell which variable is doing the work.

**Story example:** You regress patient outcomes on both "number of prescriptions" and "number of doctor visits." These are highly correlated (sicker patients have more of both). OLS cannot cleanly separate their individual effects, so both coefficients may be imprecise and sensitive to the sample.

#### 2) The VIF diagnostic

The **Variance Inflation Factor** for predictor $j$:

$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^2}
$$

where $R_j^2$ is the $R^2$ from regressing feature $j$ on all other features.

- $\mathrm{VIF}_j = 1$: no collinearity (feature $j$ is uncorrelated with others).
- $\mathrm{VIF}_j = 5$: the variance of $\hat\beta_j$ is 5x what it would be without collinearity.
- $\mathrm{VIF}_j > 10$: serious collinearity; interpret $\hat\beta_j$ with caution.

#### 3) When multicollinearity matters vs. when it does not

| Goal | Does multicollinearity matter? | Why |
|---|---|---|
| Interpreting individual coefficients | Yes | SEs are inflated; coefficients are unstable |
| Overall prediction | Usually not much | Collinear features are redundant but not harmful for $\hat{y}$ |
| Causal inference (treatment effect) | Depends | If treatment is collinear with controls, the treatment effect estimate is imprecise |
| Variable selection | Yes | Hard to tell which variables to include/exclude |

#### 4) Worked example

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

rng = np.random.default_rng(11)
n = 300

# Two highly correlated health predictors + one independent
age = rng.normal(50, 10, size=n)
bmi = 0.3 * age + rng.normal(0, 3, size=n)         # correlated with age
exercise_hours = rng.normal(5, 2, size=n)            # independent

# Outcome
cost = 200 + 50 * age + 30 * bmi + (-80) * exercise_hours + rng.normal(0, 500, size=n)

X = pd.DataFrame({"age": age, "bmi": bmi, "exercise": exercise_hours})
X_const = sm.add_constant(X)

# Compute VIF
for i, col in enumerate(X_const.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X_const.values, i)
    print(f"VIF({col}): {vif:.2f}")

# Regression
model = sm.OLS(cost, X_const).fit()
print("\n", model.summary2().tables[1].to_string())
# Note: age and bmi will have high VIF and wide CIs, exercise will not.
```

#### 5) Solutions for multicollinearity

- **Drop redundant variables** if theory supports it (e.g., keep BMI, drop weight and height separately).
- **Regularization** (ridge regression shrinks correlated coefficients toward each other; lasso selects one).
- **PCA** if you do not need interpretable coefficients.
- **Accept it** if your goal is prediction and the overall model performs well out-of-sample.

> **Full treatment**: See the [regression guide: Multicollinearity and VIF](../02_regression/00_single_factor_regression_micro.md#deep-dive-multicollinearity-and-vif--why-coefficients-become-unstable) for the complete derivation with matrix algebra.

### Deep Dive: Bias-variance tradeoff and overfitting

#### 1) Intuition (plain English)

Models can fail in two ways:
- **Too simple (high bias):** cannot capture real patterns. Example: fitting a straight line to a clearly nonlinear dose-response curve.
- **Too flexible (high variance):** fits noise that does not repeat. Example: a 20-degree polynomial that passes through every training point but oscillates wildly on new data.

Overfitting is when performance looks great on training data but poor on new data.

#### 2) Formal decomposition

Let the true outcome be $y = f(x) + \varepsilon$ and the model prediction be $\hat{f}(x)$. The expected prediction error decomposes as:

$$
\mathbb{E}[(\hat{f}(x) - y)^2]
= \underbrace{(\mathbb{E}[\hat{f}(x)] - f(x))^2}_{\text{bias}^2}
\; + \;
\underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{variance}}
\; + \;
\underbrace{\mathrm{Var}(\varepsilon)}_{\text{irreducible noise}}.
$$

- **Bias**: systematic error from underfitting / misspecification.
- **Variance**: sensitivity to the particular training sample.
- **Noise**: inherent unpredictability in outcomes. No model can reduce this.

#### 3) Practical implications

- Adding more correlated predictors can increase variance without improving out-of-sample performance.
- Specification search (trying many models) is a form of overfitting: you select the model that happened to look best on this sample.
- Regularization (ridge, lasso, elastic net) explicitly trades a small increase in bias for a large reduction in variance.

#### 4) Connection to health econ

In health economics, datasets are often moderate-sized (hundreds to low thousands of observations, not millions). This means:
- Overfitting is a constant threat, especially with many potential control variables.
- Cross-validation (time-aware for panel/time-series data) is essential for honest model evaluation.
- "Kitchen sink" regressions (throwing in every available variable) frequently overfit.

### Common Mistakes (statistics-specific)
- Interpreting a confidence interval as "95% probability the parameter is in this range" (it is not -- the parameter is fixed; the interval is random across samples).
- Treating "not statistically significant" as "no effect." It may just mean low power.
- Running many specifications or subgroup analyses and reporting only the one with $p < 0.05$ (p-hacking / multiple testing).
- Ignoring multicollinearity warnings and over-interpreting individual coefficients from a model with VIF > 10.
- Confusing statistical significance with practical/clinical significance. A drug that reduces blood pressure by 0.5 mmHg may be "significant" with $n = 50{,}000$ but clinically meaningless.
- Assuming that "controlling for confounders" in a regression makes the coefficient causal. Without an identification strategy, it does not.
- Using the wrong SE estimator for the data structure (e.g., naive SEs on clustered data).

### Project Code Map
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)

#### Exercises

- [ ] Simulate a confounder and show that the naive correlation between $X$ and $Y$ disappears when you control for $Z$.
- [ ] Compute VIF for a regression with 5+ predictors. Identify the most collinear pair and explain what it means for interpretation.
- [ ] Run a hypothesis test on a treatment effect. Report the $p$-value, the 95% CI, and write one sentence correctly interpreting each.
- [ ] Demonstrate the CLT: draw 1,000 samples of size $n=5$ and $n=500$ from a skewed distribution. Plot the sampling distribution of $\bar{X}$ for each. When does it look normal?
- [ ] Fit a simple model and a complex model on the same data. Compare train vs. test performance. Which overfits? By how much?
- [ ] Take a regression with $p = 0.04$ under naive SEs. Recompute with HC3 robust SEs. Does the conclusion change?

<a id="summary"></a>
## Summary + Suggested Readings

This guide covered the statistical foundations that every applied regression and ML analysis depends on:

1. **Hypothesis testing** formalizes "Is this pattern real or noise?" but $p$-values require correct interpretation -- they are not the probability the null is true.
2. **Confidence intervals** are often more informative than $p$-values because they convey sign, magnitude, and precision simultaneously.
3. **Correlation does not imply causation.** Without an identification strategy, regression coefficients are associations, not causal effects.
4. **The LLN and CLT** justify why sample-based inference works -- but they require (approximate) independence, which clustered or time-series data violates.
5. **Multicollinearity** inflates coefficient standard errors without biasing them. Use VIF to diagnose it and regularization or variable selection to address it.
6. **The bias-variance tradeoff** explains why more flexible models are not always better, especially in the moderate-sample-size world of health economics.

The unifying theme: statistical tools are powerful but require careful application. Every number in a regression table (coefficient, SE, $p$-value, CI) tells you something, but only if the underlying assumptions are met.

### Suggested Readings
- Wooldridge, *Introductory Econometrics* -- Chapters 4-5 on inference, Chapter 3 on multicollinearity.
- Angrist & Pischke, *Mostly Harmless Econometrics* -- Chapter 3 on regression and causality.
- Wasserstein & Lazar, "The ASA Statement on p-Values" (2016) -- essential reading on what $p$-values mean.
- James, Witten, Hastie & Tibshirani, *Introduction to Statistical Learning* -- Chapter 2 on bias-variance, Chapter 6 on regularization.
- Greenland et al., "Statistical tests, P values, confidence intervals, and power" (2016) -- practical guide to correct interpretation.
