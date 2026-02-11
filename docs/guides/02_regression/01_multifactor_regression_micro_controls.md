# Guide: 01_multifactor_regression_micro_controls

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/01_multifactor_regression_micro_controls.ipynb`.

Moving from a single regressor to multiple regressors is where applied economics really begins. The central question shifts from "what is the association?" to "what is the association *after accounting for confounders*?" This guide focuses on what happens when you add control variables and how to think about omitted variable bias, partial correlations, and specification choices.

### Key Terms (defined)
- **Omitted variable bias (OVB)**: bias arising when a relevant variable that is correlated with an included regressor is left out of the model.
- **Control variable**: a regressor included to block a confounding pathway, not because its coefficient is of direct interest.
- **Confounder**: a variable that causally affects both the treatment/exposure and the outcome, creating a spurious association.
- **Partial correlation**: the correlation between two variables after removing the linear effect of other variables from both.
- **Short regression**: the regression that omits one or more relevant controls.
- **Long regression**: the regression that includes those controls.
- **VIF (Variance Inflation Factor)**: measures how much a coefficient's variance is inflated by collinearity with other regressors.
- **Mediator**: a variable on the causal pathway between treatment and outcome; controlling for it removes part of the treatment effect.
- **Collider**: a variable caused by both treatment and outcome; conditioning on it introduces spurious association.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math and assumptions for adding controls.
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Choose a set of control variables with a clear economic rationale (write it down before fitting).
- Fit a multivariate OLS model with HC3 robust standard errors.
- Fit the "short" regression (without controls) and the "long" regression (with controls); compare the coefficient of interest.
- Compute the VIF table; identify any regressors with VIF > 5 and explain why.
- Use the OVB formula to predict the *direction* of bias from omitting a specific control; verify against your short vs long comparison.
- Interpret the coefficient of interest in units, with a causal caveat.

### Alternative Example (Not the Notebook Solution)

This example shows how adding controls changes the estimated return to education in a wage regression.

```python
# Multifactor regression: wage ~ education + experience + female
# Demonstrates coefficient change when adding controls
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

rng = np.random.default_rng(42)
n = 500

# Simulate correlated data
ability = rng.normal(size=n)                          # unobserved confounder
education = 12 + 2 * ability + rng.normal(size=n)     # ability -> education
experience = 25 - 0.8 * education + rng.normal(3, size=n)  # correlated with educ
female = rng.binomial(1, 0.5, size=n)
log_wage = (0.08 * education + 0.02 * experience
            - 0.15 * female + 0.05 * ability
            + rng.normal(0, 0.4, size=n))

df = pd.DataFrame({
    'log_wage': log_wage, 'education': education,
    'experience': experience, 'female': female
})

# Short regression (education only)
X_short = sm.add_constant(df[['education']])
res_short = sm.OLS(df['log_wage'], X_short).fit(cov_type='HC3')

# Long regression (add controls)
X_long = sm.add_constant(df[['education', 'experience', 'female']])
res_long = sm.OLS(df['log_wage'], X_long).fit(cov_type='HC3')

print("Short regression — education coeff:", round(res_short.params['education'], 4))
print("Long  regression — education coeff:", round(res_long.params['education'], 4))
# Education coefficient changes because experience and gender were confounders.

# VIF check
vif_df = pd.DataFrame({
    'feature': X_long.columns[1:],
    'VIF': [variance_inflation_factor(X_long.values, i+1)
            for i in range(X_long.shape[1]-1)]
})
print(vif_df)
```

**What to notice:** The education coefficient in the short regression is biased because it partly absorbs the effects of experience and gender (both correlated with education and predictive of wages). Adding controls moves the coefficient toward the true value of 0.08.

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Prerequisites: OLS Foundations (Guide 00)

This guide builds on the OLS foundations covered in [Guide 00](00_single_factor_regression_micro.md). That guide covers the core regression framework: notation and matrix setup, the five classical assumptions (linearity, no perfect multicollinearity, exogeneity, homoskedasticity, no autocorrelation), the OLS derivation via normal equations, coefficient interpretation, robust standard errors (HC3 and the sandwich estimator), hypothesis testing (t-stats, p-values, confidence intervals), and multicollinearity diagnostics (VIF). Read it first if you have not already. This guide focuses on what happens when you add controls to a regression -- and what can go wrong.

### Deep Dive: Omitted Variable Bias (OVB) — when a coefficient absorbs a missing cause

OVB is the simplest and most common reason regression coefficients are misleading.

#### 1) Intuition (plain English)

If you omit a variable that:
1) affects the outcome, and
2) is correlated with an included regressor,
then your estimated coefficient partly picks up the omitted variable's effect.

**Story example:** Education ($x$) and earnings ($y$).
Ability ($z$) affects both schooling and earnings. If you omit ability, the schooling coefficient can be biased upward.

#### 2) Notation + setup (define symbols)

True model:

$$
y = \beta x + \gamma z + \varepsilon
$$

Estimated (misspecified) model that omits $z$:

$$
y = b x + u
$$

**What each term means**
- $y$: outcome.
- $x$: included regressor of interest.
- $z$: omitted regressor (confounder).
- $\varepsilon$: other unobservables uncorrelated with $x$ under ideal assumptions.
- $b$: coefficient you estimate when you omit $z$.

#### 3) Assumptions (when OLS has a causal interpretation)

The key identification condition for OLS is:

$$
\mathbb{E}[\varepsilon \mid x, z] = 0
$$

If you omit $z$, you generally violate:
$$
\mathbb{E}[u \mid x] = 0.
$$

#### 4) Estimation mechanics: deriving the OVB formula

**Derivation sketch:** The true model is $y = \beta x + \gamma z + \varepsilon$. If you omit $z$ and run the short regression $y = bx + u$, OLS gives $b = \beta + \gamma \hat\delta$, where $\hat\delta$ is the coefficient from regressing $z$ on $x$. In population:

$$
\mathbb{E}[b] = \beta + \gamma \frac{\mathrm{Cov}(x,z)}{\mathrm{Var}(x)}.
$$

The bias term $\gamma \cdot \mathrm{Cov}(x,z)/\mathrm{Var}(x)$ combines two things: (1) how much $z$ affects $y$ (via $\gamma$), and (2) how correlated $z$ is with $x$ (via $\mathrm{Cov}(x,z)/\mathrm{Var}(x)$, which is the slope from regressing $z$ on $x$). If either is zero, there is no bias.

**What each term means**
- $\beta$: the "true" causal/structural slope on $x$ (in the true model).
- $\gamma$: effect of omitted variable $z$ on $y$.
- $\mathrm{Cov}(x,z)/\mathrm{Var}(x)$: how much $z$ moves with $x$ (the "regression of z on x" slope).

**Direction-of-bias intuition**
- If $\gamma > 0$ and $\mathrm{Cov}(x,z) > 0$ → upward bias.
- If $\gamma > 0$ and $\mathrm{Cov}(x,z) < 0$ → downward bias.
- Sign flips are possible.

**Numerical example:** Suppose the true return to education is $\beta = 0.05$ (5% per year). Ability ($z$) has a positive effect on wages ($\gamma = 0.03$), and ability is positively correlated with education ($\mathrm{Cov}(x,z)/\mathrm{Var}(x) = 0.8$). Then $\mathbb{E}[b] = 0.05 + 0.03 \times 0.8 = 0.074$. The short regression overestimates the return to education by 48%.

#### 5) Connection to "adding controls changes coefficients"

When you add a control that is correlated with your regressor and predictive of the outcome:
- you are trying to remove a backdoor path (confounding),
- the coefficient can move a lot.

This movement is not a "bug." It is evidence that the omitted variable mattered.

#### 6) Inference: robust SE do not fix OVB

Robust/clustered/HAC standard errors correct uncertainty calculations under dependence/heteroskedasticity.
They do **not** make $\mathbb{E}[u \mid x]=0$ true.

So you can have a "precise" but biased estimate.

#### 7) Diagnostics + robustness (minimum set)

1) **Control sensitivity**
- add plausible controls in a disciplined way; do coefficients stabilize?

2) **Conceptual confounder list**
- write down what could affect both $x$ and $y$ (before running regressions).

3) **Placebo / negative-control outcomes**
- test an outcome that should not be affected by $x$; if you see strong "effects," confounding is likely.

4) **Panel methods / FE**
- if confounding is time-invariant, FE can help; if it is time-varying, FE may not solve it.

#### 8) Interpretation + reporting

When coefficients change with controls:
- report the sequence of specifications ("spec curve" thinking),
- explain which omitted variables the controls are proxying for,
- avoid claiming causality unless you have a design.

**What this does NOT mean**
- "I controlled for a lot of variables" is not a guarantee.
- Over-controlling can also introduce bias if you control for mediators or colliders.

#### Exercises

- [ ] Simulate a confounder $z$ that affects both $x$ and $y$; show how omitting $z$ biases $b$.
- [ ] Use the OVB formula to predict the direction of bias given signs of $\gamma$ and $\mathrm{Cov}(x,z)$.
- [ ] Fit a regression with and without a plausible control in the project data; interpret the change.
- [ ] Write one paragraph: "Which confounders are most plausible here and why?"

### Adding Controls: What Changes and Why

The OVB formula above tells you *how much* a coefficient changes when you add a control. This section covers the practical thinking around *which* controls to add, when to stop, and how to report results honestly.

#### Why coefficients change when you add controls

Adding a control $z$ to a regression of $y$ on $x$ changes the coefficient on $x$ by exactly the OVB term:

$$
\hat\beta_{long} - \hat\beta_{short} \approx -\hat\gamma \cdot \frac{\mathrm{Cov}(x,z)}{\mathrm{Var}(x)}.
$$

If the control is both predictive of the outcome ($\gamma \neq 0$) and correlated with $x$, the coefficient moves. If either condition fails, the coefficient is approximately unchanged. This is why "kitchen sink" regressions (throwing in every available variable) are not always helpful: controls uncorrelated with the regressor of interest do not reduce OVB, and controls that are mediators or colliders can *introduce* bias.

#### Short vs long regression: when to add controls

**Good controls** block confounding pathways (common causes of both $x$ and $y$):
- In a wage regression, controlling for experience removes a confounder because experience is correlated with education and independently affects wages.
- In a treatment effect study, controlling for baseline health removes a confounder if sicker patients are more likely to receive treatment.

**Bad controls** are variables that lie on the causal pathway or are consequences of both treatment and outcome:

1. **Mediators** (on the causal chain from $x$ to $y$): Suppose education increases cognitive skill, which increases wages. If you control for cognitive skill, you remove part of the *true* education effect. The coefficient on education shrinks not because OVB was fixed, but because you blocked the causal mechanism.

2. **Colliders** (caused by both $x$ and $y$): Suppose both talent and family connections get you into a prestigious firm. If you condition on "employed at a prestigious firm," you induce a spurious negative association between talent and family connections within that group. Adding a collider as a control opens a bias pathway that was previously closed.

**Rule of thumb:** Draw a simple causal diagram (even informally) *before* choosing controls. Only control for pre-treatment common causes. Do not control for post-treatment variables unless you have a specific design reason.

#### Specification curve thinking

Because the "right" set of controls is rarely obvious, responsible applied work reports results across multiple reasonable specifications:

1. Start with a baseline (no controls beyond the regressor of interest).
2. Add controls one group at a time (demographics, then economic variables, then institutional variables).
3. Report how the coefficient and its confidence interval move across specifications.
4. If the coefficient is stable across reasonable specs, the result is more credible. If it is highly sensitive, flag that as a limitation.

This is sometimes called a "specification curve" or "multiverse analysis." The key discipline is deciding on the set of specifications *before* looking at results, not cherry-picking the one that tells the best story.

#### Health economics example: adding controls to a treatment effect regression

Suppose you are estimating the effect of a new medication on hospital length of stay (LOS). Your data include patient demographics, comorbidities, and hospital characteristics.

| Specification | Controls added | Medication coeff (days) | 95% CI |
|---|---|---|---|
| 1 | None | -2.1 | [-3.0, -1.2] |
| 2 | + age, sex, BMI | -1.6 | [-2.4, -0.8] |
| 3 | + comorbidity index | -1.3 | [-2.0, -0.6] |
| 4 | + hospital fixed effects | -1.2 | [-1.9, -0.5] |

**Interpretation:** The raw association (Spec 1) overstates the medication's benefit because sicker patients (who have longer stays) are less likely to receive the new drug. Adding patient demographics and comorbidities reduces the coefficient toward its "true" value. Hospital fixed effects absorb cross-hospital variation in treatment protocols. The coefficient stabilizes around -1.2 to -1.3 days across the last two specs, suggesting residual confounding is modest — but cannot rule out unobserved patient severity.

**What you should NOT add:** Post-treatment outcomes like "ICU admission during stay" or "in-hospital complications." These are potential mediators or colliders. If the medication reduces complications, and complications lengthen stays, then controlling for complications removes part of the treatment effect you are trying to measure.

#### Exercises

- [ ] For the project data, draw an informal causal diagram and identify which variables are confounders, which might be mediators, and which might be colliders.
- [ ] Fit 3-4 nested specifications (adding controls stepwise) and report how the coefficient of interest changes.
- [ ] Identify a variable that *should not* be used as a control (mediator or collider) and explain why.

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`) + multicollinearity (`vif_table`)
- `src/macro.py`: GDP + labels (`gdp_growth_*`, `technical_recession_label`)
- `src/evaluation.py`: regression metrics helpers
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Including a mediator as a control and interpreting the attenuated coefficient as "the real effect."
- Including a collider as a control and introducing spurious associations.
- Over-controlling by adding every available variable without a causal rationale.
- Ignoring multicollinearity (high VIF) when adding many correlated controls.
- Claiming causality because "I controlled for confounders" without a formal identification strategy.
- Reporting only the specification with the largest or most significant coefficient.

<a id="summary"></a>
## Summary + Suggested Readings

After working through this guide you should be able to:
- explain OVB using the formula and predict the direction of bias,
- distinguish good controls (confounders) from bad controls (mediators, colliders),
- compare short vs long regressions and interpret coefficient changes,
- report results across multiple specifications honestly, and
- recognize when "adding controls" is not enough and a causal design is needed.

Suggested readings:
- Angrist & Pischke: *Mostly Harmless Econometrics*, Ch. 3 (OVB, short vs long regression)
- Cinelli, Forney & Pearl: "A Crash Course in Good and Bad Controls" (2022) — formal treatment of mediators/colliders
- Wooldridge: *Introductory Econometrics*, Ch. 3 (multiple regression, OVB)
- Simonsohn, Simmons & Nelson: "Specification Curve Analysis" (2020) — multiverse reporting
