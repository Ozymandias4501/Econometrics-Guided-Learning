# Guide: 02a_functional_forms_and_interactions

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/02a_functional_forms_and_interactions.ipynb`.

OLS is linear in *parameters*, not in *variables*. That distinction is the key to a huge family of nonlinear relationships that remain estimable by OLS once you transform the right-hand-side (or left-hand-side) variables. This module covers log transforms, polynomials, interaction terms, and dummy variables -- the bread and butter of applied econometric specification.

### Key Terms (defined)
- **Functional form**: the mathematical shape linking regressors to the outcome (level-level, log-level, log-log, quadratic, etc.).
- **Log-level model**: $\ln y = \beta_0 + \beta_1 x + \varepsilon$; a one-unit change in $x$ is associated with an approximate $100 \beta_1$% change in $y$.
- **Level-log model**: $y = \beta_0 + \beta_1 \ln x + \varepsilon$; a 1% change in $x$ is associated with a $\beta_1 / 100$ unit change in $y$.
- **Log-log (elasticity) model**: $\ln y = \beta_0 + \beta_1 \ln x + \varepsilon$; $\beta_1$ is the elasticity of $y$ with respect to $x$.
- **Quadratic term**: $x^2$ included alongside $x$ to capture diminishing or increasing marginal effects.
- **Interaction term**: the product $x_1 \cdot x_2$, allowing the marginal effect of $x_1$ to depend on the level of $x_2$.
- **Dummy variable (indicator)**: a 0/1 variable representing a category; shifts the intercept.
- **Marginal effect**: $\partial E[y \mid X] / \partial x_j$, which may depend on the values of other regressors when interactions or polynomials are present.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Create log-transformed variables
- Complete notebook section: Estimate log-level, level-log, and log-log models
- Complete notebook section: Add quadratic terms and test joint significance
- Complete notebook section: Construct and interpret interaction terms
- Complete notebook section: Include dummy variables and interpret the shift
- Produce a marginal-effect plot for at least one nonlinear specification.
- Compare nested models using F-tests or information criteria.

### Alternative Example (Not the Notebook Solution)
```python
# Toy interaction model (not the notebook data):
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(42)
n = 300
female = rng.binomial(1, 0.5, n)
educ = rng.normal(12, 3, n)
wage = 5 + 0.8 * educ + 2.0 * female + 0.5 * female * educ + rng.normal(0, 3, n)

df = pd.DataFrame({'wage': wage, 'educ': educ, 'female': female})
df['female_x_educ'] = df['female'] * df['educ']
X = sm.add_constant(df[['educ', 'female', 'female_x_educ']])
res = sm.OLS(df['wage'], X).fit(cov_type='HC3')
print(res.summary())
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: Interpretation table for all functional form families

Choosing the right functional form is one of the most consequential specification decisions. The table below summarizes the main families.

#### 1) Intuition (plain English)

The "linear" in linear regression refers to the *parameters*, not the variables. You can take logs, squares, cubes, or products of your $x$ variables and still estimate the model with OLS. The interpretation of $\beta$ changes depending on which transformations you apply.

#### 2) The interpretation table

| Model | Equation | Interpretation of $\beta_1$ |
|---|---|---|
| Level-level | $y = \beta_0 + \beta_1 x$ | A 1-unit increase in $x$ is associated with a $\beta_1$-unit change in $y$. |
| Log-level | $\ln y = \beta_0 + \beta_1 x$ | A 1-unit increase in $x$ is associated with an approx. $100\beta_1$% change in $y$. |
| Level-log | $y = \beta_0 + \beta_1 \ln x$ | A 1% increase in $x$ is associated with a $\beta_1 / 100$ unit change in $y$. |
| Log-log | $\ln y = \beta_0 + \beta_1 \ln x$ | A 1% increase in $x$ is associated with a $\beta_1$% change in $y$ (elasticity). |
| Quadratic | $y = \beta_0 + \beta_1 x + \beta_2 x^2$ | Marginal effect: $\partial y / \partial x = \beta_1 + 2\beta_2 x$ (varies with $x$). |

#### 3) Log approximation and when it breaks down

The log-level interpretation uses the approximation:

$$
\Delta \ln y \approx \frac{\Delta y}{y}
$$

This is accurate for small changes ($|\Delta \ln y| < 0.1$). For large coefficients, use the exact formula:

$$
\%\Delta y = 100 \times (e^{\beta_1} - 1).
$$

For example, if $\hat{\beta}_1 = 0.20$, the exact percentage change is $100 \times (e^{0.20} - 1) = 22.1\%$, not 20%.

#### 4) Quadratic terms: finding the turning point

When the model is $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \varepsilon$, the marginal effect of $x$ is:

$$
\frac{\partial y}{\partial x} = \beta_1 + 2\beta_2 x.
$$

Setting this to zero gives the turning point:

$$
x^* = -\frac{\beta_1}{2\beta_2}.
$$

If $\beta_2 < 0$, the relationship is an inverted-U (concave); if $\beta_2 > 0$, it is U-shaped (convex). Always check that $x^*$ falls within the range of your data.

#### 5) Practical code for log models

```python
import numpy as np
import statsmodels.api as sm

# log-log model
df['ln_y'] = np.log(df['y'])
df['ln_x'] = np.log(df['x'])
X = sm.add_constant(df[['ln_x']])
res_loglog = sm.OLS(df['ln_y'], X).fit()
# beta_1 is the elasticity
```

#### 6) Testing functional form: nested F-test

To test whether the quadratic term is needed, compare:

- Restricted: $y = \beta_0 + \beta_1 x$
- Unrestricted: $y = \beta_0 + \beta_1 x + \beta_2 x^2$

The F-statistic for the restriction $H_0: \beta_2 = 0$:

$$
F = \frac{(SSR_R - SSR_U)/q}{SSR_U/(n - k - 1)}
$$

where $q = 1$ (one restriction). This is equivalent to the t-test on $\beta_2$ when there is a single restriction.

#### Exercises

- [ ] Estimate a log-level wage equation and interpret the coefficient on years of education.
- [ ] Compute the exact percentage effect using $100(e^{\hat\beta} - 1)$ and compare to the $100\hat\beta$ approximation.
- [ ] Fit a quadratic model for experience on wages and compute the turning point. Does it fall within the data?
- [ ] Conduct an F-test comparing a linear-only model to one with a quadratic term.

---

### Deep Dive: Interaction terms and marginal effects

Interaction terms allow the effect of one variable to depend on the level of another. They are essential for testing heterogeneous effects and are the building blocks of difference-in-differences designs.

#### 1) Intuition (plain English)

An interaction asks: "Does the relationship between $x_1$ and $y$ differ across groups or across values of $x_2$?" Without an interaction term, the model forces the slope on $x_1$ to be the same everywhere.

#### 2) Notation and setup

The model with a continuous-binary interaction:

$$
y_i = \beta_0 + \beta_1 x_{i} + \beta_2 D_{i} + \beta_3 (x_{i} \cdot D_{i}) + \varepsilon_i
$$

where $D_i \in \{0, 1\}$ is a dummy variable.

**Marginal effects by group:**
- When $D = 0$: $\partial y / \partial x = \beta_1$.
- When $D = 1$: $\partial y / \partial x = \beta_1 + \beta_3$.

So $\beta_3$ measures the *difference in slopes* between the two groups.

**Intercepts by group:**
- When $D = 0$: intercept is $\beta_0$.
- When $D = 1$: intercept is $\beta_0 + \beta_2$.

So $\beta_2$ measures the *difference in intercepts* (level shift) when $x = 0$.

#### 3) Continuous-continuous interactions

$$
y_i = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 (x_1 \cdot x_2) + \varepsilon_i
$$

Marginal effect of $x_1$:

$$
\frac{\partial y}{\partial x_1} = \beta_1 + \beta_3 x_2.
$$

This means the marginal effect of $x_1$ is a linear function of $x_2$. You must evaluate it at specific values of $x_2$ (e.g., mean, 25th percentile, 75th percentile) and report those, rather than just $\beta_1$ alone.

#### 4) The "include main effects" rule

When you include $x_1 \cdot x_2$, you must also include $x_1$ and $x_2$ individually. Omitting main effects forces the interaction coefficient to absorb both the interaction and the omitted main effects, producing biased and uninterpretable estimates.

#### 5) Marginal effect plots

For continuous interactions, the standard reporting tool is a marginal effect plot:

```python
import numpy as np
import matplotlib.pyplot as plt

# Suppose beta_1, beta_3 are estimated, and se_marginal is computed
x2_grid = np.linspace(df['x2'].min(), df['x2'].max(), 100)
marginal_effect = beta_1 + beta_3 * x2_grid

# Delta-method SE for marginal effect at each x2 value:
# Var(beta_1 + beta_3 * x2) = Var(beta_1) + x2^2 * Var(beta_3) + 2*x2*Cov(beta_1, beta_3)
V = res.cov_params()
se = np.sqrt(V.loc['x1','x1'] + x2_grid**2 * V.loc['x1_x2','x1_x2']
             + 2 * x2_grid * V.loc['x1','x1_x2'])

plt.fill_between(x2_grid, marginal_effect - 1.96*se, marginal_effect + 1.96*se, alpha=0.2)
plt.plot(x2_grid, marginal_effect)
plt.axhline(0, ls='--', color='grey')
plt.xlabel('x2')
plt.ylabel('Marginal effect of x1')
plt.title('Marginal effect of x1 as a function of x2')
plt.show()
```

#### 6) Dummy variable trap

When including a set of $K$ mutually exclusive dummies (e.g., regions), you must omit one category (the reference category) if you have an intercept. Otherwise $X'X$ is singular (perfect multicollinearity). The coefficients on the included dummies measure the *difference* relative to the omitted category.

#### Exercises

- [ ] Fit a model with an education-gender interaction and interpret $\beta_3$ in words.
- [ ] Construct a marginal effect plot for a continuous-continuous interaction, including 95% confidence bands via the delta method.
- [ ] Show what happens if you include an interaction but omit the main effects -- compare the coefficients.
- [ ] Include a set of region dummies (dropping one reference) and test their joint significance with an F-test.

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`), design matrix (`design_matrix`), multicollinearity (`vif_table`)
- `src/features.py`: feature engineering helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: regression metrics + splits (`time_train_test_split_index`, `regression_metrics`)
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)

### Common Mistakes
- Interpreting $\beta_1$ in an interaction model as "the effect of $x_1$" without noting it is conditional on $x_2 = 0$.
- Using the $100\beta$ approximation when $|\beta|$ is large; use $100(e^\beta - 1)$ instead.
- Omitting main effects when including an interaction, leading to biased interaction coefficients.
- Including a full set of $K$ dummies with an intercept (dummy variable trap).
- Forgetting that the marginal effect in a quadratic or interaction model varies with the data; reporting a single number is incomplete.

<a id="summary"></a>
## Summary + Suggested Readings

Functional forms extend OLS to a rich set of nonlinear relationships without leaving the linear-in-parameters framework. You should now be able to:
- choose and interpret log, quadratic, interaction, and dummy variable specifications,
- compute and plot marginal effects that vary across the data, and
- test whether additional terms improve the model using F-tests.

Suggested readings:
- Wooldridge, *Introductory Econometrics*, Ch. 6 (functional forms) and Ch. 7 (qualitative information)
- Angrist & Pischke, *Mostly Harmless Econometrics*, Ch. 3.3 (nonlinear CEF and regression)
- Brambor, Clark & Golder (2006), "Understanding Interaction Models: Improving Empirical Analyses" (*Political Analysis*)
