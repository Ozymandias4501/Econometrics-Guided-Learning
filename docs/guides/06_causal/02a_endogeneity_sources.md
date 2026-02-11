# Guide: 02a_endogeneity_sources

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/07_causal/02a_endogeneity_sources.ipynb`.

Endogeneity -- the correlation of a regressor with the error term -- is the central problem of causal econometrics. When exogeneity fails, OLS coefficients no longer have a causal interpretation and may not even be consistent estimates of population parameters. This module catalogs the three main sources of endogeneity (omitted variables, measurement error, and simultaneity), derives their bias formulas, and shows how to diagnose and sign the bias before reaching for identification strategies like IV or natural experiments.

### Key Terms (defined)
- **Endogeneity**: $\mathrm{Cov}(x, \varepsilon) \neq 0$; a regressor is correlated with the error term.
- **Exogeneity**: $\mathbb{E}[\varepsilon \mid X] = 0$; the error is mean-independent of the regressors.
- **Omitted variable bias (OVB)**: bias from excluding a relevant variable that is correlated with both the outcome and an included regressor.
- **Measurement error**: when the observed regressor $x^*$ differs from the true regressor $x$; classical measurement error attenuates the coefficient toward zero.
- **Simultaneity (reverse causality)**: when the outcome $y$ and the regressor $x$ are jointly determined.
- **Attenuation bias**: the systematic downward bias (toward zero) caused by classical measurement error.
- **Sign the bias**: using the OVB formula to determine the direction of bias, even when the magnitude is unknown.
- **Identification**: the conditions under which a causal parameter can be recovered from the data.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Simulate and demonstrate omitted variable bias
- Complete notebook section: Derive and verify the OVB formula
- Complete notebook section: Simulate classical measurement error and attenuation bias
- Complete notebook section: Simulate simultaneity bias
- Complete notebook section: Sign-the-bias exercise for a real-world example
- Produce at least one simulation comparing the biased OLS estimate with the true parameter.
- Write a bias table for a research question of your choice, signing the direction of each potential bias.

### Alternative Example (Not the Notebook Solution)
```python
# Toy OVB demonstration (not the notebook data):
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(0)
n = 1000
ability = rng.normal(size=n)           # omitted variable
educ = 2 + 0.5 * ability + rng.normal(size=n)  # correlated with ability
wage = 5 + 3 * educ + 2 * ability + rng.normal(size=n)  # true beta_educ = 3

# Short regression (omits ability)
X_short = sm.add_constant(educ)
res_short = sm.OLS(wage, X_short).fit()

# Long regression (includes ability)
X_long = sm.add_constant(np.column_stack([educ, ability]))
res_long = sm.OLS(wage, X_long).fit()

print(f"Short reg (biased): beta_educ = {res_short.params[1]:.3f}")
print(f"Long reg (unbiased): beta_educ = {res_long.params[1]:.3f}")
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: OVB formula and sign-the-bias

Omitted variable bias is the most common source of endogeneity in applied work. The OVB formula is the single most useful diagnostic tool in econometrics.

#### 1) Intuition (plain English)

Suppose the true model is:

$$
y = \beta_0 + \beta_1 x + \beta_2 z + \varepsilon
$$

but you estimate the short regression omitting $z$:

$$
y = \tilde\beta_0 + \tilde\beta_1 x + \tilde\varepsilon.
$$

If $z$ is correlated with both $y$ (through $\beta_2 \neq 0$) and $x$, then $\tilde\beta_1 \neq \beta_1$. The OLS estimate absorbs the effect of $z$ that runs through $x$.

#### 2) The OVB formula

The bias in the short regression estimator is:

$$
\tilde\beta_1 = \beta_1 + \beta_2 \cdot \delta_1
$$

where $\delta_1$ is the coefficient from regressing the omitted variable $z$ on the included regressor $x$:

$$
z = \delta_0 + \delta_1 x + \text{error}.
$$

In words:

$$
\text{Bias} = \underbrace{\beta_2}_{\text{effect of } z \text{ on } y} \times \underbrace{\delta_1}_{\text{association of } z \text{ with } x}.
$$

#### 3) Sign-the-bias table

You can determine the *direction* of bias even without observing $z$:

| $\beta_2$ (effect of omitted on $y$) | $\delta_1$ (correlation of omitted with $x$) | Direction of bias on $\tilde\beta_1$ |
|---|---|---|
| Positive | Positive | Upward (overestimate) |
| Positive | Negative | Downward (underestimate) |
| Negative | Positive | Downward (underestimate) |
| Negative | Negative | Upward (overestimate) |

**Example**: estimating the return to education on wages.
- Omitted variable: ability ($z$).
- $\beta_2 > 0$: ability raises wages.
- $\delta_1 > 0$: ability is positively correlated with education.
- Bias is *upward*: the naive return to education overestimates the true causal effect.

#### 4) OVB with multiple omitted variables

With multiple omitted variables $z_1, z_2, \dots$, the total bias is the sum of individual biases:

$$
\text{Bias} = \sum_j \beta_{z_j} \cdot \delta_{z_j}
$$

where $\delta_{z_j}$ is the partial association of $z_j$ with $x$ (not the marginal association). This makes sign-the-bias harder with multiple omitted variables because the partial correlations can differ from marginal ones.

#### 5) Simulation verification

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(42)
n = 5000
z = rng.normal(size=n)
x = 1 + 0.6 * z + rng.normal(size=n)  # delta_1 = 0.6
y = 2 + 3 * x + 1.5 * z + rng.normal(size=n)  # beta_1 = 3, beta_2 = 1.5

# Short regression
res_short = sm.OLS(y, sm.add_constant(x)).fit()
print(f"Short beta_1: {res_short.params[1]:.3f}")
print(f"Expected bias: {1.5 * 0.6:.3f}")
print(f"Expected short beta_1: {3 + 1.5 * 0.6:.3f}")
```

#### Exercises

- [ ] Derive the OVB formula starting from the short and long regression normal equations.
- [ ] For a research question of your choice, list three plausible omitted variables and sign the bias for each.
- [ ] Simulate OVB with a known DGP. Verify numerically that $\tilde\beta_1 = \beta_1 + \beta_2 \cdot \delta_1$.
- [ ] Show that adding a "bad control" (a mediator or collider) can *create* bias rather than remove it.

---

### Deep Dive: Attenuation bias from measurement error

Classical measurement error is pervasive in economic data (self-reported income, survey hours, proxy variables). Its signature is a systematic downward bias.

#### 1) Intuition (plain English)

Suppose you want to estimate how $x$ affects $y$, but instead of observing $x$, you observe $x^* = x + u$ where $u$ is random noise. The noise in $x^*$ acts like adding a random variable that is uncorrelated with $y$ -- it dilutes the signal, pulling the estimated coefficient toward zero.

#### 2) The classical measurement error model

True model: $y = \beta_0 + \beta_1 x + \varepsilon$.

Observed: $x^* = x + u$, where $u \perp x$, $u \perp \varepsilon$.

Substituting:

$$
y = \beta_0 + \beta_1 (x^* - u) + \varepsilon = \beta_0 + \beta_1 x^* + (\varepsilon - \beta_1 u).
$$

The new error $(\varepsilon - \beta_1 u)$ is correlated with $x^* = x + u$ through $u$. This creates endogeneity.

#### 3) The attenuation bias formula

$$
\text{plim}\,\hat\beta_1^{OLS} = \beta_1 \cdot \frac{\sigma_x^2}{\sigma_x^2 + \sigma_u^2} = \beta_1 \cdot \lambda
$$

where $\lambda = \sigma_x^2 / (\sigma_x^2 + \sigma_u^2)$ is the reliability ratio, always between 0 and 1.

Key implications:
- The OLS estimate is biased toward zero (attenuated).
- The bias is worse when measurement error is large relative to the true variation ($\sigma_u^2 / \sigma_x^2$ is large).
- With perfect measurement ($\sigma_u^2 = 0$), $\lambda = 1$ and there is no bias.

#### 4) Measurement error in the dependent variable

If $y^* = y + w$ (error in $y$, not $x$):

$$
y^* = \beta_0 + \beta_1 x + (\varepsilon + w).
$$

This does *not* cause bias (assuming $w \perp x$). It only increases the variance of the error, making estimates noisier but still unbiased. This asymmetry is important: measurement error in $y$ is relatively benign; measurement error in $x$ is not.

#### 5) Simulation

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(42)
n = 2000
x_true = rng.normal(0, 1, n)
y = 1 + 2 * x_true + rng.normal(0, 1, n)

# Add measurement error to x
sigma_u = 1.0
x_observed = x_true + rng.normal(0, sigma_u, n)

# Reliability ratio
lam = 1.0 / (1.0 + sigma_u**2)
print(f"Expected attenuation: beta * lambda = {2 * lam:.3f}")

res = sm.OLS(y, sm.add_constant(x_observed)).fit()
print(f"OLS with mismeasured x: {res.params[1]:.3f}")
```

#### 6) Solutions to measurement error

- **Instrumental variables**: find an instrument $z$ correlated with $x$ but not with $u$.
- **Two measurements**: if you have two noisy measures $x_1^* = x + u_1$ and $x_2^* = x + u_2$, use one as an instrument for the other.
- **Bounds**: if you cannot find an IV, you can bound the true $\beta_1$ between the OLS estimate (lower bound in absolute value) and the reverse regression estimate.

#### Exercises

- [ ] Simulate classical measurement error and verify the attenuation formula $\hat\beta_1 \approx \beta_1 \cdot \lambda$.
- [ ] Increase $\sigma_u^2$ and show that the bias worsens (the coefficient shrinks further toward zero).
- [ ] Add measurement error to $y$ instead and show that OLS remains unbiased (just noisier).
- [ ] Use an IV approach with a second noisy measure to recover the true $\beta_1$.

---

### Deep Dive: Simultaneity and identification

Simultaneity arises when $x$ causes $y$ and $y$ causes $x$. This feedback loop makes OLS inconsistent for either causal direction.

#### 1) Intuition (plain English)

Classic example: supply and demand. The price you observe in the market is determined simultaneously by both supply and demand. If you regress quantity on price, you estimate neither the supply curve nor the demand curve -- you get a meaningless blend of both.

#### 2) A simultaneous equations model

$$
y_1 = \alpha_1 y_2 + \beta_1 x_1 + \varepsilon_1 \quad \text{(demand)}
$$
$$
y_2 = \alpha_2 y_1 + \beta_2 x_2 + \varepsilon_2 \quad \text{(supply)}
$$

where $y_1$ and $y_2$ are jointly determined (e.g., quantity and price), and $x_1, x_2$ are exogenous shifters.

Solving for the reduced form:

$$
y_1 = \pi_{10} + \pi_{11} x_1 + \pi_{12} x_2 + v_1
$$
$$
y_2 = \pi_{20} + \pi_{21} x_1 + \pi_{22} x_2 + v_2
$$

The reduced form can be estimated consistently by OLS, but the structural parameters $(\alpha_1, \alpha_2)$ require identification.

#### 3) The identification problem

To identify the demand equation, you need at least one variable that shifts supply ($x_2$) but does not directly appear in the demand equation. This exclusion restriction provides the instrument for 2SLS.

**Order condition** (necessary): the equation must exclude at least as many exogenous variables as there are endogenous regressors on the right-hand side.

**Rank condition** (necessary and sufficient): the excluded instruments must have a non-zero effect on the endogenous regressor (relevance).

#### 4) OLS bias under simultaneity

In the simple simultaneous model $y_1 = \alpha y_2 + \varepsilon_1$ and $y_2 = \gamma y_1 + \varepsilon_2$:

$$
\text{plim}\,\hat\alpha_{OLS} = \alpha + \frac{(1 - \alpha\gamma)\,\mathrm{Cov}(y_2, \varepsilon_1)}{\mathrm{Var}(y_2)}
$$

Since $y_2$ depends on $y_1$ which depends on $\varepsilon_1$, we have $\mathrm{Cov}(y_2, \varepsilon_1) \neq 0$. The bias does not vanish as $n \to \infty$ -- it is an inconsistency.

#### 5) Simulation

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(42)
n = 5000

# Structural model: y1 = 0.5*y2 + e1, y2 = 0.3*y1 + e2
e1 = rng.normal(size=n)
e2 = rng.normal(size=n)

# Reduced form (solve the system)
# y1 = 0.5*y2 + e1 and y2 = 0.3*y1 + e2
# => y1 = 0.5*(0.3*y1 + e2) + e1 => y1(1 - 0.15) = e1 + 0.5*e2
y1 = (e1 + 0.5 * e2) / (1 - 0.15)
y2 = (e2 + 0.3 * e1) / (1 - 0.15)

# OLS of y1 on y2 (biased)
res = sm.OLS(y1, sm.add_constant(y2)).fit()
print(f"True alpha = 0.5, OLS estimate = {res.params[1]:.3f}")
```

#### 6) Solutions to simultaneity

- **Instrumental variables / 2SLS**: use an exogenous shifter of $y_2$ as an instrument (see guide 03).
- **Recursive systems**: if the model is triangular (no feedback), OLS on each equation is consistent.
- **Timing/lags**: use lagged values as instruments under the assumption that past values are predetermined.
- **Natural experiments**: find exogenous variation that shifts one equation but not the other.

#### Exercises

- [ ] Simulate a simultaneous equations model and show that OLS is biased in both directions.
- [ ] Derive the reduced form from the structural equations and estimate it with OLS. Verify consistency.
- [ ] Add an exogenous instrument (supply shifter) and use 2SLS to recover the demand parameter consistently.
- [ ] Explain in words why lagged values of an endogenous variable can serve as instruments under certain assumptions, and when this fails.

### Project Code Map
- `src/causal.py`: IV estimation (`fit_iv_2sls`), panel setup (`to_panel_index`)
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`)
- `src/evaluation.py`: regression metrics helpers (`regression_metrics`)
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `linearmodels.iv`: `IV2SLS` for instrumental variables estimation

### Common Mistakes
- Confusing OVB (omitting a confounder) with bad-control bias (conditioning on a mediator or collider).
- Assuming the sign of OVB is obvious without working through the formula; partial correlations can differ from marginal ones.
- Forgetting that classical measurement error in $y$ does *not* bias OLS, while measurement error in $x$ does.
- Treating simultaneity as a finite-sample problem; it is an inconsistency that does not vanish with more data.
- Using OLS on a simultaneous system and interpreting the result as a structural parameter.

<a id="summary"></a>
## Summary + Suggested Readings

Endogeneity is the barrier between association and causation. You should now be able to:
- apply the OVB formula and sign the bias for a given research question,
- recognize and quantify attenuation bias from classical measurement error,
- understand why simultaneity creates inconsistency, and
- articulate the identification conditions needed to recover causal parameters.

Suggested readings:
- Wooldridge, *Introductory Econometrics*, Ch. 3 (OVB), Ch. 9 (measurement error), Ch. 15 (simultaneous equations)
- Angrist & Pischke, *Mostly Harmless Econometrics*, Ch. 3 (OVB, bad controls) and Ch. 4 (IV)
- Bound, Jaeger & Baker (1995), "Problems with Instrumental Variables Estimation When the Correlation Between the Instruments and the Endogenous Explanatory Variable Is Weak" (*JASA*)
- Stock & Watson, *Introduction to Econometrics*, Ch. 9 (threats to internal validity)
