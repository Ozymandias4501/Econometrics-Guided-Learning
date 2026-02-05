### Deep Dive: Omitted Variable Bias (OVB) — when a coefficient absorbs a missing cause

OVB is the simplest and most common reason regression coefficients are misleading.

#### 1) Intuition (plain English)

If you omit a variable that:
1) affects the outcome, and
2) is correlated with an included regressor,
then your estimated coefficient partly picks up the omitted variable’s effect.

**Story example:** Education ($x$) and earnings ($y$).  
Ability ($z$) affects both schooling and earnings. If you omit ability, the schooling coefficient can be biased upward.

#### 2) Notation + setup (define symbols)

True model:

$$
y = \\beta x + \\gamma z + \\varepsilon
$$

Estimated (misspecified) model that omits $z$:

$$
y = b x + u
$$

**What each term means**
- $y$: outcome.
- $x$: included regressor of interest.
- $z$: omitted regressor (confounder).
- $\\varepsilon$: other unobservables uncorrelated with $x$ under ideal assumptions.
- $b$: coefficient you estimate when you omit $z$.

#### 3) Assumptions (when OLS has a causal interpretation)

The key identification condition for OLS is:

$$
\\mathbb{E}[\\varepsilon \\mid x, z] = 0
$$

If you omit $z$, you generally violate:
$$
\\mathbb{E}[u \\mid x] = 0.
$$

#### 4) Estimation mechanics: deriving the OVB formula

Under standard assumptions, the expected value of the omitted-variable coefficient is:

$$
\\mathbb{E}[b] = \\beta + \\gamma \\frac{\\mathrm{Cov}(x,z)}{\\mathrm{Var}(x)}.
$$

**What each term means**
- $\\beta$: the “true” causal/structural slope on $x$ (in the true model).
- $\\gamma$: effect of omitted variable $z$ on $y$.
- $\\mathrm{Cov}(x,z)/\\mathrm{Var}(x)$: how much $z$ moves with $x$ (the “regression of z on x” slope).

**Direction-of-bias intuition**
- If $\\gamma > 0$ and $\\mathrm{Cov}(x,z) > 0$ → upward bias.
- If $\\gamma > 0$ and $\\mathrm{Cov}(x,z) < 0$ → downward bias.
- Sign flips are possible.

#### 5) Connection to “adding controls changes coefficients”

When you add a control that is correlated with your regressor and predictive of the outcome:
- you are trying to remove a backdoor path (confounding),
- the coefficient can move a lot.

This movement is not a “bug.” It is evidence that the omitted variable mattered.

#### 6) Inference: robust SE do not fix OVB

Robust/clustered/HAC standard errors correct uncertainty calculations under dependence/heteroskedasticity.
They do **not** make $\\mathbb{E}[u \\mid x]=0$ true.

So you can have a “precise” but biased estimate.

#### 7) Diagnostics + robustness (minimum set)

1) **Control sensitivity**
- add plausible controls in a disciplined way; do coefficients stabilize?

2) **Conceptual confounder list**
- write down what could affect both $x$ and $y$ (before running regressions).

3) **Placebo / negative-control outcomes**
- test an outcome that should not be affected by $x$; if you see strong “effects,” confounding is likely.

4) **Panel methods / FE**
- if confounding is time-invariant, FE can help; if it is time-varying, FE may not solve it.

#### 8) Interpretation + reporting

When coefficients change with controls:
- report the sequence of specifications (“spec curve” thinking),
- explain which omitted variables the controls are proxying for,
- avoid claiming causality unless you have a design.

**What this does NOT mean**
- “I controlled for a lot of variables” is not a guarantee.
- Over-controlling can also introduce bias if you control for mediators or colliders.

#### Exercises

- [ ] Simulate a confounder $z$ that affects both $x$ and $y$; show how omitting $z$ biases $b$.
- [ ] Use the OVB formula to predict the direction of bias given signs of $\\gamma$ and $\\mathrm{Cov}(x,z)$.
- [ ] Fit a regression with and without a plausible control in the project data; interpret the change.
- [ ] Write one paragraph: “Which confounders are most plausible here and why?”
