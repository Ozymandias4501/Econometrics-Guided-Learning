# Guide: 00_predictive_vs_causal_dags

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/05_causal_inference/00_predictive_vs_causal_dags.ipynb`.

It is the conceptual entry point to causal inference in this curriculum. Earlier sections answered predictive questions (how does $x$ correlate with $y$ in the data?). This part answers interventional questions (what would happen to $y$ if we changed $x$?). Those are different questions; an OLS coefficient is not automatically the answer to either.

### Key Terms (defined)
- **Predictive estimand**: the conditional expectation $\mathbb{E}[Y \mid X = x]$. Answer to 'what should I expect $Y$ to be when I observe $X = x$?'
- **Causal estimand**: the difference $\mathbb{E}[Y(x_1)] - \mathbb{E}[Y(x_0)]$, where $Y(\cdot)$ are potential outcomes. Answer to 'what would happen to $Y$ if we set $X = x_1$ vs $X = x_0$?'
- **Confounder**: a common cause of treatment and outcome. Including it in the regression removes bias.
- **Mediator**: a variable on the causal path from treatment to outcome. Including it changes what you measure (direct vs total effect).
- **Collider (or 'common effect')**: a variable caused by both treatment and outcome (or by treatment and a confounder of the outcome). Including it *creates* spurious correlation.
- **Backdoor path**: any non-causal path from treatment to outcome that runs through a confounder. The 'backdoor criterion' identifies which sets of variables block all such paths.
- **DAG (directed acyclic graph)**: a picture of your causal assumptions. Nodes are variables, arrows are direct causal effects, no cycles.
- **Identification**: a condition under which a causal estimand is recovered exactly from observational data. Different research designs (RCT, IV, DiD, RD) have different identification arguments.

### How To Read This Guide
- Use **Step-by-Step** for the notebook checklist and a hands-on simulation pattern.
- Use **Technical Explanations** for the OVB formula, the collider mechanic, and a working taxonomy of controls.
- For DiD specifically — the most common research design used by working economists — see [Guide 01](01_difference_in_differences.md).

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- [ ] Sketch a DAG (in plain markdown, arrows like `A -> B`) for one applied question you care about.
- [ ] Simulate panel data with one treatment, one outcome, one confounder; recover the omitted-variable bias.
- [ ] Verify the OVB formula numerically: bias = (effect of confounder on outcome) × (slope of confounder regressed on treatment).
- [ ] Simulate a collider DAG; demonstrate that conditioning on the collider creates spurious association.
- [ ] Pick one regression you ran earlier in the curriculum and classify each control as confounder, mediator, collider, or post-treatment.

### Alternative Example: Bivariate OVB Formula

In a bivariate setting (one regressor + one omitted), the omitted-variable bias formula is exact and clean:

$$
\hat\beta_T^{\text{naive}} \xrightarrow{p} \beta_T + \beta_C \cdot \frac{\mathrm{Cov}(T, C)}{\mathrm{Var}(T)}
$$

where $\beta_T$ is the true causal effect, $\beta_C$ is the (true, partialled-out) effect of the confounder $C$ on the outcome, and the second factor is the slope from regressing $C$ on $T$.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)
n = 5000
C = rng.normal(size=n)
T = 0.8 * C + rng.normal(scale=0.5, size=n)
Y = 1.0 * T + 1.5 * C + rng.normal(scale=0.5, size=n)

df = pd.DataFrame({'Y': Y, 'T': T, 'C': C})
naive = sm.OLS(df['Y'], sm.add_constant(df[['T']], has_constant='add')).fit()
adj = sm.OLS(df['Y'], sm.add_constant(df[['T', 'C']], has_constant='add')).fit()
C_on_T = sm.OLS(df['C'], sm.add_constant(df[['T']], has_constant='add')).fit()

slope_C_on_T = C_on_T.params['T']
implied_bias = slope_C_on_T * adj.params['C']
empirical_bias = naive.params['T'] - 1.0  # 1.0 is the true effect
print(f'implied:   {implied_bias:.4f}')
print(f'empirical: {empirical_bias:.4f}')
```

You should see implied ≈ empirical to two decimal places.

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Why predictive ≠ causal: the potential outcomes framing

Define for each unit $i$:
- $Y_i(1)$: outcome we would observe if unit $i$ received treatment.
- $Y_i(0)$: outcome we would observe if unit $i$ did not.

The **individual treatment effect** is $Y_i(1) - Y_i(0)$. The **average treatment effect** is $\mathbb{E}[Y_i(1) - Y_i(0)]$.

The fundamental problem: for any unit, you observe at most one of $Y_i(1)$ or $Y_i(0)$. The other is the **counterfactual**. You can never see the same person both treated and untreated at the same time.

A regression of $Y$ on $T$ identifies the ATE only under specific conditions. The simplest sufficient condition is **conditional ignorability**:

$$
\{Y_i(1), Y_i(0)\} \perp T_i \mid X_i.
$$

In words: conditional on observed covariates $X$, treatment is as good as random. If this holds, OLS of $Y$ on $T$ with $X$ as controls recovers the ATE. If it fails — e.g., because of an unobserved confounder — OLS does not.

Most observational data does not satisfy conditional ignorability. The job of a research design (DiD, IV, RD) is to find structures in the data where some weaker condition does hold.

### DAG primer: chains, forks, colliders

A DAG encodes the *direct* causal effects assumed by the researcher. Three building blocks:

**Chain** $A \to B \to C$: $B$ is a mediator. $A$ affects $C$ only through $B$.
- Unconditional: $A$ and $C$ are correlated.
- Conditional on $B$: $A$ and $C$ are independent.

**Fork (confounder)** $A \leftarrow C \to B$: $C$ is a common cause.
- Unconditional: $A$ and $B$ are correlated even though neither causes the other.
- Conditional on $C$: $A$ and $B$ are independent.

**Collider** $A \to C \leftarrow B$: $C$ is a common effect.
- Unconditional: $A$ and $B$ are independent.
- Conditional on $C$: $A$ and $B$ are correlated. *This is the bad-control case.*

The **backdoor criterion** says: to estimate the causal effect of $T$ on $Y$, find a set $X$ that (a) blocks all backdoor paths from $T$ to $Y$ and (b) does not contain any descendants of $T$. Including the right confounders blocks paths; including colliders or post-treatment variables opens them.

### Why adding more controls can make things worse

A common bad heuristic: 'Just throw in everything you have, more controls is more conservative.' This is wrong in two ways:

1. **Collider bias.** A control that is a collider opens a non-causal path between treatment and outcome. The simulation in the notebook shows this concretely: a regression that should recover zero effect instead recovers a 'significant' coefficient.
2. **Post-treatment bias.** A variable that is itself caused by the treatment partially mediates the effect; controlling for it absorbs part of what you are trying to measure.

The right heuristic: each control should be **defended on theoretical grounds** as a confounder. Vague reasons like 'it might matter' are not enough.

### A working taxonomy of controls (Cinelli, Forney, Pearl 2022)

| Type | DAG position | Effect on causal coefficient |
|---|---|---|
| Confounder | $T \leftarrow X \to Y$ | Removes bias. **Include.** |
| Mediator | $T \to X \to Y$ | Blocks indirect path; you measure direct effect. **Include only if intentional.** |
| Collider | $T \to X \leftarrow Y$ | Creates spurious association. **Exclude.** |
| Descendant of outcome | $Y \to X$ | Equivalent to controlling for $Y$. **Exclude.** |
| Descendant of treatment | $T \to X$ (and $X$ does not affect $Y$ otherwise) | Often partial mediator; usually biases. **Exclude.** |
| Pure $Y$-predictor | $X \to Y$ only | Improves precision; does not affect identification. **Optional.** |
| Pure $T$-predictor | $X \to T$ only | Does not bias coefficient on $T$. **Neutral.** |

Three decisions are forced on you for every candidate control:

1. Is it caused by the treatment?
2. Is it a common cause of the treatment and outcome?
3. Is it a common effect of the treatment and outcome (or treatment and an unobserved cause of $Y$)?

Each answer comes from theory and domain knowledge, not from the data.

### Cross-Reference: Which previous coefficients were causal?

Most coefficients in §02 and §02b were **predictive**:

- $\hat\beta$ on `T10Y2Y_lag1` in a regression of next-quarter GDP growth: predictive. Yield curve slope is not assigned randomly; Fed policy and recession sentiment confound.
- $\hat\beta$ on `UNRATE_lag1` in the same regression: predictive. Unemployment is determined by the same labor market that determines GDP.

A few were closer to causal:

- $\hat\beta$ on a **policy dummy** (e.g., the COVID dummy from §04 §1) is causal *if* COVID timing is exogenous to GDP — which it largely is (a virus arriving is plausibly external to US business-cycle dynamics).
- A regression on `policy_announcement` data with **event-study windows** can be causal under timing assumptions.

The recession-classification work in §03 is squarely predictive: it predicts whether next quarter is a recession, with no claim about what would *prevent* one.

### Project Code Map
- `src/econometrics.py`: existing OLS helpers; the notebook does not introduce new wrappers because the DAG/OVB material is conceptual.
- The notebook simulates everything inline; nothing new in `src/`.

### Common Mistakes
- 'I included it because it was significant.' Inclusion criteria should be theoretical, not statistical.
- Treating large $R^2$ as evidence of a causal model. $R^2$ is a predictive metric.
- Lagging the treatment and calling it identified. Lagging removes simultaneity but not unobserved confounding.
- Confusing the average treatment effect (ATE) with the average treatment effect on the treated (ATT). Different estimands; usually different numbers.

<a id="summary"></a>
## Summary + Suggested Readings

After this notebook you should be able to:

- distinguish predictive from causal estimands precisely,
- read and write a small DAG for an applied question,
- diagnose omitted-variable bias by simulation and verify it with the bias formula,
- spot a collider trap and explain why 'controlling for' it creates spurious correlation,
- classify each candidate control as confounder, mediator, collider, post-treatment, or neutral.

**Companion guides:**
- [Guide 01 — Difference-in-Differences](01_difference_in_differences.md): the workhorse research design that operationalizes a parallel-trends assumption to identify a causal effect.

**Suggested readings:**
- Pearl, *Causality: Models, Reasoning, and Inference* (2nd ed.) — the definitive DAG reference.
- Pearl, Glymour, Jewell, *Causal Inference in Statistics: A Primer* — accessible introduction with worked examples.
- Cinelli, Forney, Pearl (2022), "A Crash Course in Good and Bad Controls," *Sociological Methods & Research* — the taxonomy in this guide.
- Angrist and Pischke, *Mostly Harmless Econometrics* — the applied-econ bible. Chapter 3 is the OVB material; Chapter 5 introduces DiD and IV.
- Cunningham, *Causal Inference: The Mixtape* (free online: mixtape.scunning.com) — modern, code-heavy, written for first-year grad students.
