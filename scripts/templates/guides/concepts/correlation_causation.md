### Deep Dive: Correlation vs causation — the question you must not confuse

Economics is full of correlated variables. Causal inference is about deciding when a relationship is more than correlation.

#### 1) Intuition (plain English)

Correlation answers:
- “Do $X$ and $Y$ move together?”

Causation answers:
- “If I intervene and change $X$, does $Y$ change?”

**Story example (macro):** Interest rates and inflation are correlated.  
That does not imply raising rates increases inflation; the correlation can reflect policy responses to expected inflation.

#### 2) Notation + setup (define symbols)

Correlation between random variables $X$ and $Y$:

$$
\\rho_{XY} = \\frac{\\mathrm{Cov}(X,Y)}{\\sqrt{\\mathrm{Var}(X)\\mathrm{Var}(Y)}}.
$$

**What each term means**
- $\\mathrm{Cov}(X,Y)$: whether $X$ and $Y$ co-move.
- correlation is unit-free and lies in $[-1,1]$.

Correlation is symmetric:
$$
\\rho_{XY} = \\rho_{YX}.
$$
But causal effects are directional.

#### 3) Assumptions (what you need for a causal claim)

A causal claim requires an identification strategy:
- randomization,
- a natural experiment,
- a credible quasi-experimental design (DiD/IV/RD),
- or a structural model with defensible assumptions.

Without identification, regression coefficients are best interpreted as conditional associations.

#### 4) Estimation mechanics: how confounding creates correlation without causation

Consider a simple confounding structure:
- $Z$ causes both $X$ and $Y$.

One possible DGP:
$$
X = aZ + \\eta, \\qquad Y = bZ + \\varepsilon.
$$

Even if $X$ does not cause $Y$, $X$ and $Y$ will be correlated because they share the common cause $Z$.

Regression “controls” can help if you measure the confounder, but:
- you rarely observe all confounders,
- controlling for the wrong variables (colliders/mediators) can introduce bias.

#### 5) Inference: significance is not causality

A small p-value means “incompatible with $\\beta=0$ under the model assumptions.”
It does not mean:
- the model is correct,
- the effect is causal,
- the effect is economically large.

#### 6) Diagnostics + robustness (minimum set)

1) **Timing sanity**
- can $X$ plausibly affect $Y$ given publication/decision timing?

2) **Confounder list**
- write down plausible common causes of $X$ and $Y$ (before running regressions).

3) **Placebos**
- test outcomes that should not respond to $X$; “effects” there suggest confounding.

4) **Design upgrade**
- if you need causality, move from “controls” to FE/DiD/IV where appropriate.

#### 7) Interpretation + reporting

Be explicit about the claim type:
- predictive association vs causal effect.

**What this does NOT mean**
- “Controlling for some variables” is not a guarantee of causality.

#### Exercises

- [ ] Write one example where correlation is expected but causality is ambiguous (macro or micro).
- [ ] Draw a simple confounding story in words (Z→X and Z→Y).
- [ ] Simulate confounding and show correlation without causation.
- [ ] Rewrite a regression interpretation paragraph to remove causal language unless justified.
