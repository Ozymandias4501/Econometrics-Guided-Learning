### Core Causal Inference: From Questions → Identification → Estimation

This project’s causal modules are built around one idea:

> **If you cannot clearly state the counterfactual, you cannot interpret a coefficient causally.**

Below is a “lecture-notes” walkthrough of the causal vocabulary you will reuse in FE / DiD / IV.

#### 1) Intuition (plain English): what problem are we solving?

In **prediction**, you ask:
- “Given what I observe today, what will happen next?”

In **causal inference**, you ask:
- “If I intervened and changed something, what would happen instead?”

**Story example (micro):** What is the effect of *one more year of schooling* on earnings?
- Prediction: people with more schooling earn more; you can forecast earnings from schooling.
- Causality: if we forcibly increased a person’s schooling by one year, would earnings rise? By how much?

**Story example (macro/policy):** What is the effect of a change in monetary policy on unemployment?
- Prediction: unemployment and interest rates move together in time series.
- Causality: if the policy rate were higher *holding everything else fixed*, what would unemployment have been?

The difference is the missing world we never observe.

#### 2) Notation + setup: potential outcomes (define every symbol)

We index:
- units by $i = 1,\\dots,N$ (counties, states, people, firms, …),
- time by $t = 1,\\dots,T$ (years, quarters, …).

Let:
- $D_{it} \\in \\{0,1\\}$ be a treatment indicator (treated vs not treated).
- $Y_{it}(1)$ be the outcome *if treated*.
- $Y_{it}(0)$ be the outcome *if not treated*.

We observe only one outcome per unit-time:

$$
Y_{it} = D_{it} \\cdot Y_{it}(1) + (1 - D_{it}) \\cdot Y_{it}(0).
$$

**What each term means**
- $Y_{it}$: observed outcome (e.g., poverty rate in county $i$ in year $t$).
- $D_{it}$: whether the unit is treated at time $t$.
- $Y_{it}(1), Y_{it}(0)$: two “parallel universes” outcomes.

The **individual causal effect** is:
$$
\\tau_{it} = Y_{it}(1) - Y_{it}(0),
$$
but we never observe both terms for the same $(i,t)$.

Common estimands:

$$
\\text{ATE} = \\mathbb{E}[Y(1) - Y(0)],
\\qquad
\\text{ATT} = \\mathbb{E}[Y(1) - Y(0) \\mid D=1].
$$

**What each term means**
- ATE: average effect in the population of interest.
- ATT: average effect for treated units (often what DiD identifies).

#### 3) Identification vs estimation (and why this distinction matters)

> **Identification** answers: “Which assumptions turn observed data into a causal effect?”

> **Estimation** answers: “Given those assumptions, how do we compute the effect?”

Example:
- DiD identification assumption: **parallel trends** (treated and control would have evolved similarly absent treatment).
- DiD estimator: a difference of differences (or a TWFE regression).

Regression output (a coefficient + p-value) is **estimation**. It is *not* identification.

#### 4) Selection bias in one equation (why naive comparisons fail)

A naive “treated vs control” difference in observed means is:
$$
\\mathbb{E}[Y \\mid D=1] - \\mathbb{E}[Y \\mid D=0].
$$

Add and subtract $\\mathbb{E}[Y(0) \\mid D=1]$ to decompose it:

$$
\\underbrace{\\mathbb{E}[Y(1) \\mid D=1] - \\mathbb{E}[Y(0) \\mid D=1]}_{\\text{ATT}}
\\; + \\;
\\underbrace{\\mathbb{E}[Y(0) \\mid D=1] - \\mathbb{E}[Y(0) \\mid D=0]}_{\\text{selection bias}}.
$$

**What each term means**
- The first bracket is the causal effect for treated units.
- The second bracket is the difference in untreated potential outcomes between treated and control units.

If treated units would have had different outcomes *even without treatment*, the naive comparison is biased.

#### 5) Core causal assumptions you will see repeatedly

You will see versions of these assumptions across FE/DiD/IV:

1) **Consistency**
- If a unit is treated ($D=1$), the observed outcome equals $Y(1)$; if not, it equals $Y(0)$.

2) **SUTVA (no interference + well-defined treatment)**
- My treatment does not change your outcome (no spillovers).
- “Treatment” is a specific, comparable intervention (not a vague label).

3) **Exchangeability / exogeneity (design-specific)**
- Some condition that makes the treatment “as good as random” *after conditioning or differencing*.
  - FE uses within-unit changes to remove time-invariant confounding.
  - DiD uses parallel trends to remove common time shocks.
  - IV uses exclusion + relevance to isolate quasi-random variation in treatment.

4) **Overlap (when conditioning is used)**
- There are both treated and untreated units for the covariate patterns you analyze.

#### 6) Estimation mechanics (high level): mapping designs → estimators

This repo focuses on three workhorse designs:

- **Panel fixed effects (FE / TWFE)**  
  Model: $Y_{it} = \\beta'X_{it} + \\alpha_i + \\gamma_t + \\varepsilon_{it}$  
  Identifying variation: within-unit changes over time.

- **Difference-in-differences (DiD / event study)**  
  Model: $Y_{it} = \\beta D_{it} + \\alpha_i + \\gamma_t + \\varepsilon_{it}$  
  Identifying variation: treated vs control *changes*.

- **Instrumental variables (IV / 2SLS)**  
  Structural: $Y = \\beta X + u$ with $\\mathrm{Cov}(X,u) \\neq 0$  
  Instrument: $Z$ shifts $X$ but is excluded from $Y$ except through $X$.

#### 7) Inference: why standard errors are part of the design

Even if your estimator is unbiased under identification assumptions, **inference can fail** if you treat dependent observations as independent.

Common dependence patterns:
- counties in the same state share shocks → errors correlated within state,
- repeated observations over time → serial correlation.

That is why causal notebooks emphasize **clustered standard errors** and “number of clusters” reporting.

#### 8) Diagnostics + robustness (minimum set you should practice)

At least three diagnostics should become habits:

1) **Design diagnostic:** does the identifying assumption look plausible?
   - DiD: pre-trends / leads in an event study.
   - IV: first-stage strength and exclusion plausibility.

2) **Specification sensitivity:** does the estimate move a lot if you change the spec?
   - add/remove plausible controls,
   - change time window,
   - change clustering level.

3) **Falsification / placebo:** does the method “find effects” where none should exist?
   - fake treatment dates,
   - outcomes that should not respond,
   - never-treated placebo groups.

#### 9) Interpretation + reporting (how to write results honestly)

Good causal write-ups answer these questions explicitly:
- What is the **causal question** (intervention, population, time horizon)?
- What is the **estimand** (ATE, ATT, dynamic effects)?
- What is the **identification assumption** (in one sentence)?
- What is the **estimation method** (FE/DiD/IV; SE choice)?
- What diagnostics/robustness checks support or weaken the claim?

**What this does NOT mean**
- A significant coefficient does **not** prove a causal story.
- “Controls” do **not** automatically eliminate bias.
- Better fit ($R^2$) does **not** imply better identification.

#### 10) Why this repo uses “semi-synthetic” exercises

Some notebooks add a **known treatment effect** to a real outcome to create a semi-synthetic truth.
That lets you verify whether an estimator can recover the known effect, without pretending the dataset is a real policy evaluation.

<details>
<summary>Optional: very light DAG intuition (one paragraph)</summary>

A causal diagram (DAG) is a picture of assumptions: arrows represent direct causal links.
Confounding is “backdoor paths” from $D$ to $Y$ (e.g., ability affects both schooling and earnings).
Designs like FE/DiD/IV are ways to block backdoor paths using time differencing, group comparisons, or instruments.

</details>

#### Exercises (do these with the matching notebooks)

- [ ] Pick one notebook question and write the potential outcomes $Y(1), Y(0)$ in words.
- [ ] Define whether you care about ATE or ATT in that notebook and explain why.
- [ ] Write the selection-bias decomposition for a naive comparison in your context.
- [ ] List 3 threats to identification for your design (confounding, anticipation, spillovers, measurement error, …).
- [ ] Write one placebo test you could run and what a “failure” would look like.
- [ ] Explain in 4 sentences why a p-value is not a substitute for identification.
