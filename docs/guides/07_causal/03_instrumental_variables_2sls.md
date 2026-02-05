# Guide: 03_instrumental_variables_2sls

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/07_causal/03_instrumental_variables_2sls.ipynb`.

This module adds identification-focused econometrics: panels, DiD/event studies, and IV.

### Key Terms (defined)
- **Identification**: assumptions that justify a causal interpretation.
- **Fixed effects (FE)**: controls for time-invariant unit differences.
- **Clustered SE**: allows correlated errors within groups (e.g., state).
- **DiD**: compares changes over time between treated and control units.
- **IV/2SLS**: uses an instrument to address endogeneity.


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Simulate endogeneity
- Complete notebook section: OLS vs 2SLS
- Complete notebook section: First-stage + weak IV checks
- Complete notebook section: Interpretation + limitations
- Write the causal question and identification assumptions before estimating.
- Run at least one diagnostic/falsification (pre-trends, placebo, weak-IV check).
- Report clustered SE (and number of clusters) when appropriate.

### Alternative Example (Not the Notebook Solution)
```python
# Toy DiD setup (not the notebook data):
import numpy as np
import pandas as pd

df = pd.DataFrame({
  'group': ['T']*50 + ['C']*50,
  'post':  [0]*25 + [1]*25 + [0]*25 + [1]*25,
})
df['treated'] = (df['group'] == 'T').astype(int)
df['D'] = df['treated'] * df['post']
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

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

### Instrumental Variables (IV) and 2SLS: isolating “as-if random” variation

IV is used when a key regressor is endogenous (correlated with the error term).

#### 1) Intuition (plain English)

Suppose you want the effect of $X$ on $Y$.
If $X$ is chosen by people/policymakers/markets, it likely reflects information that also affects $Y$.

**Story example:** Education ($X$) and earnings ($Y$).
- Ability affects both schooling and earnings.
- OLS confounds the schooling effect with ability.

IV tries to find a variable $Z$ (“instrument”) that moves $X$ for reasons unrelated to the unobserved determinants of $Y$.

#### 2) Notation + setup (define symbols)

Structural equation (what you want):

$$
Y_i = \\beta X_i + W_i'\\delta + u_i
$$

First stage (how $Z$ moves $X$):

$$
X_i = \\pi Z_i + W_i'\\rho + v_i
$$

**What each term means**
- $Y_i$: outcome.
- $X_i$: endogenous regressor (the “treatment” intensity).
- $W_i$: exogenous controls (included in both stages).
- $Z_i$: instrument(s).
- $u_i$: structural error (unobservables affecting $Y$).
- $v_i$: first-stage error.

Endogeneity means:
$$
\\mathrm{Cov}(X_i, u_i) \\neq 0.
$$

#### 3) Identification: the two IV conditions

An instrument must satisfy:

1) **Relevance**
$$
\\mathrm{Cov}(Z_i, X_i) \\neq 0
$$
The instrument meaningfully predicts $X$ (a strong first stage).

2) **Exclusion / exogeneity**
$$
\\mathrm{Cov}(Z_i, u_i) = 0
$$
$Z$ affects $Y$ only through $X$ (no direct path; no correlation with omitted determinants of $Y$).

These are design assumptions. Exclusion is especially untestable and must be defended with institutional context.

<details>
<summary>Optional: monotonicity and LATE (why IV can be “local”)</summary>

With heterogeneous treatment effects, IV often identifies a **Local Average Treatment Effect (LATE)**: the effect for “compliers” whose $X$ changes when $Z$ changes.
This adds a monotonicity assumption (“$Z$ pushes $X$ in the same direction for everyone”).

</details>

#### 4) Estimation mechanics: from IV estimand to 2SLS

**Single endogenous regressor + single instrument (no controls):**

The IV estimand is:
$$
\\beta_{IV} = \\frac{\\mathrm{Cov}(Z, Y)}{\\mathrm{Cov}(Z, X)}.
$$

**What this means**
- The numerator measures how $Z$ shifts outcomes.
- The denominator measures how $Z$ shifts the endogenous regressor.
- Their ratio interprets the outcome shift “per unit” of the regressor shift induced by the instrument.

**Two-stage least squares (2SLS) with controls:**

Stage 1 (projection):
- regress $X$ on $Z$ and $W$ to get predicted $\\hat X$ (the part of $X$ explained by instruments + controls).

Stage 2:
- regress $Y$ on $\\hat X$ and $W$.

Matrix form (compact):

$$
\\hat\\beta_{2SLS} = (X'P_Z X)^{-1} X'P_Z Y,
$$

where:
- $P_Z$ is the projection matrix onto the space spanned by instruments (and included exogenous controls).

**What each term means**
- $P_Z X$ is “the part of $X$ explained by $Z$.”
- 2SLS replaces endogenous variation in $X$ with instrumented variation.

#### 5) Inference: robust SE and weak-instrument warnings

As in OLS, you need standard errors for uncertainty.
In practice, use robust (and often clustered) SE.

Weak instruments are dangerous:
- if the first stage is weak, 2SLS estimates can be noisy and biased in finite samples.

Practical habit:
- always inspect first-stage strength (and the first-stage regression).

#### 6) Diagnostics + robustness (minimum set)

1) **First-stage strength**
- Check the magnitude and significance of $\\pi$ in the first stage.
- Use a “weak IV” diagnostic (rule-of-thumb first-stage F-type checks).

2) **Exclusion plausibility**
- Write the exclusion restriction in words and list at least 2 threats (direct effects, correlated policy, omitted shocks).

3) **Overidentification (if multiple instruments)**
- If you have more instruments than endogenous regressors, you can test instrument consistency (e.g., Hansen’s J). This is not a proof, but a diagnostic.

4) **Sensitivity**
- Try alternative instrument sets (if justified) and see if estimates are stable.

#### 7) Interpretation + reporting

Good IV reporting includes:
- structural equation and first stage written explicitly,
- relevance evidence (first stage),
- a clear statement of the exclusion restriction,
- robust/clustered SE and cluster counts (if clustered).

**What this does NOT mean**
- A strong first stage does not prove exclusion.
- A significant 2SLS coefficient is not necessarily a general ATE; it can be a LATE.

#### Exercises

- [ ] Write down one concrete endogeneity story (omitted variable or reverse causality) for a regression you care about.
- [ ] Propose a plausible instrument and write relevance + exclusion in words.
- [ ] Compute OLS and 2SLS on a simulated dataset where you control the DGP; explain which recovers the “true” effect.
- [ ] Inspect the first stage and write 3 sentences explaining whether you believe it is strong enough.

### Project Code Map
- `src/causal.py`: panel + IV helpers (`to_panel_index`, `fit_twfe_panel_ols`, `fit_iv_2sls`)
- `scripts/build_datasets.py`: ACS panel builder (writes data/processed/census_county_panel.csv)
- `src/census_api.py`: Census/ACS client (`fetch_acs`)
- `configs/census_panel.yaml`: panel config (years + variables)
- `data/sample/census_county_panel_sample.csv`: offline panel dataset
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Jumping to regression output without writing identification assumptions.
- Treating Granger-type correlations as causal effects (wrong question).
- Ignoring clustered/serial correlation and using overly small SE.
- For DiD: not checking pre-trends (leads) before interpreting effects.
- For IV: using weak instruments (no meaningful first stage).

<a id="summary"></a>
## Summary + Suggested Readings

You now have a toolkit for causal estimation under explicit assumptions (FE/DiD/IV).
The goal is disciplined thinking: identification first, estimation second.


Suggested readings:
- Angrist & Pischke: Mostly Harmless Econometrics (design-based causal inference)
- Wooldridge: Econometric Analysis of Cross Section and Panel Data (FE/IV foundations)
