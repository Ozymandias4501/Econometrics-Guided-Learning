# Guide: 00_build_census_county_panel

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/07_causal/00_build_census_county_panel.ipynb`.

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
- Complete notebook section: Choose years + variables
- Complete notebook section: Fetch/cache ACS tables
- Complete notebook section: Build panel + FIPS
- Complete notebook section: Save processed panel
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

### Panel Fixed Effects (FE): “Compare a unit to itself over time”

Fixed effects are one of the most common ways to reduce bias from **time-invariant unobservables** in panel data.

#### 1) Intuition (plain English)

Panels are powerful because they let you control for factors that differ across units but are roughly constant over time:
- geography,
- long-run institutions,
- baseline demographics,
- “culture” or other hard-to-measure features.

**Story example:** Counties differ in baseline poverty for many persistent reasons.  
If you want to study how changes in unemployment relate to changes in poverty, it is more credible to compare:
- *the same county in different years*  
than to compare two different counties once.

#### 2) Notation + setup (define symbols)

Let:
- $i = 1,\\dots,N$ index entities (counties),
- $t = 1,\\dots,T$ index time (years),
- $Y_{it}$ be the outcome,
- $X_{it}$ be a $K \\times 1$ vector of regressors,
- $\\alpha_i$ be an entity-specific intercept (unit FE),
- $\\gamma_t$ be a time-specific intercept (time FE),
- $\\varepsilon_{it}$ be the remaining error.

The two-way fixed effects (TWFE) model is:

$$
Y_{it} = X_{it}'\\beta + \\alpha_i + \\gamma_t + \\varepsilon_{it}.
$$

**What each term means**
- $X_{it}'\\beta$: the part explained by observed covariates.
- $\\alpha_i$: all time-invariant determinants of $Y$ for entity $i$ (observed or unobserved).
- $\\gamma_t$: shocks common to all entities in time $t$ (recessions, nationwide policy, measurement changes).
- $\\varepsilon_{it}$: idiosyncratic shocks not captured above.

Matrix form (stack observations by time within entity):

$$
\\mathbf{y} = \\mathbf{X}\\beta + \\mathbf{A}\\alpha + \\mathbf{G}\\gamma + \\varepsilon
$$

where:
- $\\mathbf{A}$ is the entity-dummy design matrix,
- $\\mathbf{G}$ is the time-dummy design matrix.

#### 3) What FE “assumes” (identification conditions)

FE is not magic; it removes *time-invariant* confounding, not everything.

A common identification condition is **strict exogeneity**:

$$
\\mathbb{E}[\\varepsilon_{it} \\mid X_{i1}, \\dots, X_{iT}, \\alpha_i] = 0
\\quad \\text{for all } t.
$$

**What this means**
- After controlling for unit FE, time FE, and the full history of $X$, the remaining shocks are mean-zero.
- If today’s shock changes future $X$ (feedback), strict exogeneity can fail.

Other requirements:
- **Within-unit variation:** the regressor must change over time within entities.
- **No perfect multicollinearity:** you cannot include variables fully determined by FE (e.g., a pure time trend plus full time FE can collide depending on spec).

<details>
<summary>Optional: dynamic panels (why lagged Y can be tricky)</summary>

If you include $Y_{i,t-1}$ as a regressor, FE can create “Nickell bias” when $T$ is small.
This repo does not focus on dynamic panel estimators (Arellano–Bond), but you should know this pitfall exists.

</details>

#### 4) Estimation mechanics: the “within” transformation (derivation)

The FE estimator can be understood as OLS after removing means.

**Entity FE only (simpler to see):**

Start with:
$$
Y_{it} = X_{it}'\\beta + \\alpha_i + \\varepsilon_{it}.
$$

Take the time average within each entity:
$$
\\bar{Y}_i = \\bar{X}_i'\\beta + \\alpha_i + \\bar{\\varepsilon}_i.
$$

Subtract the entity mean equation from the original:

$$
Y_{it} - \\bar{Y}_i = (X_{it} - \\bar{X}_i)'\\beta + (\\varepsilon_{it} - \\bar{\\varepsilon}_i).
$$

**What this achieves**
- The term $\\alpha_i$ disappears (it is constant within entity).
- Identification comes from deviations from the entity’s own average.

TWFE (“two-way”) uses a version of demeaning that removes both:
- entity averages, and
- time averages,
often implemented by “absorbing” FE rather than explicitly constructing every dummy.

#### 5) Frisch–Waugh–Lovell (FWL) interpretation (how software implements FE)

FWL says: if you want $\\beta$ in a regression with controls, you can:
1) residualize $Y$ on the controls,
2) residualize $X$ on the controls,
3) regress residualized $Y$ on residualized $X$.

In FE models, “controls” include thousands of entity/time dummies. Software (like `linearmodels`) uses the same logic efficiently.

#### 6) Inference: why clustering is common in panels

Panel errors are often correlated:
- within entity over time (serial correlation),
- within higher-level groups (e.g., state shocks),
- within time (common shocks).

Robust (HC) SE handle heteroskedasticity but not arbitrary within-cluster correlation.
That’s why you often see **clustered SE** in applied FE work.

#### 7) Diagnostics + robustness (at least 3)

1) **Pooled vs FE comparison**
- Fit pooled OLS and FE; if coefficients move a lot, unobserved heterogeneity likely mattered.

2) **Within-variation check**
- For each regressor, confirm it varies within entity. If not, FE cannot estimate it.

3) **Sensitivity to time FE**
- Add/remove time FE; if results change materially, time shocks matter and must be modeled.

4) **Influential units**
- Check whether a handful of entities drive the estimate (outliers / leverage).

#### 8) Interpretation + reporting (what FE coefficients mean)

In a TWFE regression, interpret $\\beta_j$ as:

> the association between *within-entity changes* in $x_j$ and *within-entity changes* in $Y$, after removing common time shocks.

**What this does NOT mean**
- FE does **not** guarantee causality (time-varying omitted variables can remain).
- FE does **not** solve reverse causality (shocks can affect both $X$ and $Y$).
- FE does **not** “fix” measurement error (it can make some bias worse).

#### Exercises

- [ ] Take one regressor in the panel notebook and compute $x_{it} - \\bar{x}_i$ manually; confirm the mean is ~0 within each entity.
- [ ] Explain in words what variation identifies $\\beta$ in a TWFE model.
- [ ] Name one time-invariant variable you *wish* you could estimate and explain why FE absorbs it.
- [ ] Fit pooled OLS and TWFE; write 3 sentences explaining why the coefficients differ.
- [ ] Try adding time FE and report whether the coefficient on your main regressor is stable.

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
