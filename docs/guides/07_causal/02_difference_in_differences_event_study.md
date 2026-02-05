# Guide: 02_difference_in_differences_event_study

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/07_causal/02_difference_in_differences_event_study.ipynb`.

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
- Complete notebook section: Synthetic adoption + treatment
- Complete notebook section: TWFE DiD
- Complete notebook section: Event study (leads/lags)
- Complete notebook section: Diagnostics: pre-trends + placebo
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

### Difference-in-Differences (DiD): learning from “changes vs changes”

DiD is a workhorse design for causal inference when treatment is not randomized but changes over time for some units.

#### 1) Intuition (plain English)

If treated units and control units start at different levels, a raw comparison is not credible.
DiD fixes *level differences* by comparing **changes**:
- how much did outcomes change in treated units after treatment?
- how much did outcomes change in control units over the same period?

**Story example:** A policy is adopted in some states in 2018 but not others.  
We compare outcome changes in adopting states vs non-adopting states.

#### 2) Notation + setup (2×2 first, define symbols)

Let:
- groups $g \\in \\{T, C\\}$ for treated vs control group,
- time $t \\in \\{0, 1\\}$ for pre vs post,
- $Y_{gt}$ be the average observed outcome in group $g$ at time $t$.

Potential outcomes:
- $Y_{gt}(1)$: outcome if treated,
- $Y_{gt}(0)$: outcome if not treated.

In the 2×2 setting, only the treated-post cell is treated:
- treated group in post: observed $Y_{T1} = Y_{T1}(1)$,
- everything else: observed is $Y(0)$.

#### 3) The DiD estimand (and what each term means)

The DiD estimand is:

$$
\\widehat{\\tau}_{\\text{DiD}}
= (Y_{T1} - Y_{T0}) - (Y_{C1} - Y_{C0}).
$$

**What each piece means**
- $Y_{T1} - Y_{T0}$: change in treated group (pre → post).
- $Y_{C1} - Y_{C0}$: change in control group (same calendar time).
- Subtracting removes common shocks that affect both groups.

#### 4) Identification: the parallel trends assumption

DiD becomes causal under **parallel trends**:

$$
\\mathbb{E}[Y_{T1}(0) - Y_{T0}(0)] = \\mathbb{E}[Y_{C1}(0) - Y_{C0}(0)].
$$

**What this says**
- In the absence of treatment, treated and control would have evolved similarly over time.
- This is a statement about *counterfactual trends*, not about levels.

Other common supporting assumptions:
- **No anticipation:** treatment does not affect outcomes before it occurs.
- **No spillovers:** treatment of one group does not directly change outcomes of the other.
- **Stable composition:** the definition of the groups does not change in a way that creates artificial trends.

#### 5) Regression form: DiD as a fixed-effects model

With many units and time periods, DiD is often estimated via a TWFE regression:

$$
Y_{it} = \\alpha_i + \\gamma_t + \\beta D_{it} + \\varepsilon_{it}.
$$

**What each term means**
- $\\alpha_i$: unit fixed effects (remove time-invariant differences).
- $\\gamma_t$: time fixed effects (remove common time shocks).
- $D_{it}$: treatment indicator (1 if treated at time $t$).
- $\\beta$: average effect under the DiD identification assumptions.

In the 2×2 case, this regression coefficient equals the DiD difference-of-differences number.

#### 6) Inference: why clustering is standard in DiD

DiD settings often have serial correlation (outcomes persist over time).
If you treat panel rows as independent, SE can be badly understated.

Practical rule:
- cluster at (or above) the level of treatment assignment / shared shocks (often state).

#### 7) Diagnostics + robustness (minimum set)

1) **Pre-trends / leads (event study)**
- If treated units were already trending differently before treatment, parallel trends is doubtful.

2) **Placebo interventions**
- Assign fake treatment dates or fake treated groups; you should not “find” big effects.

3) **Window sensitivity**
- Re-estimate using different pre/post windows; large sensitivity suggests fragility.

4) **Outcome sanity**
- Try an outcome that should not respond (if available). If it “responds,” your design is suspicious.

#### 8) Interpretation + reporting

Good DiD reporting includes:
- the estimand (ATT? average post effect? dynamic effects?),
- the identifying assumption (parallel trends) and evidence you provide,
- SE choice (cluster level + number of clusters),
- a figure (trend plot and/or event-study plot).

**What this does NOT mean**
- A TWFE coefficient is not automatically causal if parallel trends fails.
- Controlling for many covariates does not rescue a failing design.

<details>
<summary>Optional: modern caveat — TWFE pitfalls with staggered adoption</summary>

When different units adopt treatment at different times (“staggered adoption”), the simple TWFE regression can produce misleading averages if treatment effects vary over time or across cohorts.
Modern alternatives (e.g., cohort-specific DiD estimators) address this.

In this repo, we use semi-synthetic adoption/effects to learn mechanics; for real research you should study these newer estimators.

</details>

#### Exercises

- [ ] Draw a 2×2 table and compute the DiD estimand from hypothetical numbers you choose.
- [ ] Write the parallel trends assumption in words for your dataset (what would have happened absent treatment?).
- [ ] Plot group means over time (treated vs control) and visually assess pre-trends.
- [ ] Run one placebo test and explain what you expected vs what you found.
- [ ] Write a short paragraph explaining why clustering matters in DiD.

### Event Studies (Leads/Lags): dynamics + pre-trends in one picture

Event studies generalize DiD by estimating how effects evolve before and after adoption.

#### 1) Intuition (plain English)

A single DiD coefficient answers: “What is the average post-treatment effect?”

An event study answers:
- “Do we see effects **before** adoption (bad sign / anticipation / confounding)?”
- “How do effects evolve **after** adoption (ramp up, fade out, persist)?”

**Story example:** A policy starts in 2018.  
If outcomes in treated states start moving in 2016 relative to controls, your design is suspect.

#### 2) Notation + setup (define symbols)

Let:
- $T_i$ be the adoption time for unit $i$ (if it adopts),
- event time $k = t - T_i$ (years relative to adoption),
- $1[\\cdot]$ be an indicator function.

We build lead/lag dummies for a window $k \\in \\{-K,\\dots,-2,0,\\dots,L\\}$.
We omit one pre-treatment period (often $k=-1$) as the reference category.

The standard event-study regression is:

$$
Y_{it} = \\alpha_i + \\gamma_t + \\sum_{k \\neq -1} \\beta_k \\cdot 1[t - T_i = k] + \\varepsilon_{it}.
$$

**What each term means**
- $\\alpha_i$: unit FE (levels).
- $\\gamma_t$: time FE (common shocks).
- $\\beta_k$: effect at event time $k$ relative to the reference period.
- Reference period $k=-1$: all other coefficients are relative to this baseline.

#### 3) Identification: what you need for an event study to be credible

Event studies inherit DiD assumptions plus some extras:

- **Parallel trends in the pre-period:** lead coefficients should be near zero (no differential pre-trends).
- **No anticipation:** treatment should not affect outcomes before adoption.
- **No spillovers/interference:** other units are not affected directly by others’ adoption.
- **Stable treatment definition:** the “treatment” is comparable across cohorts.

#### 4) Estimation mechanics (how you build the design matrix)

The core construction is the event-time dummies:
- compute $k = t - T_i$,
- for each $k$ in your window create a dummy $D_{it}^k = 1[k=t-T_i]$ for treated units,
- omit $k=-1$ (or another base) to avoid collinearity.

Then run FE regression with clustered SE.

#### 5) Inference: multiple coefficients, multiple comparisons

Event studies estimate many $\\beta_k$’s.
That raises two practical issues:

1) **Statistical uncertainty increases** (many parameters).
2) **Multiple testing** risk: some coefficients will look significant by chance.

In this repo, treat the event-study plot primarily as:
- a diagnostic (pre-trends),
- and an effect-shape description (not a fishing expedition).

#### 6) Diagnostics + robustness (minimum set)

1) **Pre-trends check (leads)**
- Are lead coefficients jointly near 0? (informally: do they bounce around 0 with wide CI?)

2) **Placebo adoption**
- Shift adoption dates earlier or assign adoption to never-treated units; you should not see “effects.”

3) **Window/bins sensitivity**
- Change the event window; bin far leads/lags; check if conclusions are stable.

4) **Cohort heterogeneity**
- Compare early adopters vs late adopters; big differences can indicate heterogeneous effects.

#### 7) Interpretation + reporting

When you report an event study:
- Always show the plot (coefficients + CI).
- State the base period.
- Emphasize what the leads say about pre-trends.
- Summarize post-treatment dynamics (when does effect start? peak? persist?).

**What this does NOT mean**
- A flat pre-trend is supportive evidence, not proof.
- A significant lead coefficient is a warning sign, not a “cool finding.”

<details>
<summary>Optional: staggered adoption warning (one paragraph)</summary>

With staggered adoption and heterogeneous effects, classic TWFE event studies can mix comparisons across cohorts and time in subtle ways.
Modern event-study estimators avoid some of these pitfalls; study them if you apply this to real policy evaluation.

</details>

#### Exercises

- [ ] Build an event-time variable and confirm it equals 0 in the adoption year for treated units.
- [ ] Choose a base period and explain (in words) what “relative to base” means.
- [ ] Plot the event-study coefficients and write 4 sentences interpreting (a) leads and (b) lags.
- [ ] Run one placebo adoption test and explain whether it supports the design.

### Clustered Standard Errors: Inference when observations move together

Clustered standard errors are about **honest uncertainty** when errors are correlated within groups.

#### 1) Intuition (plain English)

Regression formulas for standard errors often assume each row is “independent.”
In economics, that is frequently false.

**Story example:** Counties in the same state share:
- state policy changes,
- state-level business cycles,
- housing markets and migration flows.

If your residuals are correlated within states and you ignore that, your standard errors are often too small, making effects look “significant” when they are not.

#### 2) Notation + setup (define symbols)

Consider a linear regression in stacked form:

$$
\\mathbf{y} = \\mathbf{X}\\beta + \\mathbf{u}.
$$

Let:
- $n$ be the number of observations (rows),
- $K$ be the number of regressors,
- $g \\in \\{1,\\dots,G\\}$ index clusters (e.g., states),
- $\\mathbf{X}_g$ be the rows of $\\mathbf{X}$ in cluster $g$,
- $\\mathbf{u}_g$ be the residual vector in cluster $g$.

#### 3) What clustering assumes (and what it relaxes)

Cluster-robust SE relax the “independent errors” assumption **within** clusters:
- errors can be heteroskedastic and correlated in arbitrary ways within a cluster.

But they still assume something like **independence across clusters**:
- cluster $g$ shocks are independent of cluster $h$ shocks (for $g \\neq h$).

This is why **choosing the right cluster level is part of the design**, not a technical afterthought.

#### 4) Estimation mechanics: the cluster-robust covariance (sandwich form)

The OLS coefficient estimate $\\hat\\beta$ does not change when you change SE.
What changes is the estimated variance of $\\hat\\beta$.

The cluster-robust (one-way) covariance estimator is:

$$
\\widehat{\\mathrm{Var}}_{\\text{CL}}(\\hat\\beta)
= (\\mathbf{X}'\\mathbf{X})^{-1}
\\left(\\sum_{g=1}^{G} \\mathbf{X}_g' \\hat{\\mathbf{u}}_g \\hat{\\mathbf{u}}_g' \\mathbf{X}_g \\right)
(\\mathbf{X}'\\mathbf{X})^{-1}.
$$

**What each term means**
- “Bread”: $(\\mathbf{X}'\\mathbf{X})^{-1}$ is the same matrix you see in OLS.
- “Meat”: the sum over clusters aggregates within-cluster residual covariance.
- This estimator reduces to robust (HC) SE when each observation is its own cluster.

Many libraries also apply small-sample corrections (especially when $G$ is not huge).

#### 5) Mapping to code (statsmodels / linearmodels)

In `statsmodels`, you can request clustered SE via `cov_type='cluster'` with a `groups` vector.

In `linearmodels` (PanelOLS), clustering is common:
- build a `clusters` Series aligned to the panel index,
- use `cov_type='clustered'`.

#### 6) Inference pitfalls (important!)

1) **Few clusters problem**
- When the number of clusters $G$ is small (rule of thumb: < 30–50), cluster-robust inference can be unreliable.
- In serious applied work, people use remedies like wild cluster bootstrap; we do not implement that here, but you should know it exists.

2) **Wrong clustering level**
- If treatment is assigned at the state level, clustering below that (e.g., county) can be too optimistic.
- A common rule: cluster at the level of treatment assignment or the level of correlated shocks.

3) **Serial correlation in DiD**
- Classic results (e.g., Bertrand et al.) show naive SE can be severely biased in DiD with serial correlation.
- Clustering is often the minimum fix.

#### 7) Diagnostics + robustness (at least 3)

1) **Report cluster count**
- Always report $G$ (number of clusters). If $G$ is tiny, treat inference as fragile.

2) **Sensitivity to clustering level**
- Try clustering by county vs by state (or time). If SE change a lot, dependence is important.

3) **Residual dependence check**
- Plot residuals by cluster over time or compute within-cluster autocorrelation.

4) **Aggregation robustness (DiD-style)**
- As a sanity check, aggregate to the treatment-assignment level and see if conclusions are similar.

#### 8) Interpretation + reporting

Clustered SE change your uncertainty, not your point estimate.
So the right way to write results is:
- coefficient (effect size),
- clustered SE (uncertainty),
- cluster level and number of clusters,
- design assumptions.

**What this does NOT mean**
- Clustering does not fix bias from confounding or misspecification.
- Clustering does not “prove” causality; it only helps prevent overconfident inference.

#### Exercises

- [ ] In a panel regression, compute naive (HC) SE and clustered SE; compare the ratio for your main coefficient.
- [ ] Explain in words why state-level shocks make county-level rows dependent.
- [ ] Try clustering by entity vs by state; write 3 sentences about which is more defensible and why.
- [ ] If you had only 8 states, what would make you cautious about cluster-robust p-values?

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
