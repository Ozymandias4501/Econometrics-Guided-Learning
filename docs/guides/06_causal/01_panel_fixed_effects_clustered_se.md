# Guide: 01_panel_fixed_effects_clustered_se

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/06_causal/01_panel_fixed_effects_clustered_se.ipynb`.

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
- Complete notebook section: Load panel and define variables
- Complete notebook section: Pooled OLS baseline
- Complete notebook section: Two-way fixed effects
- Complete notebook section: Clustered standard errors
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

This project's causal modules are built around one idea:

> **If you cannot clearly state the counterfactual, you cannot interpret a coefficient causally.**

Below is a "lecture-notes" walkthrough of the causal vocabulary you will reuse in FE / DiD / IV.

#### 1) Intuition (plain English): what problem are we solving?

In **prediction**, you ask:
- "Given what I observe today, what will happen next?"

In **causal inference**, you ask:
- "If I intervened and changed something, what would happen instead?"

**Story example (micro):** What is the effect of *one more year of schooling* on earnings?
- Prediction: people with more schooling earn more; you can forecast earnings from schooling.
- Causality: if we forcibly increased a person's schooling by one year, would earnings rise? By how much?

**Story example (macro/policy):** What is the effect of a change in monetary policy on unemployment?
- Prediction: unemployment and interest rates move together in time series.
- Causality: if the policy rate were higher *holding everything else fixed*, what would unemployment have been?

The difference is the missing world we never observe.

#### 2) Notation + setup: potential outcomes (define every symbol)

We index:
- units by $i = 1,\dots,N$ (counties, states, people, firms, …),
- time by $t = 1,\dots,T$ (years, quarters, …).

Let:
- $D_{it} \in \{0,1\}$ be a treatment indicator (treated vs not treated).
- $Y_{it}(1)$ be the outcome *if treated*.
- $Y_{it}(0)$ be the outcome *if not treated*.

We observe only one outcome per unit-time:

$$
Y_{it} = D_{it} \cdot Y_{it}(1) + (1 - D_{it}) \cdot Y_{it}(0).
$$

**What each term means**
- $Y_{it}$: observed outcome (e.g., poverty rate in county $i$ in year $t$).
- $D_{it}$: whether the unit is treated at time $t$.
- $Y_{it}(1), Y_{it}(0)$: two "parallel universes" outcomes.

The **individual causal effect** is:
$$
\tau_{it} = Y_{it}(1) - Y_{it}(0),
$$
but we never observe both terms for the same $(i,t)$.

Common estimands:

$$
\text{ATE} = \mathbb{E}[Y(1) - Y(0)],
\qquad
\text{ATT} = \mathbb{E}[Y(1) - Y(0) \mid D=1].
$$

**What each term means**
- ATE: average effect in the population of interest.
- ATT: average effect for treated units (often what DiD identifies).

#### 3) Identification vs estimation (and why this distinction matters)

> **Identification** answers: "Which assumptions turn observed data into a causal effect?"

> **Estimation** answers: "Given those assumptions, how do we compute the effect?"

Example:
- DiD identification assumption: **parallel trends** (treated and control would have evolved similarly absent treatment).
- DiD estimator: a difference of differences (or a TWFE regression).

Regression output (a coefficient + p-value) is **estimation**. It is *not* identification.

#### 4) Selection bias in one equation (why naive comparisons fail)

A naive "treated vs control" difference in observed means is:
$$
\mathbb{E}[Y \mid D=1] - \mathbb{E}[Y \mid D=0].
$$

Add and subtract $\mathbb{E}[Y(0) \mid D=1]$ to decompose it:

$$
\underbrace{\mathbb{E}[Y(1) \mid D=1] - \mathbb{E}[Y(0) \mid D=1]}_{\text{ATT}}
\; + \;
\underbrace{\mathbb{E}[Y(0) \mid D=1] - \mathbb{E}[Y(0) \mid D=0]}_{\text{selection bias}}.
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
- "Treatment" is a specific, comparable intervention (not a vague label).

3) **Exchangeability / exogeneity (design-specific)**
- Some condition that makes the treatment "as good as random" *after conditioning or differencing*.
  - FE uses within-unit changes to remove time-invariant confounding.
  - DiD uses parallel trends to remove common time shocks.
  - IV uses exclusion + relevance to isolate quasi-random variation in treatment.

4) **Overlap (when conditioning is used)**
- There are both treated and untreated units for the covariate patterns you analyze.

#### 6) Estimation mechanics (high level): mapping designs → estimators

This repo focuses on three workhorse designs:

- **Panel fixed effects (FE / TWFE)**
  Model: $Y_{it} = \beta'X_{it} + \alpha_i + \gamma_t + \varepsilon_{it}$
  Identifying variation: within-unit changes over time.

- **Difference-in-differences (DiD / event study)**
  Model: $Y_{it} = \beta D_{it} + \alpha_i + \gamma_t + \varepsilon_{it}$
  Identifying variation: treated vs control *changes*.

- **Instrumental variables (IV / 2SLS)**
  Structural: $Y = \beta X + u$ with $\mathrm{Cov}(X,u) \neq 0$
  Instrument: $Z$ shifts $X$ but is excluded from $Y$ except through $X$.

#### 7) Inference: why standard errors are part of the design

Even if your estimator is unbiased under identification assumptions, **inference can fail** if you treat dependent observations as independent.

Common dependence patterns:
- counties in the same state share shocks → errors correlated within state,
- repeated observations over time → serial correlation.

That is why causal notebooks emphasize **clustered standard errors** and "number of clusters" reporting.

#### 8) Diagnostics + robustness (minimum set you should practice)

At least three diagnostics should become habits:

1) **Design diagnostic:** does the identifying assumption look plausible?
   - DiD: pre-trends / leads in an event study.
   - IV: first-stage strength and exclusion plausibility.

2) **Specification sensitivity:** does the estimate move a lot if you change the spec?
   - add/remove plausible controls,
   - change time window,
   - change clustering level.

3) **Falsification / placebo:** does the method "find effects" where none should exist?
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
- "Controls" do **not** automatically eliminate bias.
- Better fit ($R^2$) does **not** imply better identification.

#### 10) Why this repo uses "semi-synthetic" exercises

Some notebooks add a **known treatment effect** to a real outcome to create a semi-synthetic truth.
That lets you verify whether an estimator can recover the known effect, without pretending the dataset is a real policy evaluation.

<details>
<summary>Optional: very light DAG intuition (one paragraph)</summary>

A causal diagram (DAG) is a picture of assumptions: arrows represent direct causal links.
Confounding is "backdoor paths" from $D$ to $Y$ (e.g., ability affects both schooling and earnings).
Designs like FE/DiD/IV are ways to block backdoor paths using time differencing, group comparisons, or instruments.

</details>

#### Exercises (do these with the matching notebooks)

- [ ] Pick one notebook question and write the potential outcomes $Y(1), Y(0)$ in words.
- [ ] Define whether you care about ATE or ATT in that notebook and explain why.
- [ ] Write the selection-bias decomposition for a naive comparison in your context.
- [ ] List 3 threats to identification for your design (confounding, anticipation, spillovers, measurement error, …).
- [ ] Write one placebo test you could run and what a "failure" would look like.
- [ ] Explain in 4 sentences why a p-value is not a substitute for identification.

### Panel Fixed Effects (FE): "Compare a unit to itself over time"

Fixed effects are one of the most common ways to reduce bias from **time-invariant unobservables** in panel data. In health economics, panel FE is arguably *the* workhorse method: it underpins studies of hospital quality, insurance expansions, provider behavior, and patient outcomes. If you learn one causal method thoroughly, this is the one.

#### 1) Intuition (plain English)

Panels are powerful because they let you control for factors that differ across units but are roughly constant over time:
- geography,
- long-run institutions,
- baseline demographics,
- "culture" or other hard-to-measure features.

**Story example (county/poverty):** Counties differ in baseline poverty for many persistent reasons.
If you want to study how changes in unemployment relate to changes in poverty, it is more credible to compare:
- *the same county in different years*
than to compare two different counties once.

**Story example (hospital quality):** Hospitals differ in persistent ways: teaching status, urban vs rural location, patient demographics, management culture, and historical investment in equipment. Suppose you want to know whether higher nurse staffing ratios reduce 30-day mortality. A naive cross-sectional comparison of high-staffing and low-staffing hospitals is confounded by all of these persistent differences. With panel data (hospitals observed over multiple years), FE lets you ask: "When *this particular hospital* increased its nurse staffing ratio, did *its own* mortality rate fall?" This strips away everything about the hospital that does not change over time.

**Story example (insurance and health):** States differ enormously in baseline health, income, demographics, and political culture. If you want to study the effect of Medicaid expansion on population health outcomes, comparing states that expanded to states that did not is confounded by all of those baseline differences. With state-year panel data and state FE, you compare each state to itself before and after expansion, removing time-invariant state characteristics from the comparison.

**Story example (patient outcomes):** Longitudinal patient records track the same individuals over time. If you want to study whether a new diabetes management protocol reduces HbA1c levels, patient FE controls for each patient's baseline health, genetics, lifestyle, and socioeconomic status -- all the persistent factors that make some patients sicker than others. Identification comes from within-patient changes: when *this patient* started the protocol, did *their own* HbA1c change?

#### 2) Notation + setup (define symbols)

Let:
- $i = 1,\dots,N$ index entities (hospitals, states, patients, counties),
- $t = 1,\dots,T$ index time (years, quarters),
- $Y_{it}$ be the outcome (e.g., 30-day mortality rate, uninsurance rate, HbA1c),
- $X_{it}$ be a $K \times 1$ vector of regressors (e.g., nurse staffing ratio, Medicaid enrollment),
- $\alpha_i$ be an entity-specific intercept (unit FE),
- $\gamma_t$ be a time-specific intercept (time FE),
- $\varepsilon_{it}$ be the remaining error.

The two-way fixed effects (TWFE) model is:

$$
Y_{it} = X_{it}'\beta + \alpha_i + \gamma_t + \varepsilon_{it}.
$$

**What each term means**
- $X_{it}'\beta$: the part explained by observed covariates.
- $\alpha_i$: all time-invariant determinants of $Y$ for entity $i$ (observed or unobserved). For a hospital, this absorbs teaching status, location, baseline patient mix, management quality, etc. For a state, this absorbs geography, political culture, historical health infrastructure, etc.
- $\gamma_t$: shocks common to all entities in time $t$ (recessions, nationwide policy changes, secular trends in medical technology, flu seasons).
- $\varepsilon_{it}$: idiosyncratic shocks not captured above -- the unit-time-specific variation that identifies $\beta$.

Matrix form (stack observations by time within entity):

$$
\mathbf{y} = \mathbf{X}\beta + \mathbf{A}\alpha + \mathbf{G}\gamma + \varepsilon
$$

where:
- $\mathbf{A}$ is the entity-dummy design matrix,
- $\mathbf{G}$ is the time-dummy design matrix.

#### 3) What FE "assumes" (identification conditions)

FE is not magic; it removes *time-invariant* confounding, not everything.

A common identification condition is **strict exogeneity**:

$$
\mathbb{E}[\varepsilon_{it} \mid X_{i1}, \dots, X_{iT}, \alpha_i] = 0
\quad \text{for all } t.
$$

**What this means**
- After controlling for unit FE, time FE, and the full history of $X$, the remaining shocks are mean-zero.
- If today's shock changes future $X$ (feedback), strict exogeneity can fail.

Other requirements:
- **Within-unit variation:** the regressor must change over time within entities.
- **No perfect multicollinearity:** you cannot include variables fully determined by FE (e.g., a pure time trend plus full time FE can collide depending on spec).

<details>
<summary>Optional: dynamic panels (why lagged Y can be tricky)</summary>

If you include $Y_{i,t-1}$ as a regressor, FE can create "Nickell bias" when $T$ is small.
This repo does not focus on dynamic panel estimators (Arellano-Bond), but you should know this pitfall exists.

</details>

#### 4) What FE can and cannot control for

This is perhaps the most important conceptual point about fixed effects. Understanding the boundary of what FE eliminates is what separates credible applied work from cargo-cult econometrics.

**What entity FE removes (time-invariant confounders):**

Anything about a unit that does not change over the sample period is absorbed by $\alpha_i$. Examples:

| Domain | Absorbed by entity FE |
|---|---|
| Hospitals | Teaching status, urban/rural location, bed count (if stable), founding mission, historical reputation, baseline patient demographics |
| States | Geography, climate, political culture, constitutional provisions, historical health infrastructure, persistent income level |
| Patients | Genetics, sex, race, baseline chronic conditions, socioeconomic background, health literacy |
| Counties | Land area, distance to metro area, historical industry base, long-run demographic composition |

You do not need to measure these. You do not need to know what they are. FE sweeps out *all* of them simultaneously, as long as they are constant over time.

**What time FE removes (common time shocks):**

Anything that hits all units equally in a given period is absorbed by $\gamma_t$. Examples:
- A nationwide recession that raises unemployment everywhere,
- Federal policy changes (e.g., ACA implementation, Medicare payment reforms),
- Secular trends in medical technology or diagnostic coding,
- Seasonal flu patterns (if using quarterly/monthly data),
- Nationwide changes in data collection methods.

**What TWFE (entity + time FE together) still cannot remove:**

The combination of entity FE and time FE is powerful -- it handles both permanent unit differences and common time shocks. But it leaves **time-varying, unit-specific confounders** on the table. These are things that change *differently across units over time*.

Concrete examples of what TWFE *cannot* handle:

- **Hospital example:** Hospital A simultaneously hires more nurses *and* installs a new electronic health record system. The mortality drop could be due to either change. Both are time-varying and unit-specific -- neither is absorbed by hospital FE or year FE.

- **State example:** A state expands Medicaid *and* simultaneously experiences a fracking boom that raises incomes. The health improvement could reflect insurance or income. Both vary within-state over time.

- **Patient example:** A patient starts a new medication *and* simultaneously retires and begins exercising. The health improvement could come from either change. Both are time-varying within-patient.

- **County example:** A county gets a new hospital *and* a new large employer arrives in the same year. Improvements in health outcomes could reflect either change.

This is why applied economists often say "FE is necessary but not sufficient." It eliminates a huge class of confounders (everything time-invariant), but the remaining identification argument must explain why the *time-varying* variation in $X$ is plausibly exogenous after conditioning on FE.

#### 5) Estimation mechanics: the "within" transformation (derivation)

The FE estimator can be understood as OLS after removing means.

**Entity FE only (simpler to see):**

Start with:
$$
Y_{it} = X_{it}'\beta + \alpha_i + \varepsilon_{it}.
$$

Take the time average within each entity:
$$
\bar{Y}_i = \bar{X}_i'\beta + \alpha_i + \bar{\varepsilon}_i.
$$

Subtract the entity mean equation from the original:

$$
Y_{it} - \bar{Y}_i = (X_{it} - \bar{X}_i)'\beta + (\varepsilon_{it} - \bar{\varepsilon}_i).
$$

**What this achieves**
- The term $\alpha_i$ disappears (it is constant within entity).
- Identification comes from deviations from the entity's own average.

TWFE ("two-way") uses a version of demeaning that removes both:
- entity averages, and
- time averages,
often implemented by "absorbing" FE rather than explicitly constructing every dummy.

#### 6) Worked numerical example: the within transformation in practice

To make the within transformation concrete, here is a complete numerical example with 3 hospitals observed over 3 years. The outcome is 30-day mortality rate (per 100 admissions), and the regressor is nurse-to-patient ratio.

**Step 1: Raw data**

| Hospital | Year | Mortality ($Y_{it}$) | Nurse Ratio ($X_{it}$) |
|----------|------|----------------------|------------------------|
| A | 2019 | 12.0 | 4.0 |
| A | 2020 | 10.0 | 5.0 |
| A | 2021 | 8.0 | 6.0 |
| B | 2019 | 6.0 | 7.0 |
| B | 2020 | 5.0 | 8.0 |
| B | 2021 | 4.0 | 9.0 |
| C | 2019 | 15.0 | 3.0 |
| C | 2020 | 14.0 | 3.5 |
| C | 2021 | 11.0 | 5.0 |

Notice: Hospital B has lower mortality *and* higher staffing -- cross-sectionally they look different. But is that because staffing *causes* lower mortality, or because Hospital B is a well-funded teaching hospital that is better in every way?

**Step 2: Compute entity (hospital) means**

| Hospital | $\bar{Y}_i$ | $\bar{X}_i$ |
|----------|-------------|-------------|
| A | 10.0 | 5.0 |
| B | 5.0 | 8.0 |
| C | 13.33 | 3.83 |

**Step 3: Demean -- subtract entity means**

| Hospital | Year | $\tilde{Y}_{it} = Y_{it} - \bar{Y}_i$ | $\tilde{X}_{it} = X_{it} - \bar{X}_i$ |
|----------|------|---------------------------------------|---------------------------------------|
| A | 2019 | 2.0 | -1.0 |
| A | 2020 | 0.0 | 0.0 |
| A | 2021 | -2.0 | 1.0 |
| B | 2019 | 1.0 | -1.0 |
| B | 2020 | 0.0 | 0.0 |
| B | 2021 | -1.0 | 1.0 |
| C | 2019 | 1.67 | -0.83 |
| C | 2020 | 0.67 | -0.33 |
| C | 2021 | -2.33 | 1.17 |

**Step 4: OLS on demeaned data**

The FE estimate of $\beta$ is obtained by regressing $\tilde{Y}$ on $\tilde{X}$ (no intercept needed -- demeaned data has mean zero). You can verify:

$$
\hat{\beta}_{FE} = \frac{\sum_{i,t} \tilde{X}_{it} \tilde{Y}_{it}}{\sum_{i,t} \tilde{X}_{it}^2}
$$

Numerator: $(-1)(2) + (0)(0) + (1)(-2) + (-1)(1) + (0)(0) + (1)(-1) + (-0.83)(1.67) + (-0.33)(0.67) + (1.17)(-2.33)$
$= -2 + 0 - 2 - 1 + 0 - 1 - 1.39 - 0.22 - 2.73 = -10.34$

Denominator: $1 + 0 + 1 + 1 + 0 + 1 + 0.69 + 0.11 + 1.37 = 6.17$

$$
\hat{\beta}_{FE} = \frac{-10.34}{6.17} \approx -1.68
$$

**Interpretation:** When a hospital increases its nurse-to-patient ratio by 1 unit *relative to its own average*, its 30-day mortality rate falls by approximately 1.68 points *relative to its own average*. This is a within-hospital comparison -- Hospital B's permanently lower mortality is irrelevant. We are asking: "When Hospital A went from 4.0 to 5.0 to 6.0 nurses per patient, did *Hospital A's own* mortality decline?"

That is what "compare a unit to itself" means concretely.

#### 7) FE vs First Differences

The within (FE) estimator is not the only way to eliminate $\alpha_i$. An alternative is **first differencing (FD)**: instead of subtracting the entity mean, subtract the previous period's value.

Starting from $Y_{it} = X_{it}'\beta + \alpha_i + \varepsilon_{it}$, take first differences:

$$
Y_{it} - Y_{i,t-1} = (X_{it} - X_{i,t-1})'\beta + (\varepsilon_{it} - \varepsilon_{i,t-1}).
$$

Again, $\alpha_i$ drops out.

**When are FE and FD equivalent?**
- With exactly $T = 2$ periods, FE and FD produce *identical* estimates. The within transformation and first differencing are algebraically the same operation when there are only two time points.

**When do they differ?**
- With $T > 2$, they generally differ because they weight the data differently. FE uses deviations from the full entity mean; FD uses period-to-period changes.
- Under the assumption of serially uncorrelated $\varepsilon_{it}$, FE is more efficient (uses more information).
- If $\varepsilon_{it}$ follows a random walk (highly persistent errors), FD is more efficient because $\varepsilon_{it} - \varepsilon_{i,t-1}$ is well-behaved while $\varepsilon_{it} - \bar{\varepsilon}_i$ is not.

**Practical guidance:** Most applied health economics papers use FE rather than FD, but comparing FE and FD estimates is a useful robustness check. If the two diverge substantially, it suggests the error structure matters and you should investigate further.

#### 8) Frisch-Waugh-Lovell (FWL) interpretation (how software implements FE)

FWL says: if you want $\beta$ in a regression with controls, you can:
1) residualize $Y$ on the controls,
2) residualize $X$ on the controls,
3) regress residualized $Y$ on residualized $X$.

In FE models, "controls" include thousands of entity/time dummies. Software (like `linearmodels`) uses the same logic efficiently.

#### 9) Practical code template: PanelOLS with entity and time effects

Below is a complete, annotated example showing how to set up and estimate a TWFE model using `linearmodels`. This is the pattern you will use in the notebook.

```python
import pandas as pd
from linearmodels.panel import PanelOLS

# ---- 1. Load data (example: hospital-year panel) ----
df = pd.read_csv("data/processed/hospital_panel.csv")

# ---- 2. Set up the multi-index (entity, time) ----
# PanelOLS requires a MultiIndex with entity as level 0 and time as level 1.
df = df.set_index(["hospital_id", "year"])

# ---- 3. Define dependent variable and regressors ----
y = df["mortality_rate"]
X = df[["nurse_ratio", "bed_occupancy", "pct_medicare"]]

# ---- 4. Fit the TWFE model ----
# entity_effects=True adds hospital fixed effects (alpha_i)
# time_effects=True adds year fixed effects (gamma_t)
model = PanelOLS(y, X, entity_effects=True, time_effects=True)

# cov_type='clustered' with cluster_entity=True clusters SE at the entity level
result = model.fit(cov_type="clustered", cluster_entity=True)

# ---- 5. Inspect output ----
print(result.summary)

# Key things to check in the output:
# - Coefficients: within-unit associations after removing FE
# - Std Errors: clustered at entity level
# - F-statistic: joint significance of regressors
# - R-squared (within): how much within-unit variation X explains
# - Number of entities and time periods

# ---- 6. Extract specific results programmatically ----
beta_nurse = result.params["nurse_ratio"]
se_nurse = result.std_errors["nurse_ratio"]
print(f"Nurse ratio effect: {beta_nurse:.3f} (SE = {se_nurse:.3f})")
```

**Notes on the multi-index:** The most common error when starting with `linearmodels` is forgetting to set the MultiIndex. If your data has columns `entity_id` and `year`, call `df.set_index(["entity_id", "year"])` *before* creating the PanelOLS object. The first level must be the entity; the second must be time.

**Notes on `cov_type` options:**
- `"unadjusted"`: classical OLS SE (assumes homoskedastic, independent errors -- almost never appropriate for panels).
- `"robust"`: heteroskedasticity-robust (HC) SE, but still assumes independence across observations.
- `"clustered"`: cluster-robust SE. Use `cluster_entity=True` to cluster at entity level, or pass a custom `clusters` Series for clustering at a different level (e.g., state).
- `"kernel"`: HAC (Newey-West style) SE for serial correlation.

#### 10) Inference: why clustering is common in panels

Panel errors are often correlated:
- within entity over time (serial correlation),
- within higher-level groups (e.g., state shocks),
- within time (common shocks).

Robust (HC) SE handle heteroskedasticity but not arbitrary within-cluster correlation.
That's why you often see **clustered SE** in applied FE work. (The next section covers clustering in detail.)

#### 11) Diagnostics + robustness (at least 3)

1) **Pooled vs FE comparison**
- Fit pooled OLS and FE; if coefficients move a lot, unobserved heterogeneity likely mattered.

2) **Within-variation check**
- For each regressor, confirm it varies within entity. If not, FE cannot estimate it.

3) **Sensitivity to time FE**
- Add/remove time FE; if results change materially, time shocks matter and must be modeled.

4) **Influential units**
- Check whether a handful of entities drive the estimate (outliers / leverage).

5) **FE vs FD comparison**
- If FE and first-differenced estimates diverge substantially, the serial correlation structure of errors matters and warrants investigation.

#### 12) Interpretation + reporting (what FE coefficients mean)

In a TWFE regression, interpret $\beta_j$ as:

> the association between *within-entity changes* in $x_j$ and *within-entity changes* in $Y$, after removing common time shocks.

**What this does NOT mean**
- FE does **not** guarantee causality (time-varying omitted variables can remain).
- FE does **not** solve reverse causality (shocks can affect both $X$ and $Y$).
- FE does **not** "fix" measurement error (it can make some bias worse -- since the within transformation removes signal along with noise, the signal-to-noise ratio can deteriorate, amplifying attenuation bias).

#### 13) Health economics applications

Panel FE is the backbone of empirical health economics. Here are examples of influential published studies that rely on this method:

- **Finkelstein et al. (2012), "The Oregon Health Insurance Experiment"**: Used panel data from Oregon's Medicaid lottery to study the effect of Medicaid coverage on health care utilization, financial strain, and health outcomes. While the lottery provided random assignment, the panel structure with individual FE was critical for measuring within-person changes over time.

- **Gruber & McKnight (2016), "Controlling Health Care Costs Through Limited Network Insurance Plans"**: Used employer-year panel data with employer FE to estimate how switching to limited-network insurance plans affected health care spending, isolating the within-employer change from persistent differences across employers.

- **Duggan (2000), "Hospital Ownership and Public Medical Spending"**: Used hospital-year panel data with hospital FE to study how the shift from public to private ownership affected hospital behavior and costs, comparing each hospital to itself before and after ownership changes.

- **Currie & MacLeod (2017), "Diagnosing Expertise: Human Capital, Decision Making, and Performance Among Physicians"**: Used physician-year panel data with physician FE to separate the effect of physician skill from patient selection, exploiting within-physician variation in patient mix over time.

These papers share a common structure: panel data with entity FE to absorb persistent unobserved heterogeneity, combined with careful arguments about why the remaining within-unit variation is plausibly exogenous.

#### Exercises

- [ ] Take one regressor in the panel notebook and compute $x_{it} - \bar{x}_i$ manually; confirm the mean is ~0 within each entity.
- [ ] Explain in words what variation identifies $\beta$ in a TWFE model.
- [ ] Name one time-invariant variable you *wish* you could estimate and explain why FE absorbs it.
- [ ] Fit pooled OLS and TWFE; write 3 sentences explaining why the coefficients differ.
- [ ] Try adding time FE and report whether the coefficient on your main regressor is stable.
- [ ] Using the worked numerical example as a template, construct a 2-hospital, 2-period example where FE and FD give the same answer. Verify algebraically.
- [ ] Give an example of a time-varying confounder in a hospital staffing study that TWFE would *not* remove.

---

### Clustered Standard Errors: Inference when observations move together

Clustered standard errors are about **honest uncertainty** when errors are correlated within groups. In health economics, where data is almost always structured in groups (patients within hospitals, counties within states, providers within health systems), clustering is not optional -- it is a baseline requirement for credible inference.

#### 1) Intuition (plain English)

Regression formulas for standard errors often assume each row is "independent."
In economics, that is frequently false.

**Story example (counties in states):** Counties in the same state share:
- state policy changes,
- state-level business cycles,
- housing markets and migration flows.

If your residuals are correlated within states and you ignore that, your standard errors are often too small, making effects look "significant" when they are not.

**Story example (hospitals in health systems):** Hospitals within the same health system share administrative practices, purchasing contracts, IT systems, and often clinical protocols. If you study the effect of a policy that varies across health systems, hospitals within the same system will have correlated residuals. Treating each hospital as independent overstates your effective sample size.

#### 2) The effective sample size problem: why clustering matters so much

This is the single most important intuition about clustering. Consider a concrete example.

**The classroom analogy:** Imagine you survey 100 students in 5 classrooms of 20 students each. You want to estimate average math ability. If every student were truly independent, you would have 100 independent data points and a tight confidence interval.

But students in the same classroom share a teacher. A great teacher raises all 20 students' scores; a poor teacher depresses all 20. If the teacher effect is strong, knowing one student's score in Classroom A tells you a lot about the other 19. Those 20 observations are not 20 independent pieces of information -- they are closer to *one* piece of information (the classroom mean) measured with some within-classroom noise.

Your effective sample size is somewhere between 5 (the number of classrooms) and 100 (the number of students), depending on how strong the within-classroom correlation is. If you compute standard errors pretending you have 100 independent observations, you will be wildly overconfident.

**The intraclass correlation coefficient (ICC):** The degree of within-cluster dependence is measured by the ICC, denoted $\rho$:

$$
\rho = \frac{\sigma^2_{\text{between clusters}}}{\sigma^2_{\text{between clusters}} + \sigma^2_{\text{within cluster}}}
$$

When $\rho = 0$, observations are independent and clustering makes no difference. When $\rho = 1$, all observations within a cluster are identical and your effective sample size equals the number of clusters. In practice, even moderate values of $\rho$ (e.g., 0.05-0.10) can dramatically inflate standard errors when clusters are large.

**The variance inflation formula:** For a cluster of size $m$ with intraclass correlation $\rho$, the variance of the cluster mean is inflated by a factor of:

$$
1 + (m - 1)\rho
$$

relative to what you would compute assuming independence. With $m = 20$ and $\rho = 0.10$, this factor is $1 + 19 \times 0.10 = 2.9$. Your standard errors should be $\sqrt{2.9} \approx 1.7$ times larger than naive SE. With $m = 50$ and $\rho = 0.10$, the factor is $5.9$ and SE should be $\sqrt{5.9} \approx 2.4$ times larger. Ignoring this leads to rejecting null hypotheses far too often.

**Health economics implication:** In health economics, clusters can be large. A state-level policy studied with county-level data might have 50+ counties per state. A hospital-level study with patient-level data might have thousands of patients per hospital. Even modest within-cluster correlation makes naive SE deeply misleading.

#### 3) Notation + setup (define symbols)

Consider a linear regression in stacked form:

$$
\mathbf{y} = \mathbf{X}\beta + \mathbf{u}.
$$

Let:
- $n$ be the number of observations (rows),
- $K$ be the number of regressors,
- $g \in \{1,\dots,G\}$ index clusters (e.g., states),
- $\mathbf{X}_g$ be the rows of $\mathbf{X}$ in cluster $g$,
- $\mathbf{u}_g$ be the residual vector in cluster $g$.

#### 4) What clustering assumes (and what it relaxes)

Cluster-robust SE relax the "independent errors" assumption **within** clusters:
- errors can be heteroskedastic and correlated in arbitrary ways within a cluster.

But they still assume something like **independence across clusters**:
- cluster $g$ shocks are independent of cluster $h$ shocks (for $g \neq h$).

This is why **choosing the right cluster level is part of the design**, not a technical afterthought.

#### 5) Estimation mechanics: the cluster-robust covariance (sandwich form)

The OLS coefficient estimate $\hat\beta$ does not change when you change SE.
What changes is the estimated variance of $\hat\beta$.

The cluster-robust (one-way) covariance estimator is:

$$
\widehat{\mathrm{Var}}_{\text{CL}}(\hat\beta)
= (\mathbf{X}'\mathbf{X})^{-1}
\left(\sum_{g=1}^{G} \mathbf{X}_g' \hat{\mathbf{u}}_g \hat{\mathbf{u}}_g' \mathbf{X}_g \right)
(\mathbf{X}'\mathbf{X})^{-1}.
$$

**What each term means**
- "Bread": $(\mathbf{X}'\mathbf{X})^{-1}$ is the same matrix you see in OLS.
- "Meat": the sum over clusters aggregates within-cluster residual covariance. The outer product $\hat{\mathbf{u}}_g \hat{\mathbf{u}}_g'$ allows *all* pairwise correlations within cluster $g$ to contribute -- this is what makes it robust to arbitrary within-cluster dependence.
- This estimator reduces to robust (HC) SE when each observation is its own cluster ($G = n$).

Many libraries also apply small-sample corrections (especially when $G$ is not huge). A common correction multiplies by $\frac{G}{G-1} \cdot \frac{n-1}{n-K}$.

#### 6) HC vs Clustered vs HAC: which standard errors when?

There are several types of "robust" standard errors, each designed for a different dependence structure. Choosing the wrong one can be just as misleading as using classical SE.

| SE Type | What it handles | What it assumes | When to use |
|---------|----------------|-----------------|-------------|
| **Classical (OLS)** | Nothing (assumes $\varepsilon \sim \text{iid}$) | Homoskedastic, independent errors | Almost never in practice |
| **HC (robust / White)** | Heteroskedasticity | Independence across observations | Cross-sectional data with no group structure |
| **Clustered** | Heteroskedasticity + arbitrary within-cluster correlation | Independence *across* clusters | Panel data; grouped data; treatment assigned at group level |
| **HAC (Newey-West)** | Heteroskedasticity + serial correlation (up to some lag) | Correlation dies off with distance in time | Single time series or short panels with few entities |
| **Two-way clustered** | Arbitrary correlation along two dimensions (e.g., entity and time) | Independence across both dimensions simultaneously | Panel data where shocks are correlated both within entity over time and across entities within time |

**Decision rule for health economics:**
- If you have panel data (entities observed over time), **clustered SE at the entity level** is the default starting point.
- If treatment is assigned at a higher level (e.g., state policy in county data), **cluster at the treatment level**.
- If you have a single long time series, use **HAC**.
- If you have a pure cross-section with no group structure, **HC** may suffice.

#### 7) When to cluster at what level

The question "at what level should I cluster?" is one of the most important practical decisions in applied econometrics. The answer depends on the structure of the data and the source of treatment variation.

**General principles:**

1. **Cluster at the level of treatment assignment (at minimum).** If a policy is assigned at the state level, you must cluster at least at the state level, even if your data is at the county or individual level. The logic: all units within a treated state share the same treatment status, so their treatment-related residuals are mechanically correlated.

2. **Cluster at the level where shocks are correlated.** If counties within the same state share unobserved economic shocks (even absent treatment), those shocks correlate the residuals and clustering is needed.

3. **When in doubt, cluster at the higher level.** Clustering at a level that is too low (e.g., county when the relevant correlation is at the state level) under-adjusts SE and leads to over-rejection. Clustering at a level that is too high is conservative (SE may be larger than necessary) but does not cause false positives.

4. **Panel data with repeated observations on the same entity:** cluster at the entity level at minimum, because serial correlation within an entity is almost always present.

**Common health economics scenarios:**

| Scenario | Data level | Treatment varies at | Cluster at |
|----------|-----------|-------------------|------------|
| State Medicaid expansion | County-year | State | State |
| Hospital pay-for-performance policy | Hospital-year | Hospital | Hospital (or health system) |
| Physician prescribing guideline | Physician-patient | Physician | Physician |
| State scope-of-practice law | Provider-year | State | State |
| Patient-level drug trial in multi-site study | Patient | Site | Site |
| County health rankings and spending | County-year | County | State (if state policies drive correlation) |

**Reference:** Abadie, Athey, Imbens, and Wooldridge (2023), "When Should You Cluster Standard Errors? New Wisdom on an Old Question," provides a modern framework. Their key insight: clustering is about the *design* (how treatment was assigned and how the sample was drawn), not just about correlation in residuals. If treatment is assigned by cluster and/or the sample is drawn by cluster, you should cluster. If treatment is assigned at the individual level in a simple random sample, clustering may not be necessary even if residuals are correlated.

#### 8) Expanded code example: clustered SE in practice

```python
import pandas as pd
from linearmodels.panel import PanelOLS

# ---- Load and index the panel ----
df = pd.read_csv("data/processed/census_county_panel.csv")
df = df.set_index(["county_fips", "year"])

y = df["poverty_rate"]
X = df[["unemployment_rate", "median_income_log"]]

# ---- Fit TWFE model ----
model = PanelOLS(y, X, entity_effects=True, time_effects=True)

# ---- Compare different SE approaches ----

# 1. Clustered at entity (county) level
result_entity = model.fit(cov_type="clustered", cluster_entity=True)

# 2. Clustered at a higher level (state)
# Create a state FIPS column from county FIPS (first 2 digits)
df["state_fips"] = df.index.get_level_values("county_fips").astype(str).str[:2]
state_clusters = df["state_fips"]
result_state = model.fit(cov_type="clustered", clusters=state_clusters)

# 3. Robust (HC) SE for comparison
result_robust = model.fit(cov_type="robust")

# ---- Compare SE across approaches ----
comparison = pd.DataFrame({
    "HC (robust)": result_robust.std_errors,
    "Clustered (county)": result_entity.std_errors,
    "Clustered (state)": result_state.std_errors,
})
print("Standard errors under different assumptions:")
print(comparison.round(4))
print(f"\nNote: Point estimates are identical across all three.")
print(f"Number of counties: {df.index.get_level_values(0).nunique()}")
print(f"Number of states: state_clusters.nunique()")

# ---- What to look for ----
# If clustered SE >> HC SE, within-cluster correlation is substantial.
# If state-clustered SE >> county-clustered SE, state-level shocks matter.
# Always report the version that matches your design.
```

#### 9) Inference pitfalls (important!)

1) **Few clusters problem**
- When the number of clusters $G$ is small (rule of thumb: < 30-50), cluster-robust inference can be unreliable. The cluster-robust variance estimator is consistent as $G \to \infty$, not as $n \to \infty$. With 10 states, you effectively have 10 "observations" for estimating the variance, which is not many.
- In serious applied work, people use remedies like **wild cluster bootstrap**; we do not implement that here, but you should know it exists. The `wildboottest` Python package implements this.
- For health economics: if your treatment varies at the state level and you only have 10-15 states in your sample, be upfront about the fragility of inference.

2) **Wrong clustering level**
- If treatment is assigned at the state level, clustering below that (e.g., county) can be too optimistic.
- A common rule: cluster at the level of treatment assignment or the level of correlated shocks, whichever is higher.

3) **Serial correlation in DiD**
- Classic results (Bertrand, Duflo, and Mullainathan, 2004) show naive SE can be severely biased in DiD with serial correlation.
- Clustering at the entity level is often the minimum fix.

4) **Moulton problem**
- When a group-level variable (e.g., state policy) is assigned to individual-level data, the correlation between same-group observations inflates the t-statistic dramatically. This was formalized by Moulton (1990) and is one of the original motivations for clustered SE.

#### 10) Diagnostics + robustness (at least 3)

1) **Report cluster count**
- Always report $G$ (number of clusters). If $G$ is tiny, treat inference as fragile.

2) **Sensitivity to clustering level**
- Try clustering by county vs by state (or time). If SE change a lot, dependence is important.

3) **Residual dependence check**
- Plot residuals by cluster over time or compute within-cluster autocorrelation.

4) **Aggregation robustness (DiD-style)**
- As a sanity check, aggregate to the treatment-assignment level and see if conclusions are similar.

5) **Compare HC and clustered SE**
- If clustered SE are substantially larger than HC SE, within-cluster correlation is meaningful and clustering is necessary. If they are similar, the clustering level may not matter much (but err on the side of clustering).

#### 11) Interpretation + reporting

Clustered SE change your uncertainty, not your point estimate.
So the right way to write results is:
- coefficient (effect size),
- clustered SE (uncertainty),
- cluster level and number of clusters,
- design assumptions.

**Example write-up:** "We estimate that a one-unit increase in the nurse-to-patient ratio is associated with a 1.68 percentage point decrease in 30-day mortality ($\hat{\beta} = -1.68$, SE = 0.52, $p < 0.01$). Standard errors are clustered at the hospital level (G = 312) to account for serial correlation in hospital-specific shocks."

**What this does NOT mean**
- Clustering does not fix bias from confounding or misspecification.
- Clustering does not "prove" causality; it only helps prevent overconfident inference.
- Clustering does not change your point estimate -- only your uncertainty about it.

#### Exercises

- [ ] In a panel regression, compute naive (HC) SE and clustered SE; compare the ratio for your main coefficient.
- [ ] Explain in words why state-level shocks make county-level rows dependent.
- [ ] Try clustering by entity vs by state; write 3 sentences about which is more defensible and why.
- [ ] If you had only 8 states, what would make you cautious about cluster-robust p-values?
- [ ] Using the classroom analogy, compute the variance inflation factor for a cluster of size 30 with ICC = 0.15. How much larger should the SE be compared to naive SE?
- [ ] For a study of state Medicaid expansion using county-level data, explain why clustering at the county level is insufficient.

---

### Hausman Test: FE vs RE (Preview)

The choice between fixed effects and random effects is not arbitrary -- it depends on whether the unobserved unit effect $\alpha_i$ is correlated with the regressors. The **Hausman test** formalizes this choice by testing whether FE and RE estimates are systematically different. If they diverge, the RE assumption ($\mathrm{Cov}(\alpha_i, X_{it}) = 0$) is violated and FE is preferred. The next guide, `01a_random_effects_hausman`, covers the RE model, partial demeaning, the Hausman test, and the Mundlak (1978) approach in detail.

---

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
- Clustering at the wrong level (too low relative to treatment assignment).
- For DiD: not checking pre-trends (leads) before interpreting effects.
- For IV: using weak instruments (no meaningful first stage).
- Confusing "FE controls for everything" with "FE controls for time-invariant things only."
- Forgetting that measurement error bias is often *worse* after FE demeaning (attenuation bias amplified).

<a id="summary"></a>
## Summary + Suggested Readings

You now have a toolkit for causal estimation under explicit assumptions (FE/DiD/IV).
The goal is disciplined thinking: identification first, estimation second.

**Key takeaways from this guide:**
- Panel FE eliminates time-invariant confounders by comparing each unit to itself over time. It is the default tool in health economics for a reason: hospitals, states, and patients differ in persistent ways that are hard to measure.
- TWFE (entity + time FE) is more powerful than entity FE alone because it also removes common time shocks. But it cannot remove time-varying, unit-specific confounders.
- Clustered SE are essential whenever observations within groups are not independent. The choice of clustering level is a design decision, not a technical afterthought.
- The number of clusters determines the reliability of cluster-robust inference. With few clusters (< 30), standard asymptotic approximations may be poor.

Suggested readings:
- Angrist & Pischke: *Mostly Harmless Econometrics* (design-based causal inference, FE/IV/DiD)
- Wooldridge: *Econometric Analysis of Cross Section and Panel Data* (FE/IV foundations, RE, clustering)
- Cameron & Miller (2015): "A Practitioner's Guide to Cluster-Robust Inference" (*Journal of Human Resources*) -- the definitive practical guide to when and how to cluster
- Abadie, Athey, Imbens & Wooldridge (2023): "When Should You Cluster Standard Errors?" -- modern framework linking clustering to research design
- Bertrand, Duflo & Mullainathan (2004): "How Much Should We Trust Differences-in-Differences Estimates?" -- foundational paper on serial correlation and inference in DiD
- Currie, MacLeod & Van Parys (2010): "Provider Practice Style and Medical Decision Making" -- example of physician FE in health economics
