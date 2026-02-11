# Guide: Difference-in-Differences and Event Studies

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
  - [Difference-in-Differences (DiD)](#did)
  - [Event Studies (Leads/Lags)](#event-studies)
  - [Staggered Adoption and Modern DiD](#staggered)
  - [Diagnostics and Robustness](#diagnostics)
  - [Practical Code Templates](#code-templates)
- [Common Mistakes](#common-mistakes)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/07_causal/02_difference_in_differences_event_study.ipynb`.

**Prerequisites.** This guide assumes you have read
[Guide 01: Panel Fixed Effects and Clustered SE](01_panel_fixed_effects_clustered_se.md),
which covers the Core Causal Inference primer (potential outcomes, selection bias,
identification vs estimation, the ATE/ATT distinction) and the full treatment of
clustered standard errors (sandwich estimator, few-clusters problem, choosing the
cluster level). We will not repeat that material here. If terms like "parallel
trends" or "ATT" feel unfamiliar, start with Guide 01 first.

**Why DiD dominates health economics.** Difference-in-differences is the single
most common identification strategy in modern health economics and health
services research. The reason is institutional: health policy changes at discrete
moments in time for identifiable groups of people, creating natural experiments.
Medicaid expansions happen in some states but not others. The ACA dependent
coverage mandate applied to adults under 26 but not those over 26. Hospital
mergers affect some markets but leave neighboring markets untouched. Certificate-
of-need law repeals happen in specific states in specific years. In each case,
DiD compares changes in outcomes for affected units against changes for unaffected
units, removing both permanent group differences and common time trends.

This guide covers two closely related tools:
1. **Difference-in-differences (DiD)** -- the foundational design.
2. **Event studies (leads/lags)** -- the dynamic extension that shows how effects
   evolve and provides the key diagnostic for the identifying assumption.

### Key Terms (DiD-specific)

- **Treatment group**: units that receive the intervention (e.g., Medicaid expansion states).
- **Control group**: units that do not receive the intervention (non-expansion states).
- **Pre-period**: time before treatment begins.
- **Post-period**: time after treatment begins.
- **Parallel trends**: the assumption that treated and control groups would have followed the same trajectory in the absence of treatment.
- **Treatment adoption**: the moment a unit switches from untreated to treated.
- **Staggered adoption**: different units adopt treatment at different times.
- **Event time**: calendar time re-centered around each unit's adoption date ($k = t - T_i$).
- **Leads**: event-time dummies for periods before adoption ($k < 0$); used to diagnose pre-trends.
- **Lags**: event-time dummies for periods after adoption ($k \ge 0$); these capture the treatment effect path.
- **Anticipation effects**: changes in behavior before formal treatment onset (e.g., hospitals adjusting staffing before a policy takes effect).
- **Pre-trends**: differential trends between treated and control units before treatment; a red flag for the parallel trends assumption.
- **TWFE (two-way fixed effects)**: a regression with unit fixed effects and time fixed effects, the standard DiD estimator.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** for the math, assumptions, and interpretation.
- Open any `<details>` blocks for optional depth on advanced topics.
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- [ ] State the causal question: what treatment, what outcome, what population, what time horizon?
- [ ] Identify the treatment and control groups and justify why the control group is a plausible counterfactual.
- [ ] Write the parallel trends assumption in words for your specific setting.
- [ ] Plot raw group means over time (treated vs control) and visually assess whether pre-trends look parallel.
- [ ] Complete notebook section: Synthetic adoption + treatment assignment.
- [ ] Complete notebook section: TWFE DiD estimation with clustered SE.
- [ ] Complete notebook section: Event study (leads/lags) estimation.
- [ ] Complete notebook section: Diagnostics -- pre-trends test and placebo checks.
- [ ] Report: point estimate, clustered SE, cluster level, number of clusters, and event study plot.

### Alternative Example: A Complete 2x2 DiD Calculation

This example is separate from the notebook data. It walks through a Medicaid
expansion DiD from start to finish.

**Setting.** Suppose 10 states expanded Medicaid in 2014 ("treated") and 10 did
not ("control"). We measure the uninsurance rate (%) in 2013 (pre) and 2015 (post).

```python
import numpy as np
import pandas as pd

# 2x2 group means (uninsurance rate, %)
data = {
    'Group':   ['Expansion', 'Expansion', 'Non-expansion', 'Non-expansion'],
    'Period':  ['Pre (2013)',  'Post (2015)', 'Pre (2013)',    'Post (2015)'],
    'Uninsurance_Rate': [15.0, 9.0, 18.0, 16.0],
}
table = pd.DataFrame(data)
print(table.to_string(index=False))
#        Group       Period  Uninsurance_Rate
#    Expansion   Pre (2013)              15.0
#    Expansion  Post (2015)               9.0
# Non-expansion  Pre (2013)              18.0
# Non-expansion Post (2015)              16.0
```

**Step 1: Within-group changes.**
- Expansion states: 9.0 - 15.0 = **-6.0 pp** (uninsurance fell 6 points).
- Non-expansion states: 16.0 - 18.0 = **-2.0 pp** (uninsurance fell 2 points).

**Step 2: Difference-in-differences.**
$$
\hat{\tau}_{\text{DiD}} = (-6.0) - (-2.0) = -4.0 \text{ pp}.
$$

**Step 3: Interpretation.** The DiD estimate is -4.0 percentage points. This
means that, relative to the secular decline in uninsurance that both groups
experienced (approximated by the -2.0 pp change in non-expansion states),
expansion states saw an *additional* 4.0 pp reduction in uninsurance
attributable to the Medicaid expansion -- under the assumption that without
the expansion, expansion states would have experienced the same -2.0 pp
decline as non-expansion states (parallel trends).

Note that expansion and non-expansion states started at different *levels*
(15% vs 18%). That is fine. DiD does not require equal levels, only equal
trends in the absence of treatment.

```python
# The same calculation in code:
treated_change = 9.0 - 15.0   # -6.0
control_change = 16.0 - 18.0  # -2.0
did_estimate = treated_change - control_change  # -4.0
print(f"DiD estimate: {did_estimate:.1f} pp")
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

<a id="did"></a>
### Difference-in-Differences (DiD): Learning from "Changes vs Changes"

DiD is a workhorse design for causal inference when treatment is not randomized
but changes over time for some units. It removes time-invariant confounders
(through group differencing) and common time shocks (through time differencing).

#### 1) Intuition: Why Compare Changes, Not Levels

If treated and control units differ at baseline -- and they almost always do --
a raw post-treatment comparison conflates the treatment effect with pre-existing
differences. DiD compares *changes*: how much did outcomes change in the treated
group vs the control group over the same period? The difference between those
two changes isolates the treatment effect, because any permanent group difference
cancels out and any common time shock (recession, national policy, seasonal flu)
also cancels out.

#### 2) The 2x2 Framework: Notation and Setup

Define:
- Groups $g \in \{T, C\}$ for treated and control.
- Time periods $t \in \{0, 1\}$ for pre and post.
- $Y_{gt}$ as the average observed outcome in group $g$ at time $t$.
- $Y_{gt}(0)$ as the average potential outcome *without treatment*.
- $Y_{gt}(1)$ as the average potential outcome *with treatment*.

Only the treated group in the post-period receives treatment:
- $Y_{T1} = Y_{T1}(1)$ (treated group, post-period: treated, so we observe $Y(1)$).
- $Y_{T0} = Y_{T0}(0)$ (treated group, pre-period: not yet treated).
- $Y_{C0} = Y_{C0}(0)$ and $Y_{C1} = Y_{C1}(0)$ (control group: never treated).

The DiD estimand is:

$$
\hat{\tau}_{\text{DiD}} = (Y_{T1} - Y_{T0}) - (Y_{C1} - Y_{C0}).
$$

**What each piece means:**
- $Y_{T1} - Y_{T0}$: total change in the treated group (includes both the treatment effect and any common time trend).
- $Y_{C1} - Y_{C0}$: change in the control group (captures the common time trend only).
- Subtracting the second from the first removes the common trend, leaving the treatment effect.

**Worked example (from above):**

|                 | Pre (2013) | Post (2015) | Change |
|-----------------|-----------|------------|--------|
| Expansion       | 15.0%     | 9.0%       | -6.0 pp |
| Non-expansion   | 18.0%     | 16.0%      | -2.0 pp |
| **Difference**  |           |            | **-4.0 pp** |

The -4.0 pp is the ATT: the average treatment effect on the treated states.

#### 3) The Parallel Trends Assumption: Deep Treatment

Parallel trends is the core identifying assumption of DiD:

$$
E[Y_{T1}(0) - Y_{T0}(0)] = E[Y_{C1}(0) - Y_{C0}(0)].
$$

**In words:** absent treatment, the treated group would have experienced the
same change over time as the control group.

**What parallel trends requires:**
- The *trends* (slopes, changes over time) must be the same across groups in the
  counterfactual world without treatment.
- It does NOT require that the groups have the same *levels*. Expansion states
  can start at 15% and non-expansion at 18% -- that is perfectly fine, because
  DiD removes level differences.

**When parallel trends is plausible:**
- Units face similar macro shocks and secular trends.
- Treatment assignment is driven by something (e.g., political composition) that does not also cause differential outcome trends.

**When parallel trends is suspect:**
- Treated and control units are fundamentally different in ways that generate different trajectories (e.g., large urban hospitals vs small rural clinics face different market forces).
- Adoption is driven by the outcome itself -- if states expanded Medicaid *because* their uninsurance was rising faster, treated states were already on a different trajectory.

**The fundamental untestability problem:**
Parallel trends is a statement about the *counterfactual* -- what would have happened absent treatment. We can check pre-period trends, but pre-period parallelism does not guarantee post-period parallelism.

**"Pre-trends are necessary but not sufficient":** Suppose two groups have perfectly parallel uninsurance trends from 2008-2013. In 2014, one group expands Medicaid. Pre-trends look great. But if the expansion states also experienced an unrelated local economic boom starting in 2014, your DiD estimate would wrongly attribute the boom's effect to Medicaid. Pre-trends were fine; parallel trends failed in the post-period. DiD is always an *assumption-based* claim, not a proof.

#### 4) The TWFE Regression: Connecting the 2x2 Math to Regression

With many units and time periods, DiD is estimated via a two-way fixed effects
(TWFE) regression:

$$
Y_{it} = \alpha_i + \gamma_t + \beta D_{it} + \varepsilon_{it}.
$$

**What each term does:**
- $\alpha_i$ (unit FE): absorb time-invariant differences between units.
- $\gamma_t$ (time FE): absorb common time shocks.
- $D_{it}$: treatment indicator (1 if unit $i$ is treated at time $t$).
- $\beta$: the DiD coefficient -- the average treatment effect under parallel trends.

**Equivalence to the 2x2 formula.** In the simple 2x2 case (two groups, two
periods), the OLS coefficient $\hat{\beta}$ from this regression is *exactly*
equal to the difference-of-differences:

$$
\hat{\beta} = (\bar{Y}_{T,\text{post}} - \bar{Y}_{T,\text{pre}}) - (\bar{Y}_{C,\text{post}} - \bar{Y}_{C,\text{pre}}).
$$

**What happens if you omit one set of fixed effects?**
- *Only entity FE (no time FE):* removes level differences but not common time shocks -- a recession would be attributed to treatment.
- *Only time FE (no entity FE):* removes common shocks but not level differences -- you are comparing treated vs control levels, biased at baseline.
- *Neither:* a pooled regression that confounds everything.

TWFE uses *both* because each set removes a different source of confounding.

#### 5) Adding Covariates in DiD: When and Why

The basic TWFE regression can include time-varying covariates:

$$
Y_{it} = \alpha_i + \gamma_t + \beta D_{it} + \mathbf{X}_{it}'\delta + \varepsilon_{it}.
$$

**When covariates help:**
- *Precision:* Covariates that predict the outcome but are unrelated to treatment reduce residual variance (e.g., county age composition when studying health outcomes).
- *Conditional parallel trends:* Sometimes parallel trends holds only after conditioning on $X$. Controlling for time-varying economic indicators can make the assumption more credible when expansion and non-expansion states differ on baseline demographics.

**The "bad controls" warning:**
Do NOT control for variables affected by the treatment (post-treatment variables / mediators).
- Bad: controlling for insurance enrollment when studying Medicaid expansion's effect on health (enrollment is the *mechanism*).
- Bad: controlling for hospital capacity when studying a merger's effect on prices, if the merger changed capacity.
- Good: controlling for county unemployment rate (an economic confounder not caused by the policy).

**Rule of thumb:** Ask "could this variable be affected by the treatment?" If yes, exclude it. If unsure, show results with and without.

#### 6) Health Economics Applications of DiD

DiD is everywhere in health economics. Key examples:

**Medicaid expansion:** Finkelstein et al. (2012) used an Oregon Medicaid lottery; Sommers et al. (2012) used the 2006 Massachusetts reform. The ACA Medicaid expansion (2014+) generated hundreds of DiD studies comparing expansion vs non-expansion states on coverage, utilization, health outcomes, and finances.

**ACA dependent coverage mandate (age 26 provision):** Antwi, Moriya, and Simon (2013) used the age-26 cutoff -- adults 19-25 (treated) vs 27-29 (control), before and after 2010.

**Hospital mergers:** Dafny (2009) compared markets with mergers to similar markets without, measuring price effects. Post-merger quality studies use the same treated-vs-matched-control structure.

**State policy changes:** Tobacco taxes and smoking cessation; scope-of-practice laws for nurse practitioners; certificate-of-need law repeals and hospital entry.

#### 7) Inference in DiD

Inference in DiD requires clustered standard errors. See
[Guide 01](01_panel_fixed_effects_clustered_se.md) for the full treatment of
the cluster-robust sandwich estimator, the few-clusters problem, and choosing
the cluster level.

The short version for DiD:
- Cluster at (or above) the level of treatment assignment. If treatment is
  assigned at the state level, cluster at the state level.
- Report the number of clusters. If $G < 30$, be cautious; consider
  wild cluster bootstrap for more reliable inference.
- Bertrand, Duflo, and Mullainathan (2004) showed that ignoring serial
  correlation in DiD panels can produce severely misleading standard errors.

<a id="event-studies"></a>
### Event Studies (Leads/Lags): Dynamics and Pre-Trends in One Picture

Event studies generalize DiD by estimating how effects evolve before and after
treatment adoption. They are the primary tool for both assessing the parallel
trends assumption and characterizing the dynamics of the treatment effect.

#### 1) Why Event Studies Are Better Than Pooled DiD

A single DiD coefficient answers "What is the average post-treatment effect?"
An event study answers three richer questions:

**a) Dynamics.** Does the effect appear immediately, ramp up, fade out, or persist? Medicaid expansion might reduce uninsurance immediately but improve health outcomes only after years of chronic disease management.

**b) Pre-trends diagnostic.** If lead coefficients are systematically nonzero, parallel trends is suspect. This is the most important visual diagnostic in applied DiD.

**c) Anticipation effects.** Units may change behavior before the formal treatment date (hospitals adjust staffing, patients delay procedures). Event studies reveal these in near-treatment leads.

#### 2) Building Event-Time Dummies: Step by Step

**Step 1: Define adoption year.** Each treated unit $i$ has an adoption year $T_i$. For never-treated units, $T_i$ is undefined.

**Step 2: Compute event time.** For each treated unit-period, compute $k = t - T_i$. Negative $k$ = pre-treatment (leads); $k = 0$ = treatment onset; positive $k$ = post-treatment (lags).

**Step 3: Create dummies.** For a chosen window $k \in \{-K, \dots, -2, 0, 1, \dots, L\}$, create indicators $D_{it}^{(k)} = \mathbf{1}[t - T_i = k]$. Each dummy equals 1 only for treated units at exactly that event time.

**Step 4: Choose the reference period.** Omit one period to avoid collinearity. The standard choice is $k = -1$ (one period before treatment), the last "clean" pre-treatment period. All coefficients are then interpreted *relative to the period just before treatment began*.

**Step 5: Handle never-treated units.** These units have no $T_i$, so all event-time dummies are zero. They serve as pure controls, helping pin down the common time trend via the time fixed effects.

**Step 6: Bin endpoints.** Observations far from treatment ($k \le -K$ or $k \ge L$) are often grouped into single endpoint bins to avoid estimating coefficients from very few observations.

#### 3) The Event Study Regression

The standard event-study regression is:

$$
Y_{it} = \alpha_i + \gamma_t + \sum_{k \neq -1} \beta_k \cdot \mathbf{1}[t - T_i = k] + \varepsilon_{it}.
$$

**What each term means:**
- $\alpha_i$: unit fixed effects (absorb permanent unit differences).
- $\gamma_t$: time fixed effects (absorb common time shocks).
- $\beta_k$: the effect at event time $k$, relative to the reference period $k = -1$.
- The omitted $k = -1$ is normalized to zero; all $\beta_k$ are relative to it.

**What $\beta_k$ represents:**
- For $k < 0$ (leads): $\beta_k$ measures whether there was a differential change between treated and control units at $k$ periods before treatment, relative to $k = -1$. Under parallel trends + no anticipation, these should all be zero.
- For $k \ge 0$ (lags): $\beta_k$ measures the treatment effect at $k$ periods after adoption.

#### 4) Reading an Event Study Plot: Detailed Walkthrough

An event study plot is the single most important figure in a DiD paper.

**Anatomy of the plot:**
- *X-axis:* event time $k$ (periods relative to treatment). A vertical dashed line marks treatment onset.
- *Y-axis:* estimated coefficients $\hat{\beta}_k$, in outcome units (e.g., percentage points). A horizontal line at zero means "no effect relative to baseline."
- *Dots:* point estimates at each event time.
- *Error bars / bands:* 95% confidence intervals. If the CI includes zero, the estimate is not statistically distinguishable from no effect.

**Reading pre-trends (leads, $k < -1$):**
- Under parallel trends + no anticipation, lead coefficients should hover near zero.
- A "good" pattern: dots bounce randomly around zero with no systematic drift.
- A "bad" pattern: dots trend toward the eventual post-treatment direction -- this suggests treated units were already diverging before treatment.

**Reading the treatment effect path (lags, $k \ge 0$):**
- *Immediate effect:* sharp jump at $k = 0$ that persists (e.g., price cap).
- *Ramp-up:* effect starts small and grows (e.g., insurance coverage improving chronic disease management over years).
- *Fade-out:* effect appears then shrinks toward zero (adaptation, non-compliance, or temporary intervention).
- *Permanent shift:* effect stabilizes at a new level.

**Good vs bad plots:**
- Good: flat pre-trends, clear break at treatment, consistently nonzero post-treatment coefficients, reasonably tight CIs.
- Bad: pre-trends drifting in the same direction as the "effect," huge CIs everywhere, or erratic pattern with no clear break at treatment.

#### 5) Identification for Event Studies

Event studies inherit the DiD assumptions with some refinements:

- **Parallel trends in the pre-period:** Lead coefficients should be jointly
  near zero. This is necessary (but not sufficient) evidence for the assumption.
- **No anticipation:** Treatment should not affect outcomes before the formal
  adoption date. If anticipation is plausible (e.g., a policy was announced a
  year before taking effect), you may need to shift the reference period earlier,
  such as using $k = -2$ as the base instead of $k = -1$.
- **No spillovers:** Other units are not affected by one unit's adoption.
- **Stable treatment definition:** The "treatment" means the same thing across
  cohorts and over time. If early adopters face a different version of the
  policy than late adopters, pooling their event-study coefficients is misleading.

#### 6) Inference in Event Studies

Event studies estimate many $\beta_k$'s. Individual coefficients have wider CIs than the pooled DiD estimate, and with many coefficients some will look "significant" by chance (multiple testing). Do not cherry-pick individual leads -- look at the overall pattern and consider a joint F-test (see Diagnostics below). The shape of the event study matters more than any single dot.

<a id="staggered"></a>
### Staggered Adoption and Modern DiD

#### 1) What Is Staggered Adoption?

In many health economics settings, treatment arrives at different times: Medicaid expansion in 2014 for some states, 2015 for others, 2019 for still others. TWFE handles this mechanically -- $D_{it}$ switches on at different times -- but there is a subtle problem.

#### 2) The TWFE Problem with Heterogeneous Effects

Goodman-Bacon (2021) showed that the TWFE estimator $\hat{\beta}$ in staggered
settings is a weighted average of *all possible 2x2 DiD comparisons* in the
data. This includes three types:

- **Early treated vs never treated** (clean comparison).
- **Late treated vs never treated** (clean comparison).
- **Late treated vs early treated** (problematic: uses already-treated units as "controls").

The third type is the problem. When you use early adopters as "controls" for
late adopters, the early adopters are already treated. If their treatment
effect is changing over time (growing, shrinking), that time-varying effect
contaminates the "control" trend.

#### 3) The Negative Weighting Problem in Plain English

Imagine State A expanded Medicaid in 2014 and State B expanded in 2019. TWFE partly estimates State B's effect by comparing State B (before vs after 2019) against State A (2014-2019 vs 2019+). But State A has been treated since 2014. If State A's treatment effect was *growing* over that period, State A's outcomes were rising for treatment-related reasons. TWFE interprets this rise as "the control trend," making State B's effect look smaller -- or even negative.

In extreme cases, some 2x2 comparisons receive *negative weights*, entering the overall average with the wrong sign. The pooled $\hat{\beta}$ can be negative even if every unit has a positive treatment effect.

#### 4) Modern Solutions (Know They Exist)

Several estimators address these issues. You do not need to implement them here, but should know they exist:

- **Callaway and Sant'Anna (2021):** Group-time-specific ATTs, aggregated; avoids using already-treated units as controls.
- **Sun and Abraham (2021):** Interaction-weighted estimator robust to heterogeneous effects.
- **de Chaisemartin and D'Haultfoeuille (2020):** Estimators that avoid negative weights.
- **Borusyak, Jaravel, and Spiess (2024):** Imputation estimator that builds counterfactuals from untreated observations.

#### 5) When Classic TWFE Is Still Fine

Classic TWFE works well when: (a) treatment timing is uniform (the 2x2 case), (b) effects are homogeneous across cohorts and over time, or (c) a large never-treated group dominates the comparisons. This repo's semi-synthetic exercises use simple adoption structures where TWFE is appropriate. For real research with staggered adoption, use (or at least check) modern estimators.

<a id="diagnostics"></a>
### Diagnostics and Robustness

#### 1) Pre-Trends Tests

**Visual test.** Plot the event study and examine lead coefficients. This is
the most common and most informative diagnostic. A clear upward or downward
drift in pre-treatment coefficients is a red flag.

**Joint F-test on leads.** Test the null hypothesis $H_0: \beta_{-K} = \beta_{-K+1} = \dots = \beta_{-2} = 0$ (all lead coefficients are jointly zero).
A rejection suggests differential pre-trends. But note: failure to reject does
not prove parallel trends -- it could reflect low power (too few pre-periods
or noisy data).

```python
# After estimating the event study, test joint significance of leads:
from scipy import stats

# Suppose lead_coefs is a vector of pre-treatment beta_k estimates
# and lead_vcov is their variance-covariance matrix
# F = lead_coefs' @ inv(lead_vcov) @ lead_coefs / num_leads
# Compare to F(num_leads, df_residual) distribution
```

#### 2) Placebo Treatments

**Fake treatment dates.** Shift the treatment date earlier (e.g., pretend
treatment happened in 2011 instead of 2014) and re-estimate. If you "find"
a significant effect at the fake date, your design is suspect -- the treated
group was already trending differently.

**Fake outcomes.** Use an outcome that should not be affected by the treatment.
If studying Medicaid expansion's effect on uninsurance, try an outcome like
traffic fatalities. If DiD finds a "significant effect" on traffic fatalities,
something is wrong with your comparison group.

**Fake treatment groups.** Assign treatment randomly among control units. The
DiD estimate should be near zero. This is a permutation-style placebo that
tests whether your method is prone to false positives.

#### 3) Sensitivity Analysis: What if Parallel Trends Is Slightly Violated?

Rambachan and Roth (2023) formalize a sensitivity analysis: "What if parallel
trends is violated, but only by a small amount?" They construct bounds on the
treatment effect under the assumption that violations are bounded (e.g., the
differential trend is at most $\bar{M}$ per period). This is conceptually
similar to Oster's (2019) approach for omitted variable bias, but tailored to
DiD.

At minimum, you should informally assess: "If the small pre-trend I see
continued into the post-period, how much would it change my estimate?"

#### 4) Window Sensitivity

Re-estimate the event study with different pre/post windows:
- Shorten the pre-period: does the estimate change?
- Shorten the post-period: does the effect persist or fade?
- Extend the window: does the estimate remain stable, or do distant periods
  introduce noise?

If conclusions are highly sensitive to the window choice, the result is fragile.

<a id="code-templates"></a>
### Practical Code Templates

#### Template 1: DiD Estimation with linearmodels

```python
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# --- Data setup ---
# Assume df has columns: unit_id, year, outcome, treated (0/1), post (0/1)
# Treatment indicator: D = treated * post
df['D'] = df['treated'] * df['post']

# Set panel index
df = df.set_index(['unit_id', 'year'])

# --- TWFE DiD ---
mod = PanelOLS.from_formula(
    'outcome ~ D + EntityEffects + TimeEffects',
    data=df,
    check_rank=False,
)
res = mod.fit(cov_type='clustered', cluster_entity=True)
print(res.summary)

# The coefficient on D is the DiD estimate.
# cluster_entity=True clusters SE at the unit (entity) level.
# If treatment is at a higher level (e.g., state), create a state
# variable and use clusters=df['state'] instead.
```

#### Template 2: Event Study Estimation and Plot

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

# Assume df has: unit_id, year, outcome, adopt_year (NaN if never treated)
df['event_time'] = df['year'] - df['adopt_year']
K_pre, K_post = 5, 5
df['event_time_binned'] = df['event_time'].clip(lower=-K_pre, upper=K_post)

# Create dummies (drop k=-1 as reference; never-treated get all zeros)
for k in range(-K_pre, K_post + 1):
    if k == -1:
        continue
    col = f'k_{k}' if k < 0 else f'k_plus_{k}'
    df[col] = ((df['event_time_binned'] == k) & df['adopt_year'].notna()).astype(int)

dummy_cols = [c for c in df.columns if c.startswith('k_')]
formula = 'outcome ~ ' + ' + '.join(dummy_cols) + ' + EntityEffects + TimeEffects'
df_panel = df.set_index(['unit_id', 'year'])
res = PanelOLS.from_formula(formula, data=df_panel, check_rank=False)\
      .fit(cov_type='clustered', cluster_entity=True)

# Extract coefficients + CIs
coefs = []
for k in range(-K_pre, K_post + 1):
    if k == -1:
        coefs.append({'k': k, 'beta': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0})
        continue
    col = f'k_{k}' if k < 0 else f'k_plus_{k}'
    b, se = res.params[col], res.std_errors[col]
    coefs.append({'k': k, 'beta': b, 'ci_lo': b - 1.96*se, 'ci_hi': b + 1.96*se})
coef_df = pd.DataFrame(coefs)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(coef_df['k'], coef_df['beta'],
            yerr=[coef_df['beta']-coef_df['ci_lo'], coef_df['ci_hi']-coef_df['beta']],
            fmt='o', capsize=3, color='steelblue')
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.axvline(-0.5, color='red', lw=0.8, ls=':', label='Treatment onset')
ax.set_xlabel('Event time (k)'); ax.set_ylabel('Coefficient (rel. to k=-1)')
ax.set_title('Event Study Plot'); ax.legend(); plt.tight_layout(); plt.show()
```

#### Template 3: Pre-Trends Joint Test

```python
import numpy as np
from scipy import stats

# After estimation, extract lead coefficients and their covariance
lead_names = [f'k_{k}' for k in range(-K_pre, -1)]
lead_betas = np.array([res.params[n] for n in lead_names])
lead_vcov = res.cov.loc[lead_names, lead_names].values

# Wald test: beta' @ inv(V) @ beta ~ chi-squared(len(leads))
wald_stat = lead_betas @ np.linalg.solve(lead_vcov, lead_betas)
p_value = 1 - stats.chi2.cdf(wald_stat, df=len(lead_betas))
print(f"Joint pre-trends test: Wald = {wald_stat:.2f}, p = {p_value:.4f}")
# If p < 0.05, pre-trends are jointly significant -- a warning sign.
```

### Project Code Map
- `src/causal.py`: panel + IV helpers (`to_panel_index`, `fit_twfe_panel_ols`, `fit_iv_2sls`).
- `scripts/build_datasets.py`: ACS panel builder. `src/census_api.py`: Census/ACS client.
- `configs/census_panel.yaml`: panel config. `data/sample/census_county_panel_sample.csv`: offline dataset.
- `src/data.py`: caching helpers. `src/features.py`: feature engineering. `src/evaluation.py`: splits + metrics.

<a id="common-mistakes"></a>
### Common Mistakes in DiD and Event Studies

1. **Not checking pre-trends before interpreting effects.**
   The event study plot is not optional decoration. If you skip the pre-trends
   check and report a pooled DiD coefficient, reviewers will (rightly) question
   whether your parallel trends assumption is credible.

2. **Controlling for post-treatment variables.**
   Adding insurance enrollment as a control when studying Medicaid expansion's
   effect on health is controlling for a *mediator*. This biases the estimate
   and changes the estimand from the total effect to something difficult to
   interpret. Only control for variables that are not affected by treatment.

3. **Using the wrong comparison group.**
   The control group must be a plausible counterfactual. Comparing states that
   expanded Medicaid to states that *could not* expand due to fundamentally
   different political/economic structures may violate parallel trends. Look for
   control units that are similar on observables and pre-treatment trends.

4. **Ignoring staggered adoption issues.**
   If treatment rolls out at different times and you use classic TWFE without
   thinking about heterogeneous effects, your estimate may be misleading or even
   wrong-signed. At minimum, check whether your results are sensitive to
   dropping early adopters or late adopters.

5. **Not clustering at the right level.**
   If treatment is assigned at the state level, you must cluster standard errors
   at the state level (or higher). Clustering at a finer level (county, individual)
   produces standard errors that are too small, leading to false rejections.
   See [Guide 01](01_panel_fixed_effects_clustered_se.md) for the full treatment.

6. **Confusing the pre-trends statistical test with the untestable assumption.**
   Failing to reject the null of zero pre-trends does NOT prove parallel trends
   holds. It may simply reflect low statistical power (small samples, noisy data,
   few pre-periods). Conversely, small but statistically significant pre-trends
   with a large sample may not be economically meaningful. The test informs but
   does not resolve the identification question.

7. **Cherry-picking the event window.**
   Showing only a narrow event window that looks clean while hiding a wider
   window with problematic pre-trends is p-hacking. Report a reasonable window
   and show sensitivity to alternatives.

8. **Interpreting every event-study coefficient literally.**
   Individual $\hat{\beta}_k$ estimates can be noisy. The *pattern* matters more
   than any single coefficient. Do not write "the effect was 2.3 pp at $k = 3$
   but 1.8 pp at $k = 4$, showing a decline" if the difference is well within
   the confidence intervals.

#### Exercises

- [ ] Draw a 2x2 table with your own hypothetical numbers and compute the DiD estimate by hand.
- [ ] Write the parallel trends assumption in plain English for a Medicaid expansion study.
- [ ] List two scenarios where pre-trends could look fine but parallel trends still fails.
- [ ] Build event-time dummies from a panel dataset and confirm $k = 0$ corresponds to the adoption year.
- [ ] Plot an event study and write four sentences interpreting (a) the leads and (b) the lags.
- [ ] Run a placebo test using a fake treatment date and explain what you expected vs what you found.
- [ ] Explain in three sentences why using already-treated units as controls in staggered DiD is problematic.
- [ ] Look up one health economics DiD paper and identify: the treatment, the control group, the parallel trends argument, and the event study diagnostic.

<a id="summary"></a>
## Summary + Suggested Readings

This guide covered the two most important tools in the applied health economics
causal toolkit: difference-in-differences and event studies. The key ideas:

- DiD removes time-invariant confounders and common time shocks by comparing
  changes across groups.
- The identifying assumption (parallel trends) is powerful but fundamentally
  untestable. Pre-trends evidence is necessary but not sufficient.
- Event studies extend DiD by showing how effects evolve dynamically and by
  providing the primary pre-trends diagnostic.
- Staggered adoption creates subtle problems for classic TWFE; modern estimators
  address these but classic TWFE is fine in simpler settings.
- Inference requires clustered standard errors at (or above) the treatment level.

### Suggested Readings

**Textbooks:**
- Angrist and Pischke, *Mostly Harmless Econometrics* (2009), Ch. 5: DiD foundations.
- Cunningham, *Causal Inference: The Mixtape* (2021), Ch. 9: DiD and event studies with code.
- Huntington-Klein, *The Effect* (2021), Ch. 18: DiD with intuitive explanations.

**Key methodological papers:**
- Bertrand, Duflo, and Mullainathan (2004), "How Much Should We Trust Differences-in-Differences Estimates?" -- the foundational paper on serial correlation and inference in DiD.
- Goodman-Bacon (2021), "Difference-in-Differences with Variation in Treatment Timing" -- the decomposition theorem for staggered TWFE.
- Callaway and Sant'Anna (2021), "Difference-in-Differences with Multiple Time Periods" -- modern estimator for staggered adoption.
- Sun and Abraham (2021), "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects" -- interaction-weighted estimator.
- Rambachan and Roth (2023), "A More Credible Approach to Parallel Trends" -- sensitivity analysis for violations.
- Roth (2022), "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends" -- why pre-testing can distort inference.

**Health economics applications:**
- Finkelstein et al. (2012), "The Oregon Health Insurance Experiment: Evidence from the First Year" -- QJE.
- Sommers, Long, and Baicker (2012), "Changes in Utilization and Health Status after Massachusetts Health Reform."
- Antwi, Moriya, and Simon (2013), "Effects of Federal Policy to Insure Young Adults" -- ACA age-26 provision.
- Dafny (2009), "Estimation and Identification of Merger Effects" -- hospital mergers and prices.
