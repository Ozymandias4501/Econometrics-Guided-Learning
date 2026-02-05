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
