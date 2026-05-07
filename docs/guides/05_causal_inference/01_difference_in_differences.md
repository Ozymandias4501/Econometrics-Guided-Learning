# Guide: 01_difference_in_differences

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/05_causal_inference/01_difference_in_differences.ipynb`.

DiD is the dominant research design for causal inference in applied economics: minimum wage, Medicaid expansion, charter schools, environmental regulation. If you read more than a handful of empirical-econ papers, you have already seen DiD even if it was not labeled. This guide focuses on the mechanics, the parallel-trends assumption, and the small-cluster pitfall.

### Key Terms (defined)
- **Two-by-two DiD**: simplest case — two groups (treated, control), two periods (pre, post), one treatment date.
- **Parallel trends**: the load-bearing identifying assumption — in the absence of treatment, treated and control groups would have moved in parallel.
- **Two-way fixed effects (TWFE)**: the regression form $y_{i,t} = \alpha_i + \delta_t + \beta \cdot \text{treat}_{i,t} + u_{i,t}$ with unit and time fixed effects. Generalizes the 2x2 DiD.
- **Event study**: a DiD-like design that estimates separate effects at each lead and lag of treatment, producing a coefficient path over time.
- **Clustered standard errors**: variance estimator that accounts for within-cluster (e.g., within-state, within-firm) correlation. Required for almost all panel inference.
- **Wild cluster bootstrap**: small-cluster correction. Necessary when the number of clusters is below ~30.
- **Anticipation**: behavior change *before* treatment because units expect it. Violates the standard DiD timing.
- **Spillover (SUTVA violation)**: control units affected by treatment of others. Invalidates the control as a counterfactual.

### How To Read This Guide
- Use **Step-by-Step** for the notebook checklist and a placebo-test pattern.
- Use **Technical Explanations** for the algebra, the parallel-trends assumption, the modern critique of TWFE under heterogeneous effects, and clustering details.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- [ ] Write down the four cell means $\bar Y_{T,0}, \bar Y_{T,1}, \bar Y_{C,0}, \bar Y_{C,1}$ for your data.
- [ ] Compute DiD by hand from those four means.
- [ ] Run an OLS with the interaction $\text{treated} \times \text{post}$ and confirm the coefficient matches.
- [ ] Cluster the standard errors at the level of the treatment (state, county, firm — wherever treatment is assigned).
- [ ] Plot pre-treatment trends for treated vs control. They should look parallel.
- [ ] Apply the workflow to a panel from your work or the bundled Census panel with a synthesized policy.
- [ ] Write down at least one specific way the parallel-trends assumption could fail in your application.

### Alternative Example: Placebo Test

A placebo test pretends the treatment happened at a date where no treatment actually occurred. If your DiD coefficient is large at the placebo date, the parallel-trends assumption is suspect.

```python
# Pretend the policy started in 2016 instead of 2018
cdf['post_placebo'] = (cdf['year'] >= 2016) & (cdf['year'] < 2018)
cdf['treated_x_post_placebo'] = cdf['treated'] * cdf['post_placebo'].astype(int)

# Use only pre-2018 observations (so the real policy hasn't kicked in)
pre_only = cdf.loc[cdf['year'] < 2018]
X = sm.add_constant(pre_only[['treated', 'post_placebo', 'treated_x_post_placebo']], has_constant='add')
y = pre_only['unemployment_rate'] * 100.0
res_placebo = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': pre_only['state']})
print(res_placebo.summary().tables[1])
```

A coefficient near zero with a CI that includes zero is evidence the pre-trends are well-behaved. A large or significant placebo coefficient is a warning that the real DiD coefficient may be picking up a pre-existing differential trend rather than the policy.

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### The 2x2 DiD identity

Let $G \in \{T, C\}$ index treatment status and $P \in \{0, 1\}$ index pre/post. Let $\bar Y_{G, P}$ be the mean outcome in cell $(G, P)$.

Two equivalent forms:

$$
\hat\beta_{DiD} = (\bar Y_{T,1} - \bar Y_{T,0}) - (\bar Y_{C,1} - \bar Y_{C,0})
$$

$$
\hat\beta_{DiD} = (\bar Y_{T,1} - \bar Y_{C,1}) - (\bar Y_{T,0} - \bar Y_{C,0})
$$

The first form: 'change in treated minus change in control.' The second form: 'gap after minus gap before.' They are algebraically identical; pick whichever explains better.

The OLS regression:

$$
y_{u,t} = \alpha + \beta_T \text{treated}_u + \beta_P \text{post}_t + \beta_{DiD} (\text{treated}_u \times \text{post}_t) + u_{u,t}
$$

evaluates to those same four cell means at the four combinations of $(\text{treated}, \text{post})$. Solving for the coefficients reproduces the DiD identity.

### The parallel-trends assumption, formally

DiD identifies the average treatment effect on the treated (ATT) under the assumption:

$$
\mathbb{E}[Y(0)_{T,1} - Y(0)_{T,0}] = \mathbb{E}[Y(0)_{C,1} - Y(0)_{C,0}].
$$

In words: in the absence of treatment, the treated group would have followed the same time path as the control group. You cannot observe $Y(0)_{T,1}$ — what would have happened to treated units if they had not been treated — so this is a counterfactual claim, not a directly testable one.

The standard partial test: examine *pre-treatment* trends. If they look parallel before treatment, the assumption that they would have stayed parallel without treatment is at least not contradicted. If they were diverging before treatment, parallel trends is implausible — your DiD will pick up the pre-existing divergence.

### The TWFE generalization and its modern critique

The two-way fixed effects (TWFE) regression generalizes 2x2 DiD to multiple periods and staggered treatment timing:

$$
y_{i,t} = \alpha_i + \delta_t + \beta \text{treat}_{i,t} + u_{i,t}
$$

where $\alpha_i$ is a unit fixed effect, $\delta_t$ is a time fixed effect, and $\text{treat}_{i,t}$ is 1 if unit $i$ is treated at time $t$.

For 2x2 DiD with homogeneous effects, $\hat\beta_{TWFE}$ is the ATT. **For staggered or heterogeneous treatment, it is not.**

Goodman-Bacon (2021), Sun and Abraham (2021), Callaway and Sant'Anna (2021) showed that with staggered timing and effects that vary by cohort, TWFE puts negative weights on some 2x2 comparisons, can produce signs opposite to every individual treatment effect, and generally has no clean causal interpretation.

What to do in practice:
- For 2x2 DiD with one treatment date: TWFE is fine.
- For staggered timing or suspected effect heterogeneity: use Callaway-Sant'Anna or Sun-Abraham estimators (available in the `differences` Python package or `did` in R).

The notebook stays with 2x2 because the underlying mechanics are clearer, and we flag the staggered-timing issue without solving it.

### Clustering: why and how

Panel data has within-unit dependence: today's $u_{u,t}$ is correlated with yesterday's $u_{u,t-1}$ if the unit has any persistent shocks (and they almost always do).

Naive OLS SE assumes all $n \cdot T$ observations are independent. For coefficients identified from **between-unit** variation (constant, group dummies), naive SE is far too small — it pretends you have $n \cdot T$ independent observations when really you have $n$ units. Clustered SE corrects this and is typically much larger.

For coefficients identified from **within-unit** variation (post dummy, treated×post interaction), the direction depends on the within-unit shock structure. Positive within-unit autocorrelation makes within-unit *differences* less noisy than naive iid would predict, so clustered SE on the DiD interaction can actually shrink. The simulation in the notebook is calibrated to show this surprising fact.

The takeaway: **always cluster at the level of the treatment**. Whether the SE goes up or down at any particular coefficient is a fact about your data, not a rule to memorize.

### Small-cluster issues

Standard clustered SE is asymptotically valid as the number of clusters $G \to \infty$. Practical guidance:

- $G \geq 50$: standard clustered SE is fine.
- $G$ between 20 and 50: borderline; standard inference becomes liberal.
- $G < 20$: do not trust the p-values from standard clustered SE. Use the **wild cluster bootstrap** (Cameron, Gelbach, Miller 2008).

The notebook's Census-panel example has 3 states. The clustered SE shown there are *much* too narrow. The example is a workflow demonstration, not a substantive inference. State-level DiD studies in the literature typically use all 50 states (sometimes plus DC) and still apply small-cluster corrections.

### How DiD can fail in practice

| Threat | What it does | Diagnostic |
|---|---|---|
| Pre-treatment trend differences | Bias is hard-coded into pre-period; DiD attributes it to policy | Plot trends. Run placebo. |
| Anticipation | Outcome moves before treatment date | Plot pre-trends. Look for inflections at announcement, not just enactment. |
| Spillovers | Control absorbs treatment indirectly | Substantive knowledge. Try alternative controls geographically distant from treatment. |
| Staggered timing + heterogeneous effects | TWFE coefficient has wrong sign possibilities | Use Callaway-Sant'Anna or Sun-Abraham. |
| Composition change | Selection into the panel changes with treatment | Check sample sizes by cell. |
| Few clusters | Standard SE under-cover | Wild cluster bootstrap. |

### Project Code Map
- `src/econometrics.py`: `fit_ols` is used for the underlying regression. No new helpers; the notebook keeps the setup explicit.
- The notebook synthesizes a policy on the bundled `census_county_panel_sample.csv` rather than pulling new data.

### Common Mistakes
- Reporting a DiD coefficient without a parallel-trends plot. Mandatory.
- Using OLS SE on panel data. Always cluster.
- Clustering at the wrong level (e.g., individual when treatment varies at state level). Cluster at the level of treatment assignment.
- Dropping treated units from the pre-period because 'they aren't treated yet.' That breaks the panel. Treated units are in both periods; the variable that flips at treatment is `post`, not the unit's existence.
- Trusting clustered p-values with fewer than ~20 clusters.
- Using TWFE for staggered treatment timing without checking for heterogeneous effects.

<a id="summary"></a>
## Summary + Suggested Readings

After this notebook you should be able to:

- compute a 2x2 DiD by hand and via OLS interaction,
- explain the parallel-trends assumption and check it visually,
- cluster standard errors at the right level and read the comparison sensibly,
- recognize when 2x2 DiD is sufficient and when modern staggered-timing estimators are needed,
- write down at least three plausible threats to identification for any DiD setup you build.

**Companion guides:**
- [Guide 00 — DAGs and OVB](00_predictive_vs_causal_dags.md): the conceptual basis. DiD is a particular research design that operationalizes the assumption 'parallel trends' as the identifying restriction.

**Suggested readings:**
- Card and Krueger (1994), "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania" — the seminal applied DiD paper.
- Goodman-Bacon (2021), "Difference-in-Differences with Variation in Treatment Timing" — the negative-weights problem.
- Callaway and Sant'Anna (2021), "Difference-in-Differences with Multiple Time Periods" — the modern staggered-timing estimator.
- Roth, Sant'Anna, Bilinski, Poe (2023), "What's Trending in Difference-in-Differences? A Synthesis of the Recent Econometrics Literature" — a clean overview of the post-Goodman-Bacon DiD landscape.
- Cunningham, *Causal Inference: The Mixtape*, Ch. 9 (DiD) — the most readable applied introduction.
- Cameron and Miller (2015), "A Practitioner's Guide to Cluster-Robust Inference" — clustering done right, including the wild bootstrap.
