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
