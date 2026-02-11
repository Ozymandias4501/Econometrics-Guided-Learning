# Cheatsheet: Causal Inference Methods

## The Fundamental Problem

Regression can tell you that $X$ and $Y$ move together. It cannot tell you that changing $X$ would change $Y$. Three threats prevent causal interpretation of a regression coefficient:

- **Omitted variable bias (OVB)**: A third variable drives both $X$ and $Y$. Example: counties with more hospitals have higher mortality — but sicker populations attract more hospitals. Sickness is the omitted confounder.
- **Reverse causality**: $Y$ causes $X$, not the other way around. Example: does health spending improve outcomes, or do worse outcomes drive higher spending?
- **Selection bias**: Treated and untreated groups differ in ways that also affect the outcome. Example: patients who choose surgery are systematically different from those who don't (age, severity, preferences).

**The causal question**: What would happen to $Y$ if we *intervened* to change $X$, holding everything else fixed? Every method below is a strategy for answering this question credibly.

## Method Comparison

| Method | Core strategy | When it identifies a causal effect | What it requires | Typical health econ application |
|---|---|---|---|---|
| **OLS + controls** | Include observable confounders as covariates so the coefficient on $X$ is "adjusted" | All confounders are observed and correctly specified | Selection on observables — a strong, untestable assumption | Estimating the association between insurance status and utilization, controlling for age, income, and health status |
| **Panel Fixed Effects** | Include entity-specific intercepts that absorb all time-invariant unobserved differences | The only confounders are characteristics that don't change over time (geography, genetics, institutional culture) | Panel data (repeated observations of the same entities over time) | Effect of Medicaid expansion on county-level ER visits, with county fixed effects absorbing baseline differences |
| **Difference-in-Differences (DiD)** | Compare the change in outcomes for a treated group vs. a control group, before vs. after treatment | The treated group would have followed the same trend as the control group absent treatment (parallel trends) | A treatment that affects some units but not others at a specific time, plus a credible control group | Effect of ACA Medicaid expansion (2014) on uninsured rates: expansion states vs. non-expansion states, before vs. after 2014 |
| **Instrumental Variables (2SLS)** | Find a variable $Z$ that shifts $X$ but has no direct effect on $Y$ except through $X$ | The instrument is relevant (strongly predicts $X$) and valid (exclusion restriction: $Z \to X \to Y$ only) | A valid instrument — intellectually the hardest requirement | Effect of hospital volume on surgical mortality, using distance to hospital as an instrument (patients near high-volume hospitals are more likely to go there, but distance doesn't directly affect surgical skill) |
| **Regression Discontinuity (RDD)** | Compare units just above vs. just below a threshold that determines treatment | Assignment is based on a continuous "running variable" with a cutoff, and units near the cutoff are comparable | A sharp or fuzzy threshold rule; enough observations near the cutoff | Effect of Medicare eligibility on healthcare utilization: compare 64-year-olds (ineligible) to 65-year-olds (eligible) |

## Panel Fixed Effects — In Depth

**Model**:
$$
y_{it} = \alpha_i + \gamma_t + \beta x_{it} + \varepsilon_{it}
$$

| Term | What it does | What it absorbs |
|---|---|---|
| $\alpha_i$ (entity fixed effects) | A separate intercept for each entity (county, hospital, patient) | All characteristics of that entity that don't change over time: geography, institutional history, demographics that evolve slowly |
| $\gamma_t$ (time fixed effects) | A separate intercept for each time period | All shocks common to every entity in that period: national recessions, federal policy changes, seasonal patterns |
| $\beta$ | The coefficient of interest | Identified from *within-entity, within-period* variation — changes in $x$ for the same entity over time, after removing economy-wide trends |

**What FE can and cannot do**:
- Can remove: permanent differences between entities (e.g., some counties are always sicker)
- Cannot remove: time-varying confounders that differ across entities (e.g., a state-level policy change that affects only some counties AND correlates with $x$)

**Key assumption**: Strict exogeneity — $E[\varepsilon_{it} | x_{i1}, \dots, x_{iT}, \alpha_i, \gamma_t] = 0$. Past, present, and future values of $x$ are uncorrelated with the error. This rules out feedback (e.g., bad outcomes this period causing a policy change next period).

**Python**: `PanelOLS(y, x, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)`

**Why cluster SE?** Observations within the same entity are correlated (a county that has high spending this year will likely have high spending next year). Ignoring this correlation makes standard errors too small, inflating t-stats and producing false significance.

## Difference-in-Differences (DiD) — In Depth

**The 2x2 setup**:

| | Before treatment | After treatment |
|---|---|---|
| **Treated group** | $\bar Y_{T,pre}$ | $\bar Y_{T,post}$ |
| **Control group** | $\bar Y_{C,pre}$ | $\bar Y_{C,post}$ |

$$
\hat\tau_{DiD} = \underbrace{(\bar Y_{T,post} - \bar Y_{T,pre})}_{\text{change in treated}} - \underbrace{(\bar Y_{C,post} - \bar Y_{C,pre})}_{\text{change in control}}
$$

The first difference removes time-invariant differences between groups. The second difference removes common time trends. What's left is the treatment effect — *if* the parallel trends assumption holds.

**Parallel trends assumption**: Absent the treatment, the treated group would have evolved along the same trajectory as the control group. This is the identifying assumption. It's untestable for the post-treatment period but you can assess its plausibility:

- **Event study plot**: Estimate separate coefficients for each time period relative to treatment. Pre-treatment coefficients should be near zero (no divergence before the policy). Post-treatment coefficients show the dynamic treatment effect.
- If pre-treatment trends diverge, DiD is not credible.

**Regression implementation**:
$$
y_{it} = \alpha + \beta_1 \cdot \text{Treated}_i + \beta_2 \cdot \text{Post}_t + \tau \cdot (\text{Treated}_i \times \text{Post}_t) + \varepsilon_{it}
$$

$\tau$ is the DiD estimate of the treatment effect.

## Instrumental Variables (2SLS) — In Depth

IV addresses endogeneity directly: when $X$ is correlated with the error term (due to OVB, reverse causality, or measurement error), OLS is biased. IV uses an external source of variation in $X$ that is unrelated to the confounders.

**Two stages**:

1. **First stage** — predict $X$ using the instrument $Z$:
$$
x_i = \pi_0 + \pi_1 z_i + v_i
$$

2. **Second stage** — regress $Y$ on the predicted values $\hat x_i$ (which contain only the variation in $X$ driven by $Z$):
$$
y_i = \beta_0 + \beta_1 \hat x_i + \varepsilon_i
$$

**Two requirements for a valid instrument**:

| Requirement | What it means | How to assess |
|---|---|---|
| **Relevance** | $Z$ must actually predict $X$ ($\pi_1 \neq 0$) | First-stage F-statistic > 10 (Staiger-Stock rule). Below 10 = weak instrument, and 2SLS becomes unreliable (biased toward OLS, wide CIs, over-rejection) |
| **Exclusion restriction** | $Z$ affects $Y$ only through $X$, not directly | Cannot be tested statistically — requires a substantive argument. This is the hardest part of IV. With multiple instruments, the Hansen J test checks over-identifying restrictions, but it has limited power |

**What IV estimates**: With a heterogeneous treatment effect, IV estimates the Local Average Treatment Effect (LATE) — the causal effect for "compliers" (units whose $X$ changes because of $Z$). This may differ from the Average Treatment Effect (ATE) for the whole population.

**Python**: `IV2SLS(dependent, exog, endog, instruments).fit()`

## OVB Formula

When the true model is $y = \beta_1 x + \beta_2 z + \varepsilon$ but you omit $z$ and run the short regression $y = b_1 x + e$:

$$
E[\hat b_1] = \beta_1 + \underbrace{\beta_2 \cdot \frac{Cov(x, z)}{Var(x)}}_{\text{bias}}
$$

The bias is the product of (1) the effect of the omitted variable on $Y$ and (2) the relationship between the omitted variable and the included regressor.

**Direction of bias**:

| $\beta_2$ (effect of omitted on $Y$) | $Cov(x, z)$ (relationship of omitted with $X$) | Bias direction | Example |
|---|---|---|---|
| Positive | Positive | Upward (overestimate $\beta_1$) | Returns to education, omitting ability: ability raises wages ($\beta_2 > 0$) and correlates with education ($Cov > 0$), so OLS overestimates the return to schooling |
| Positive | Negative | Downward (underestimate $\beta_1$) | Effect of hospital spending on outcomes, omitting severity: sicker patients cost more ($Cov < 0$ between spending and health) and have worse outcomes ($\beta_2 > 0$), biasing the spending coefficient downward |
| Negative | Positive | Downward | — |
| Negative | Negative | Upward | — |

**Worked example**: True model: $\text{wage} = 0.08 \cdot \text{education} + 0.05 \cdot \text{ability} + \varepsilon$. Suppose $\frac{Cov(\text{education}, \text{ability})}{Var(\text{education})} = 0.6$. Then $E[\hat b_{\text{education}}] = 0.08 + 0.05 \times 0.6 = 0.11$. The short regression overestimates the return to education by 38%.

## Diagnostics Checklist

| Method | Key diagnostic | What to check | Red flag |
|---|---|---|---|
| OLS + controls | Coefficient sensitivity | Does $\hat\beta$ change substantially when you add or remove controls? Large changes suggest OVB | Coefficient flips sign or magnitude changes by >50% when adding a plausible confounder |
| Panel FE | Hausman test | FE vs RE — are entity effects correlated with regressors? If yes, RE is inconsistent | Hausman test rejects: use FE. Hausman test fails to reject: RE is more efficient, but FE is still consistent |
| DiD | Event study plot | Are pre-treatment coefficients near zero and flat? | Pre-treatment coefficients trending or significantly different from zero — parallel trends assumption is violated |
| IV | First-stage F-statistic | F > 10 (Staiger-Stock rule of thumb) | F < 10 means weak instrument: 2SLS is biased toward OLS, confidence intervals are unreliable, and tests over-reject |
| IV | Over-identification test | If you have more instruments than endogenous variables: Hansen J test | J test rejects: at least one instrument is invalid (fails exclusion restriction) |
| All methods | Placebo tests | Run the same analysis on outcomes or time periods where you expect no effect | Finding a "treatment effect" where there shouldn't be one undermines credibility |

## Common Mistakes

| Mistake | What goes wrong | What to do instead |
|---|---|---|
| "I controlled for everything, so it's causal" | You can only control for *observed* confounders. Unobserved confounders still bias OLS | Acknowledge limitations; use FE, IV, or DiD if you need a causal claim |
| Using FE when confounders are time-varying | FE only absorbs differences that are constant over time. A time-varying confounder (e.g., a state policy that changes over your sample period) is not removed by entity FE | Add time-varying controls, use DiD with appropriate controls, or find an instrument |
| Not clustering standard errors in panel/DiD | Within-entity observations are correlated over time. Ignoring this correlation makes SE too small, producing falsely significant results | Always cluster at the entity level in panel regressions |
| Weak instruments (first-stage F < 10) | 2SLS is biased toward OLS; confidence intervals have incorrect coverage; tests over-reject | Find a stronger instrument, combine instruments, or use weak-instrument-robust inference (Anderson-Rubin test) |
| Choosing the instrument after seeing the data | If you try many instruments and report the one that "works," you're data mining | Pre-specify the instrument based on institutional knowledge before running regressions |
| Ignoring parallel trends violations in DiD | If treated and control groups were already diverging before treatment, the DiD estimate captures pre-existing trends, not the treatment effect | Always plot pre-treatment trends; if they diverge, DiD is not credible for this comparison |
| Conflating LATE with ATE in IV | IV estimates the effect for compliers (units affected by the instrument), not the whole population | Be explicit about what population IV identifies: "the effect for patients who chose Hospital A because it was closer" |
