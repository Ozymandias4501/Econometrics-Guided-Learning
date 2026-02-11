# Cheatsheet: Causal Inference Methods

## The Fundamental Problem

**Correlation $\neq$ causation** because of:
- **Omitted variable bias (OVB)**: An unobserved confounder drives both $X$ and $Y$
- **Reverse causality**: $Y$ causes $X$, not the other way around
- **Selection bias**: Treated and untreated groups differ in unobserved ways

**Goal**: Estimate the causal effect of $X$ on $Y$ — what happens to $Y$ when you *change* $X$, holding everything else fixed.

## Method Comparison

| Method | Key idea | Identifies causation when... | Requires |
|---|---|---|---|
| **OLS + controls** | Control for observable confounders | All confounders are observed and included | Selection on observables (strong assumption) |
| **Panel Fixed Effects** | Absorb time-invariant unobserved heterogeneity | Confounders are fixed over time (e.g., geography, culture) | Panel data (entity $\times$ time) |
| **Difference-in-Differences** | Compare treated vs. control, before vs. after | Parallel trends: absent treatment, groups would have evolved similarly | Treatment timing variation, credible control group |
| **Instrumental Variables (2SLS)** | Use an exogenous instrument $Z$ that affects $Y$ only through $X$ | Instrument is relevant ($Z$ predicts $X$) and valid (exclusion restriction) | A valid instrument (hardest part) |
| **Regression Discontinuity** | Compare units just above/below a threshold | Assignment is based on a running variable with a cutoff | Sharp or fuzzy threshold rule |

## Panel Fixed Effects

**Model**:
$$
y_{it} = \alpha_i + \gamma_t + \beta x_{it} + \varepsilon_{it}
$$

| Term | What it absorbs |
|---|---|
| $\alpha_i$ (entity FE) | All time-invariant differences between entities |
| $\gamma_t$ (time FE) | All entity-invariant shocks common to each period |
| $\beta$ | Within-entity, within-time variation in $x$ |

**Key assumption**: No time-varying unobserved confounders (strict exogeneity).

**Python**: `PanelOLS(y, x, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)`

## Difference-in-Differences (DiD)

**The 2x2 setup**:

| | Before | After |
|---|---|---|
| **Treated** | $\bar Y_{T,pre}$ | $\bar Y_{T,post}$ |
| **Control** | $\bar Y_{C,pre}$ | $\bar Y_{C,post}$ |

$$
\hat\tau_{DiD} = (\bar Y_{T,post} - \bar Y_{T,pre}) - (\bar Y_{C,post} - \bar Y_{C,pre})
$$

**Parallel trends assumption**: Without treatment, the treated group would have followed the same trajectory as the control group. Test by checking pre-treatment trends are similar (event study plot).

## Instrumental Variables (2SLS)

**Two stages**:

1. **First stage**: Regress endogenous $X$ on instrument $Z$
$$
x_i = \pi_0 + \pi_1 z_i + v_i
$$

2. **Second stage**: Regress $Y$ on predicted $\hat X$
$$
y_i = \beta_0 + \beta_1 \hat x_i + \varepsilon_i
$$

**Two requirements**:
- **Relevance**: $\pi_1 \neq 0$ (first-stage F-stat > 10)
- **Exclusion restriction**: $Z$ affects $Y$ only through $X$ (untestable)

**Python**: `IV2SLS(y, exog, endog, instruments).fit()`

## OVB Formula

When the true model is $y = \beta_1 x + \beta_2 z + \varepsilon$ but you omit $z$:

$$
E[\hat\beta_{short}] = \beta_1 + \beta_2 \cdot \frac{Cov(x, z)}{Var(x)}
$$

**Direction of bias**:

| $\beta_2$ (effect of omitted) | $Cov(x, z)$ | Bias direction |
|---|---|---|
| Positive | Positive | Upward (overestimate) |
| Positive | Negative | Downward (underestimate) |
| Negative | Positive | Downward |
| Negative | Negative | Upward |

## Diagnostics Checklist

| Method | Key diagnostic | What to check |
|---|---|---|
| OLS + controls | Sensitivity to controls | Does $\hat\beta$ change when you add/remove controls? |
| Panel FE | Hausman test | FE vs RE — are entity effects correlated with regressors? |
| DiD | Event study plot | Are pre-treatment coefficients near zero? |
| IV | First-stage F-statistic | F > 10 (weak instrument threshold) |
| IV | Over-identification test | If multiple instruments: Hansen J test |

## Common Mistakes

| Mistake | Why it's wrong |
|---|---|
| "I controlled for everything, so it's causal" | You can never control for unobserved confounders |
| Using FE when the treatment varies within entity *and* is correlated with time-varying unobservables | FE only removes time-invariant confounders |
| Not clustering standard errors in panel/DiD | Ignoring within-entity correlation inflates t-stats |
| Weak instruments (first-stage F < 10) | 2SLS estimates are biased toward OLS; inference is unreliable |
| Parallel trends violated | DiD estimate is meaningless if pre-trends diverge |
