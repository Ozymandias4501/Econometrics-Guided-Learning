# Guide: 01_panel_fixed_effects_clustered_se

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/07_causal/01_panel_fixed_effects_clustered_se.ipynb`.

This module adds identification-focused econometrics: panels, DiD/event studies, and IV.

### Key Terms (defined)
- **Identification**: assumptions that justify a causal interpretation.
- **Fixed effects (FE)**: controls for time-invariant unit differences.
- **Clustered SE**: allows correlated errors within groups (e.g., state).
- **DiD**: compares changes over time between treated and control units.
- **IV/2SLS**: uses an instrument to address endogeneity.


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

### Core Causal Inference: Identification Before Estimation

In econometrics, the hard part is rarely the regression code. It is the logic of **identification**.

> **Definition:** **Identification** means: “Under which assumptions can we interpret a statistic as a causal effect?”

> **Definition:** **Estimation** means: “How do we compute that statistic from data?”

Most datasets in this repo are observational (not randomized experiments), so causal claims require extra structure:
- a policy cutoff (RD),
- a shock that changes some units but not others (DiD),
- an instrument that shifts treatment but not outcomes directly (IV),
- or a strong structural model (not covered here).

#### Potential outcomes (one-line model)
For each unit $i$ and time $t$:
- $Y_{it}(1)$: outcome if treated
- $Y_{it}(0)$: outcome if not treated

The causal effect is $Y_{it}(1) - Y_{it}(0)$, but we never observe both at once.

#### Why this repo uses “semi-synthetic” exercises
To practice mechanics safely, some notebooks construct a **known treatment effect** on top of real outcomes.
That lets you verify whether your estimator can recover the truth, without pretending this is a real policy evaluation.

### Panel Fixed Effects (Entity + Time)

> **Definition:** A **panel dataset** follows the same units over time (or repeated cross-sections at a unit level).

In this project, the “unit” is a county and time is year.

#### Two-way fixed effects (TWFE)
A common model is:

$$
Y_{it} = \\beta'X_{it} + \\alpha_i + \\gamma_t + \\varepsilon_{it}
$$

- $\\alpha_i$: **entity fixed effects** (time-invariant county differences)
- $\\gamma_t$: **time fixed effects** (common shocks in a given year)

Interpretation:
- $\\beta$ is identified from **within-county changes over time**, after subtracting common year shocks.

Key implications:
- Time-invariant regressors (e.g., a county’s latitude) are absorbed by $\\alpha_i$ and cannot be estimated.
- If treatment varies only at the time level, it is absorbed by $\\gamma_t$.

### Clustered Standard Errors (Why Robust SE Still Isn’t Enough)

Robust (HC) standard errors handle heteroskedasticity.
They do **not** handle correlation in errors across related observations.

> **Definition:** **Clustered standard errors** allow errors to be correlated within groups (clusters), but assume independence across clusters.

Why this matters in panels:
- Counties within the same state can share shocks (policy, labor markets).
- A county’s errors can be correlated over time (serial correlation).

Common choices:
- cluster by entity (county)
- cluster by higher-level geography (state)

Practical caution:
- With very few clusters, cluster-robust inference can be unreliable.
- Always report the number of clusters you used.

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
