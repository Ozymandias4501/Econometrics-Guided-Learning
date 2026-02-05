# Guide: 03_instrumental_variables_2sls

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/07_causal/03_instrumental_variables_2sls.ipynb`.

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
- Complete notebook section: Simulate endogeneity
- Complete notebook section: OLS vs 2SLS
- Complete notebook section: First-stage + weak IV checks
- Complete notebook section: Interpretation + limitations
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

### Instrumental Variables (IV) and 2SLS

IV is used when a regressor is **endogenous** (correlated with the error term), often due to:
- omitted variables,
- reverse causality,
- measurement error.

> **Definition:** An **instrument** $Z$ must satisfy:
1) **Relevance:** $Z$ shifts the endogenous regressor $X$ (strong first stage).
2) **Exclusion:** $Z$ affects $Y$ only through $X$ (no direct path).

#### 2SLS mechanics (high level)
1. First stage: regress $X$ on $Z$ (and controls) to get predicted $\\hat{X}$.
2. Second stage: regress $Y$ on $\\hat{X}$ (and controls).

#### Weak instruments
If $Z$ barely predicts $X$, 2SLS can be badly behaved.
Always inspect first-stage strength (e.g., an F-statistic style check).

Interpretation caution:
- With heterogeneous treatment effects, IV often identifies a **local** effect (LATE) for compliers.

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
