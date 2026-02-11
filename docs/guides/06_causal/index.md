# Part 7: Causal Inference (Panels + Quasi-Experiments)

This part adds **identification-focused econometrics** on top of the regression module.
You will use panel data and quasi-experimental designs to estimate causal effects under explicit assumptions.

## What You Will Learn
- How to separate **identification** (assumptions) from **estimation** (math/code)
- Panel data basics and two-way fixed effects (TWFE)
- Clustered standard errors (why they’re often the default)
- Difference-in-differences (DiD) and event studies (leads/lags)
- Instrumental variables (IV) and two-stage least squares (2SLS)

## Prerequisites
- Foundations (Part 0)
- Census/ACS micro dataset skills (Part 1)
- Regression interpretation + robust SE (Part 2)

## How To Study This Part
- For every method, write down:
  - the causal question (intervention, population, time horizon)
  - the identification assumptions (parallel trends, exclusion restriction, etc.)
  - at least one falsification/diagnostic (pre-trends, placebo, weak IV checks)
- Treat these notebooks as **method practice**, not real policy evaluation.

## Chapters
- [00_build_census_county_panel](00_build_census_county_panel.md) — Notebook: [00_build_census_county_panel.ipynb](../../../notebooks/07_causal/00_build_census_county_panel.ipynb)
- [01_panel_fixed_effects_clustered_se](01_panel_fixed_effects_clustered_se.md) — Notebook: [01_panel_fixed_effects_clustered_se.ipynb](../../../notebooks/07_causal/01_panel_fixed_effects_clustered_se.ipynb)
- [02_difference_in_differences_event_study](02_difference_in_differences_event_study.md) — Notebook: [02_difference_in_differences_event_study.ipynb](../../../notebooks/07_causal/02_difference_in_differences_event_study.ipynb)
- [03_instrumental_variables_2sls](03_instrumental_variables_2sls.md) — Notebook: [03_instrumental_variables_2sls.ipynb](../../../notebooks/07_causal/03_instrumental_variables_2sls.ipynb)
