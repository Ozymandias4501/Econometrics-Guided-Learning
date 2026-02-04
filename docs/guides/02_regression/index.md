# Part 2: Regression (Inference + Prediction)

Regression is the bridge between statistics and ML. In this project you will use regression for two purposes:
- prediction (how accurate is it out-of-sample?)
- inference (what do coefficients mean, and how uncertain are they?)

## What You Will Learn
- OLS mechanics and coefficient interpretation
- Robust standard errors (HC3 for cross-section, HAC/Newey-West for time series)
- Hypothesis testing in regression (t-tests, p-values, confidence intervals)
- Multicollinearity diagnostics (VIF) and why coefficients can be unstable
- Regularization (ridge/lasso) for correlated predictors
- Rolling regressions to study stability/regime changes

## Prerequisites
- Foundations (Part 0)
- Data tables built in Part 1

## How To Study This Part
- Always separate two questions:
  - "Does it predict well?" (out-of-sample)
  - "Is this coefficient interpretable?" (assumptions + diagnostics)
- For every model, write down:
  - the unit interpretation of each feature
  - at least one limitation (confounding, nonstationarity, measurement)

## Chapters
- [00_single_factor_regression_micro](00_single_factor_regression_micro.md) — Notebook: [00_single_factor_regression_micro.ipynb](../../../notebooks/02_regression/00_single_factor_regression_micro.ipynb)
- [01_multifactor_regression_micro_controls](01_multifactor_regression_micro_controls.md) — Notebook: [01_multifactor_regression_micro_controls.ipynb](../../../notebooks/02_regression/01_multifactor_regression_micro_controls.ipynb)
- [02_single_factor_regression_macro](02_single_factor_regression_macro.md) — Notebook: [02_single_factor_regression_macro.ipynb](../../../notebooks/02_regression/02_single_factor_regression_macro.ipynb)
- [03_multifactor_regression_macro](03_multifactor_regression_macro.md) — Notebook: [03_multifactor_regression_macro.ipynb](../../../notebooks/02_regression/03_multifactor_regression_macro.ipynb)
- [04_inference_time_series_hac](04_inference_time_series_hac.md) — Notebook: [04_inference_time_series_hac.ipynb](../../../notebooks/02_regression/04_inference_time_series_hac.ipynb)
- [05_regularization_ridge_lasso](05_regularization_ridge_lasso.md) — Notebook: [05_regularization_ridge_lasso.ipynb](../../../notebooks/02_regression/05_regularization_ridge_lasso.ipynb)
- [06_rolling_regressions_stability](06_rolling_regressions_stability.md) — Notebook: [06_rolling_regressions_stability.ipynb](../../../notebooks/02_regression/06_rolling_regressions_stability.ipynb)
