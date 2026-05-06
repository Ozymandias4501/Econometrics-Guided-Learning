# Part 02b: Machine Learning for Regression

This part takes the OLS regression problem from `02_regression/` — predicting next-quarter real GDP growth from lagged macro indicators — and reruns it with two of the most widely used ML regression models: Random Forest and XGBoost. Same target. Same features. Same train/test split. Different model class.

## What You Will Learn
- How to set up a fair head-to-head between linear and tree-based models on time-series data
- How to use `walk_forward_splits` from `src/evaluation.py` for time-aware cross-validation
- How `RandomForestRegressor` works (bagging, decorrelating trees, why averaging reduces variance)
- How `XGBoost` differs (sequential boosting, the `learning_rate` × `n_estimators` tradeoff, early stopping)
- Permutation importance vs built-in importance, and why permutation importance is the honest one
- How to read RMSE / MAE / R² out-of-sample numbers and tell when a model is genuinely better

## Prerequisites
- The full `02_regression/` section (you should be comfortable with OLS, HAC inference, and reading regression output)
- The metrics primer in `00a_statistics_primer/08_metrics_and_accuracy_vs_precision`
- Familiarity with `src/evaluation.py` (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`)

## How To Study This Part
- Work `00 → 02` in order. Notebook `00` records the OLS baseline numbers; `01` and `02` compare against them.
- After each notebook, write down the test RMSE / MAE / R² you got. These numbers are the whole point.
- The ML models will not always beat OLS on this small synthetic macro panel — that is a real-world result, not a bug. Write 2–3 sentences after each model on *why* you think it did or did not win.

## Chapters
- [00_ml_regression_setup_and_baselines](00_ml_regression_setup_and_baselines.md) — Notebook: [00_ml_regression_setup_and_baselines.ipynb](../../../notebooks/02b_ml_regression/00_ml_regression_setup_and_baselines.ipynb)
- [01_random_forest_regressor_macro](01_random_forest_regressor_macro.md) — Notebook: [01_random_forest_regressor_macro.ipynb](../../../notebooks/02b_ml_regression/01_random_forest_regressor_macro.ipynb)
- [02_xgboost_regressor_macro](02_xgboost_regressor_macro.md) — Notebook: [02_xgboost_regressor_macro.ipynb](../../../notebooks/02b_ml_regression/02_xgboost_regressor_macro.ipynb)
