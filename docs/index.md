# Curriculum Index

This is the navigation hub for the curriculum.

## How To Use This Repo
- Work the notebooks in the order below.
- Each notebook has a matching deep guide under `docs/guides/<section>/`.
- Notebooks are hands-on: code cells are intentionally incomplete (TODO-driven).
- Each notebook ends with a collapsed **Solutions (Reference)** section to self-check.

## Recommended Learning Path

The curriculum is six sections, ~32 notebooks, end-to-end on real (or bundled) FRED macro data.

### 1. Statistics Primer (Start Here)
Build the vocabulary for everything that follows. Distributions, sampling, hypothesis testing, and how to read every metric in a regression / ML output table.

1. [00_descriptive_statistics](../notebooks/00a_statistics_primer/00_descriptive_statistics.ipynb) — guide: [00](guides/00a_statistics_primer/00_descriptive_statistics.md)
2. [01_probability_distributions](../notebooks/00a_statistics_primer/01_probability_distributions.ipynb) — guide: [01](guides/00a_statistics_primer/01_probability_distributions.md)
3. [02_sampling_and_central_limit_theorem](../notebooks/00a_statistics_primer/02_sampling_and_central_limit_theorem.ipynb) — guide: [02](guides/00a_statistics_primer/02_sampling_and_central_limit_theorem.md)
4. [03_z_scores_and_standardization](../notebooks/00a_statistics_primer/03_z_scores_and_standardization.ipynb) — guide: [03](guides/00a_statistics_primer/03_z_scores_and_standardization.md)
5. [04_confidence_intervals](../notebooks/00a_statistics_primer/04_confidence_intervals.ipynb) — guide: [04](guides/00a_statistics_primer/04_confidence_intervals.md)
6. [05_hypothesis_testing_foundations](../notebooks/00a_statistics_primer/05_hypothesis_testing_foundations.ipynb) — guide: [05](guides/00a_statistics_primer/05_hypothesis_testing_foundations.md)
7. [06_common_statistical_tests](../notebooks/00a_statistics_primer/06_common_statistical_tests.ipynb) — guide: [06](guides/00a_statistics_primer/06_common_statistical_tests.md)
8. [07_correlation_and_covariance](../notebooks/00a_statistics_primer/07_correlation_and_covariance.ipynb) — guide: [07](guides/00a_statistics_primer/07_correlation_and_covariance.md)
9. [08_metrics_and_accuracy_vs_precision](../notebooks/00a_statistics_primer/08_metrics_and_accuracy_vs_precision.ipynb) — guide: [08](guides/00a_statistics_primer/08_metrics_and_accuracy_vs_precision.md)

### 2. Foundations
Project setup and time-series indexing patterns.

1. [00_setup](../notebooks/00b_foundations/00_setup.ipynb) — guide: [00](guides/00b_foundations/00_setup.md)
2. [01_time_series_basics](../notebooks/00b_foundations/01_time_series_basics.ipynb) — guide: [01](guides/00b_foundations/01_time_series_basics.md)

### 3. Data (FRED Macro)
Pull macro data from FRED, cache it, and build the panel everything else uses.

1. [00_fred_api_and_caching](../notebooks/01_data/00_fred_api_and_caching.ipynb) — guide: [00](guides/01_data/00_fred_api_and_caching.md)
2. [01_build_macro_monthly_panel](../notebooks/01_data/01_build_macro_monthly_panel.ipynb) — guide: [01](guides/01_data/01_build_macro_monthly_panel.md)
3. [02_gdp_growth_and_recession_label](../notebooks/01_data/02_gdp_growth_and_recession_label.ipynb) — guide: [02](guides/01_data/02_gdp_growth_and_recession_label.md)
4. [03_build_macro_quarterly_features](../notebooks/01_data/03_build_macro_quarterly_features.ipynb) — guide: [03](guides/01_data/03_build_macro_quarterly_features.md)

### 4. Regression (OLS Core)
Single- and multi-factor OLS, functional forms, residual diagnostics, HAC robust inference, regularization, rolling stability.

1. [00_single_factor_regression_micro](../notebooks/02_regression/00_single_factor_regression_micro.ipynb) — guide: [00](guides/02_regression/00_single_factor_regression_micro.md)
2. [01_multifactor_regression_micro_controls](../notebooks/02_regression/01_multifactor_regression_micro_controls.ipynb) — guide: [01](guides/02_regression/01_multifactor_regression_micro_controls.md)
3. [02_single_factor_regression_macro](../notebooks/02_regression/02_single_factor_regression_macro.ipynb) — guide: [02](guides/02_regression/02_single_factor_regression_macro.md)
4. [02a_functional_forms_and_interactions](../notebooks/02_regression/02a_functional_forms_and_interactions.ipynb)
5. [03_multifactor_regression_macro](../notebooks/02_regression/03_multifactor_regression_macro.ipynb) — guide: [03](guides/02_regression/03_multifactor_regression_macro.md)
6. [04_inference_time_series_hac](../notebooks/02_regression/04_inference_time_series_hac.ipynb) — guide: [04](guides/02_regression/04_inference_time_series_hac.md)
7. [04a_residual_diagnostics](../notebooks/02_regression/04a_residual_diagnostics.ipynb)
8. [05_regularization_ridge_lasso](../notebooks/02_regression/05_regularization_ridge_lasso.ipynb) — guide: [05](guides/02_regression/05_regularization_ridge_lasso.md)
9. [06_rolling_regressions_stability](../notebooks/02_regression/06_rolling_regressions_stability.ipynb) — guide: [06](guides/02_regression/06_rolling_regressions_stability.md)

### 5. ML for Regression (Random Forest + XGBoost)
Same target as section 4, same features, same split — but with tree-based models. Compare RMSE / MAE / R² head-to-head against OLS.

1. [00_ml_regression_setup_and_baselines](../notebooks/02b_ml_regression/00_ml_regression_setup_and_baselines.ipynb) — guide: [00](guides/02b_ml_regression/00_ml_regression_setup_and_baselines.md)
2. [01_random_forest_regressor_macro](../notebooks/02b_ml_regression/01_random_forest_regressor_macro.ipynb) — guide: [01](guides/02b_ml_regression/01_random_forest_regressor_macro.md)
3. [02_xgboost_regressor_macro](../notebooks/02b_ml_regression/02_xgboost_regressor_macro.ipynb) — guide: [02](guides/02b_ml_regression/02_xgboost_regressor_macro.md)

### 6. Classification (Recession Prediction)
Logistic regression and tree/XGBoost classifiers. Confusion matrix, accuracy, precision, recall, F1, ROC-AUC, calibration, walk-forward validation.

1. [00_recession_classifier_baselines](../notebooks/03_classification/00_recession_classifier_baselines.ipynb) — guide: [00](guides/03_classification/00_recession_classifier_baselines.md)
2. [01_logistic_recession_classifier](../notebooks/03_classification/01_logistic_recession_classifier.ipynb) — guide: [01](guides/03_classification/01_logistic_recession_classifier.md)
3. [02_calibration_and_costs](../notebooks/03_classification/02_calibration_and_costs.ipynb) — guide: [02](guides/03_classification/02_calibration_and_costs.md)
4. [03_tree_models_and_importance](../notebooks/03_classification/03_tree_models_and_importance.ipynb) — guide: [03](guides/03_classification/03_tree_models_and_importance.md)
5. [04_walk_forward_validation](../notebooks/03_classification/04_walk_forward_validation.ipynb) — guide: [04](guides/03_classification/04_walk_forward_validation.md)

## Extra References
- [Guides Index](guides/index.md)
- Cheatsheets: [classification metrics](cheatsheets/classification_metrics.md), [hypothesis testing](cheatsheets/hypothesis_testing.md), [model comparison](cheatsheets/model_comparison.md), [regression diagnostics](cheatsheets/regression_diagnostics.md)
