# Curriculum Index

This is the navigation hub for the full tutorial project.

## How To Use This Repo
- Work through the notebooks in order (recommended path below).
- Each notebook has a matching deep guide under `docs/guides/<category>/`.
- Notebooks are hands-on: code cells are intentionally incomplete (TODO-driven).
- Each notebook ends with a collapsed **Solutions (Reference)** section to self-check.

## Recommended Learning Path

### Foundations
1. [Notebook: 00_setup](../notebooks/00_foundations/00_setup.ipynb)  
   Guide: [00_setup](guides/00_foundations/00_setup.md)
2. [Notebook: 01_time_series_basics](../notebooks/00_foundations/01_time_series_basics.ipynb)  
   Guide: [01_time_series_basics](guides/00_foundations/01_time_series_basics.md)
3. [Notebook: 02_stats_basics_for_ml](../notebooks/00_foundations/02_stats_basics_for_ml.ipynb)  
   Guide: [02_stats_basics_for_ml](guides/00_foundations/02_stats_basics_for_ml.md)

### Data (Macro + Micro)
1. [Notebook: 00_fred_api_and_caching](../notebooks/01_data/00_fred_api_and_caching.ipynb)  
   Guide: [00_fred_api_and_caching](guides/01_data/00_fred_api_and_caching.md)
2. [Notebook: 01_build_macro_monthly_panel](../notebooks/01_data/01_build_macro_monthly_panel.ipynb)  
   Guide: [01_build_macro_monthly_panel](guides/01_data/01_build_macro_monthly_panel.md)
3. [Notebook: 02_gdp_growth_and_recession_label](../notebooks/01_data/02_gdp_growth_and_recession_label.ipynb)  
   Guide: [02_gdp_growth_and_recession_label](guides/01_data/02_gdp_growth_and_recession_label.md)
4. [Notebook: 03_build_macro_quarterly_features](../notebooks/01_data/03_build_macro_quarterly_features.ipynb)  
   Guide: [03_build_macro_quarterly_features](guides/01_data/03_build_macro_quarterly_features.md)
5. [Notebook: 04_census_api_microdata_fetch](../notebooks/01_data/04_census_api_microdata_fetch.ipynb)  
   Guide: [04_census_api_microdata_fetch](guides/01_data/04_census_api_microdata_fetch.md)

### Regression (Micro then Macro)
1. [Notebook: 00_single_factor_regression_micro](../notebooks/02_regression/00_single_factor_regression_micro.ipynb)  
   Guide: [00_single_factor_regression_micro](guides/02_regression/00_single_factor_regression_micro.md)
2. [Notebook: 01_multifactor_regression_micro_controls](../notebooks/02_regression/01_multifactor_regression_micro_controls.ipynb)  
   Guide: [01_multifactor_regression_micro_controls](guides/02_regression/01_multifactor_regression_micro_controls.md)
3. [Notebook: 02_single_factor_regression_macro](../notebooks/02_regression/02_single_factor_regression_macro.ipynb)  
   Guide: [02_single_factor_regression_macro](guides/02_regression/02_single_factor_regression_macro.md)
4. [Notebook: 03_multifactor_regression_macro](../notebooks/02_regression/03_multifactor_regression_macro.ipynb)  
   Guide: [03_multifactor_regression_macro](guides/02_regression/03_multifactor_regression_macro.md)
5. [Notebook: 04_inference_time_series_hac](../notebooks/02_regression/04_inference_time_series_hac.ipynb)  
   Guide: [04_inference_time_series_hac](guides/02_regression/04_inference_time_series_hac.md)
6. [Notebook: 05_regularization_ridge_lasso](../notebooks/02_regression/05_regularization_ridge_lasso.ipynb)  
   Guide: [05_regularization_ridge_lasso](guides/02_regression/05_regularization_ridge_lasso.md)
7. [Notebook: 06_rolling_regressions_stability](../notebooks/02_regression/06_rolling_regressions_stability.ipynb)  
   Guide: [06_rolling_regressions_stability](guides/02_regression/06_rolling_regressions_stability.md)

### Classification (Technical Recession)
1. [Notebook: 00_recession_classifier_baselines](../notebooks/03_classification/00_recession_classifier_baselines.ipynb)  
   Guide: [00_recession_classifier_baselines](guides/03_classification/00_recession_classifier_baselines.md)
2. [Notebook: 01_logistic_recession_classifier](../notebooks/03_classification/01_logistic_recession_classifier.ipynb)  
   Guide: [01_logistic_recession_classifier](guides/03_classification/01_logistic_recession_classifier.md)
3. [Notebook: 02_calibration_and_costs](../notebooks/03_classification/02_calibration_and_costs.ipynb)  
   Guide: [02_calibration_and_costs](guides/03_classification/02_calibration_and_costs.md)
4. [Notebook: 03_tree_models_and_importance](../notebooks/03_classification/03_tree_models_and_importance.ipynb)  
   Guide: [03_tree_models_and_importance](guides/03_classification/03_tree_models_and_importance.md)
5. [Notebook: 04_walk_forward_validation](../notebooks/03_classification/04_walk_forward_validation.ipynb)  
   Guide: [04_walk_forward_validation](guides/03_classification/04_walk_forward_validation.md)

### Unsupervised (Macro Structure)
1. [Notebook: 01_pca_macro_factors](../notebooks/04_unsupervised/01_pca_macro_factors.ipynb)  
   Guide: [01_pca_macro_factors](guides/04_unsupervised/01_pca_macro_factors.md)
2. [Notebook: 02_clustering_macro_regimes](../notebooks/04_unsupervised/02_clustering_macro_regimes.ipynb)  
   Guide: [02_clustering_macro_regimes](guides/04_unsupervised/02_clustering_macro_regimes.md)
3. [Notebook: 03_anomaly_detection](../notebooks/04_unsupervised/03_anomaly_detection.ipynb)  
   Guide: [03_anomaly_detection](guides/04_unsupervised/03_anomaly_detection.md)

### Model Ops
1. [Notebook: 01_reproducible_pipeline_design](../notebooks/05_model_ops/01_reproducible_pipeline_design.ipynb)  
   Guide: [01_reproducible_pipeline_design](guides/05_model_ops/01_reproducible_pipeline_design.md)
2. [Notebook: 02_build_cli_train_predict](../notebooks/05_model_ops/02_build_cli_train_predict.ipynb)  
   Guide: [02_build_cli_train_predict](guides/05_model_ops/02_build_cli_train_predict.md)
3. [Notebook: 03_model_cards_and_reporting](../notebooks/05_model_ops/03_model_cards_and_reporting.ipynb)  
   Guide: [03_model_cards_and_reporting](guides/05_model_ops/03_model_cards_and_reporting.md)

### Capstone
1. [Notebook: 00_capstone_brief](../notebooks/06_capstone/00_capstone_brief.ipynb)  
   Guide: [00_capstone_brief](guides/06_capstone/00_capstone_brief.md)
2. [Notebook: 01_capstone_workspace](../notebooks/06_capstone/01_capstone_workspace.ipynb)  
   Guide: [01_capstone_workspace](guides/06_capstone/01_capstone_workspace.md)

## Extra References
- [Guides Index](guides/index.md)
- [Monolithic Deep Dive (Optional)](technical_deep_dive.md)
