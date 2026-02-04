# Part 3: Classification (Recession Probability Models)

This part turns the technical recession label into a probabilistic classification problem: predicting next-quarter recession risk from macro indicators.

## What You Will Learn
- Baselines for rare-event prediction (and why accuracy is misleading)
- Logistic regression as a probability model (log-odds interpretation)
- Threshold selection using decision costs
- Calibration (Brier score, calibration curves)
- Tree-based models and feature importance (and what not to over-interpret)
- Walk-forward validation to measure stability across eras

## Prerequisites
- Foundations (Part 0)
- Macro dataset built in Part 1 (`macro_quarterly.csv`)

## How To Study This Part
- Treat probabilities as decision inputs, not as "truth".
- Always report multiple metrics:
  - ROC-AUC and PR-AUC for ranking
  - Brier score (or log loss) for probability quality
- Always analyze failures (false positives vs false negatives).

## Chapters
- [00_recession_classifier_baselines](00_recession_classifier_baselines.md) — Notebook: [00_recession_classifier_baselines.ipynb](../../../notebooks/03_classification/00_recession_classifier_baselines.ipynb)
- [01_logistic_recession_classifier](01_logistic_recession_classifier.md) — Notebook: [01_logistic_recession_classifier.ipynb](../../../notebooks/03_classification/01_logistic_recession_classifier.ipynb)
- [02_calibration_and_costs](02_calibration_and_costs.md) — Notebook: [02_calibration_and_costs.ipynb](../../../notebooks/03_classification/02_calibration_and_costs.ipynb)
- [03_tree_models_and_importance](03_tree_models_and_importance.md) — Notebook: [03_tree_models_and_importance.ipynb](../../../notebooks/03_classification/03_tree_models_and_importance.ipynb)
- [04_walk_forward_validation](04_walk_forward_validation.md) — Notebook: [04_walk_forward_validation.ipynb](../../../notebooks/03_classification/04_walk_forward_validation.ipynb)
