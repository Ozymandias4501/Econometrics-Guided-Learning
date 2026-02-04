# Part 5: Model Ops (Reproducible Runs + Artifacts)

This part turns notebook experiments into a reproducible pipeline with configs and saved artifacts. The goal is not "production MLOps" in full complexity, but real habits that scale.

## What You Will Learn
- Why configs matter (making decisions explicit)
- How to produce run artifacts (models, metrics, predictions, plots)
- How to load artifacts and generate reports
- How to think about monitoring and limitations

## Prerequisites
- A working model from Parts 2-4 (at least one regression or classifier)

## How To Study This Part
- Treat every run as something you may need to reproduce later.
- Write down:
  - which data snapshot you used
  - which features you used
  - how you split/evaluated

## Chapters
- [01_reproducible_pipeline_design](01_reproducible_pipeline_design.md) — Notebook: [01_reproducible_pipeline_design.ipynb](../../../notebooks/05_model_ops/01_reproducible_pipeline_design.ipynb)
- [02_build_cli_train_predict](02_build_cli_train_predict.md) — Notebook: [02_build_cli_train_predict.ipynb](../../../notebooks/05_model_ops/02_build_cli_train_predict.ipynb)
- [03_model_cards_and_reporting](03_model_cards_and_reporting.md) — Notebook: [03_model_cards_and_reporting.ipynb](../../../notebooks/05_model_ops/03_model_cards_and_reporting.ipynb)
