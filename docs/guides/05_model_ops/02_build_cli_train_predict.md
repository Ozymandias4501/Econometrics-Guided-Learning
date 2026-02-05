# Guide: 02_build_cli_train_predict

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/05_model_ops/02_build_cli_train_predict.ipynb`.

This module turns notebooks into a reproducible workflow with configs, scripts, and artifacts.

### Key Terms (defined)
- **Artifact**: a saved output (model file, metrics JSON, predictions CSV).
- **Run ID**: a unique identifier for a training run.
- **Config**: a file that records choices (series list, features, split rules).
- **Reproducibility**: ability to re-run and get consistent outputs.


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Training CLI
- Complete notebook section: Prediction CLI
- Complete notebook section: Artifacts
- Run the build/train scripts and confirm outputs/<run_id> is created.
- Inspect run metadata and explain what is captured (and what is missing).
- Make at least one CLI extension (flag or config option).

### Alternative Example (Not the Notebook Solution)
```python
# Toy artifact layout:
# outputs/20250101_120000/
#   model.joblib
#   metrics.json
#   predictions.csv
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Core Model Ops: reproducibility, artifacts, and “make it runnable by someone else”

Model ops in this repo is intentionally lightweight: the goal is not production ML infrastructure, but **reproducible experiments**.

#### 1) Intuition (plain English)

If a result cannot be reproduced, it is not trustworthy.
Reproducibility in applied econometrics means:
- you can re-run data building + training and get the same outputs,
- you can explain which config/dataset produced which figure/table,
- you can compare runs without guessing.

#### 2) Notation + setup (define terms)

> **Definition:** An **artifact** is a saved output: model file, metrics JSON, predictions CSV, plots.

> **Definition:** A **run ID** is a unique identifier for one training/evaluation run.

> **Definition:** A **config** is a recorded set of choices (features, model type, hyperparameters, split dates).

#### 3) Assumptions (what reproducibility requires)

To reproduce results you need:
- fixed code + fixed config,
- fixed data version (processed file hash or timestamp),
- controlled randomness (seeds),
- clear train/test boundaries.

#### 4) Mechanics: the minimal pipeline

This repo’s scripts are organized around:
- build datasets → train → evaluate → save artifacts → make a report/dashboard.

Good run outputs typically include:
- model object (pickle/joblib),
- metrics (JSON),
- predictions (CSV with timestamps),
- a small “run summary” (config + git hash if desired).

#### 5) Diagnostics + robustness (minimum set)

1) **Re-run check**
- run the same script twice with the same config and confirm outputs match.

2) **Config traceability**
- every artifact folder should contain the config that produced it.

3) **Data provenance**
- record whether you used sample vs processed data.

4) **Monitoring mindset (even in a notebook project)**
- if this were shipped, what would you track over time? (drift, calibration, performance)

#### 6) Interpretation + reporting

Model ops is not about claiming your model is “deployable.”
It is about making your analysis:
- auditable,
- comparable,
- easy to extend.

#### Exercises

- [ ] Run a training script that writes outputs and confirm artifacts appear under a run ID directory.
- [ ] Add one field to a metrics JSON (e.g., split dates) and verify it is written.
- [ ] Re-run with a different config and compare runs in a short table.

### Project Code Map
- `configs/recession.yaml`: example config for recession training
- `scripts/build_datasets.py`: dataset builder (writes to data/processed/)
- `scripts/train_recession.py`: training script (writes to outputs/<run_id>/)
- `scripts/predict_recession.py`: prediction script (writes predictions.csv)

### Common Mistakes
- Overwriting outputs without run IDs (losing provenance).
- Not recording feature list used for training (cannot reproduce predictions).
- Mixing data preparation and modeling in one script without clear interfaces.

<a id="summary"></a>
## Summary + Suggested Readings

You now have a minimal 'model ops' workflow: build datasets, train models, save artifacts, and load them for prediction.


Suggested readings:
- Google: ML Test Score (checklist for ML systems)
- Model Cards (Mitchell et al.)
