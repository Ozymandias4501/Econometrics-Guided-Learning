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

### Why configs matter
- They turn hidden notebook state into explicit, reviewable decisions.

### Dataset hashing
- Hashes help you confirm which dataset a model was trained on.
- In production you would also track schema versions and feature code versions.


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
