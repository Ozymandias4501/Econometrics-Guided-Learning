# Guide: 01_capstone_workspace

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/06_capstone/01_capstone_workspace.ipynb`.

This is the capstone: you will synthesize macro concepts, statistical inference, model evaluation, and reproducibility.

### Key Terms (defined)
- **Capstone**: a final project that integrates the entire curriculum.
- **Rubric**: the criteria you will be evaluated against.
- **Monitoring**: how you would detect degradation in a model over time.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Data
- Complete notebook section: Modeling
- Complete notebook section: Interpretation
- Complete notebook section: Artifacts
- Define your final target/feature set and justify key design choices.
- Run walk-forward evaluation and report stability over time.
- Produce a complete report and a working Streamlit dashboard based on artifacts.

### Alternative Example (Not the Notebook Solution)
```python
# Capstone structure (high level):
# 1) build dataset
# 2) train multiple candidate models
# 3) choose threshold and evaluate
# 4) interpret and write report
# 5) serve results in dashboard
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Core Capstone: How To Produce a High-Quality Final Project

A strong capstone is not just a model. It is a full argument:
- what is the task?
- what is the label and why is it defensible?
- what data is available when?
- how did you evaluate without leakage?
- what do results mean and what are limitations?

Write like a reviewer is reading.

### Project Code Map
- `apps/streamlit_app.py`: dashboard that reads artifacts
- `reports/capstone_report.md`: report template/output
- `outputs/`: artifact bundles from training runs (models/metrics/preds/plots)

### Common Mistakes
- Changing feature engineering after looking at test results without re-running from scratch.
- Reporting a single metric without describing decision costs.
- Overclaiming ("this predicts recessions") without scope/limitations.

<a id="summary"></a>
## Summary + Suggested Readings

If you complete the capstone well, you will have a portfolio-quality project: reproducible code, a report, and an interactive dashboard.


Suggested readings:
- Wooldridge (inference) + Hyndman (forecasting) as complementary perspectives
- Mitchell et al.: Model Cards
