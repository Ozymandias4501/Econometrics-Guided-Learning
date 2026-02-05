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


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

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

### Core Capstone: how to produce a portfolio-quality econometrics/ML project

The capstone is not “a model.” It is an argument with evidence.

#### 1) Intuition (plain English)

A strong final project answers:
- What is the question and why does it matter?
- What data is used and what is known when?
- What methods are appropriate (prediction vs causal inference)?
- What do results mean, and what are the limitations?

Think like a reviewer: your job is to make it hard to misunderstand or overclaim.

#### 2) Notation + setup (define deliverables)

Capstone deliverables typically include:
- a reproducible pipeline run (artifacts),
- a written report (methods, results, limitations),
- a lightweight dashboard (optional) that visualizes risk/outputs.

#### 3) Assumptions (make them explicit)

You must state assumptions at three levels:
- data (measurement, timing, revisions),
- model (stationarity, independence, functional form),
- interpretation (predictive vs causal).

#### 4) Mechanics: the capstone workflow

1) Define the task and evaluation scheme.
2) Build the modeling table with past-only features.
3) Establish baselines.
4) Fit at least two model families.
5) Diagnose errors and stability.
6) Write the report in a way a non-author could follow.

#### 5) Diagnostics + robustness (minimum set)

1) **Backtest stability**
- do results hold across subperiods or regimes?

2) **Calibration (for probabilities)**
- do predicted risks match observed frequencies?

3) **Sensitivity**
- how fragile are results to feature changes, lag choices, or thresholds?

4) **Limitations**
- explicitly list what your approach cannot answer.

#### 6) Interpretation + reporting

Good reports include:
- a one-paragraph executive summary,
- a methods section with assumptions,
- a results section with uncertainty and robustness,
- a limitations section that is not an afterthought.

**What this does NOT mean**
- A high metric score does not imply a model is “true.”
- A good backtest does not guarantee future performance.

#### Exercises

- [ ] Write a 10-sentence capstone brief (task, data, evaluation, success criteria).
- [ ] Build a baseline model and compare to a stronger model; explain why one wins.
- [ ] List 5 limitations and how they could be addressed in future work.

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
