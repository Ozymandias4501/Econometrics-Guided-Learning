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
