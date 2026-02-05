### Deep Dive: Class imbalance and metrics (why accuracy is misleading)

Recessions are rare. Rare events require different evaluation habits.

#### 1) Intuition (plain English)

If 95% of quarters are “no recession,” a dumb model that always predicts “no recession” has 95% accuracy.
That accuracy is useless.

So we use metrics that focus on:
- ranking risk (AUC),
- detecting positives (recall),
- avoiding false alarms (precision),
- probability quality (Brier/log loss).

#### 2) Notation + setup (define symbols)

Confusion matrix terms:
- TP: true positives
- FP: false positives
- TN: true negatives
- FN: false negatives

Key metrics:

$$
\\text{Precision} = \\frac{TP}{TP + FP}
\\qquad
\\text{Recall} = \\frac{TP}{TP + FN}
$$

$$
\\text{F1} = 2 \\cdot \\frac{\\text{Precision}\\cdot \\text{Recall}}{\\text{Precision}+\\text{Recall}}
$$

Baseline (prevalence):
$$
\\pi = \\Pr(y=1).
$$

#### 3) Assumptions (what metrics assume)

Metrics assume:
- you evaluate on future-like data (time-aware splits),
- labels are correctly aligned to horizon,
- the positive class definition is stable.

#### 4) Estimation mechanics: ranking vs thresholding

Two different tasks:
- **ranking:** can the model rank high-risk periods above low-risk periods? (ROC-AUC, PR-AUC)
- **decisions:** choose a threshold $\\tau$ and act (precision/recall at $\\tau$)

PR-AUC is often more informative than ROC-AUC when positives are rare.

#### 5) Inference: cost is part of evaluation

A threshold is not a statistical property; it is a decision rule.
Choosing it requires a cost story:
- false negatives (missed recessions) vs false positives (false alarms).

#### 6) Diagnostics + robustness (minimum set)

1) **Report prevalence**
- always report the positive rate in the test period.

2) **Use PR curves**
- PR curves are sensitive to imbalance and directly reflect precision/recall trade-offs.

3) **Threshold sweep**
- show metrics across thresholds; do not report a single arbitrary threshold.

4) **Error analysis**
- inspect false positives and false negatives; are they clustered in certain regimes?

#### 7) Interpretation + reporting

Report:
- ROC-AUC + PR-AUC,
- at least one thresholded operating point (precision/recall),
- and how threshold was chosen.

**What this does NOT mean**
- a high accuracy can be meaningless under imbalance,
- AUC does not tell you calibration.

#### Exercises

- [ ] Compute accuracy, precision, recall for a baseline “always negative” classifier; interpret.
- [ ] Plot ROC and PR curves and explain why PR is more informative here.
- [ ] Choose a threshold based on a cost story and report the resulting confusion matrix.
