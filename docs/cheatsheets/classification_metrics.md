# Cheatsheet: Classification Metrics

## The Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

**Reading the matrix**: The rows are ground truth, the columns are what the model predicted. A false negative means the event happened but the model said it didn't. A false positive means the model raised an alarm but nothing actually happened.

**Example**: A recession classifier predicts whether next quarter is a recession.
- **TP**: Model predicted recession, recession occurred. Good — policymakers prepared.
- **FN**: Model predicted no recession, recession occurred. Dangerous — no preparation.
- **FP**: Model predicted recession, no recession occurred. Costly but survivable — unnecessary tightening.
- **TN**: Model predicted no recession, no recession occurred. Routine — nothing needed.

## Core Metrics

| Metric | Formula | What it tells you | When it matters most |
|---|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall fraction of correct predictions | Only meaningful when classes are roughly balanced (e.g., 50/50 split) |
| **Precision** | $\frac{TP}{TP + FP}$ | When the model says "positive," how often is it right? High precision means few false alarms | When acting on a positive prediction is expensive (e.g., launching an intervention based on a predicted recession) |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | What fraction of actual positive cases did the model identify? High recall means few missed events | When failing to detect a positive is dangerous (e.g., missing an actual recession, failing to diagnose a disease) |
| **Specificity** | $\frac{TN}{TN + FP}$ | What fraction of actual negative cases did the model correctly classify? | When false alarms have real costs (e.g., unnecessary medical procedures) |
| **F1 Score** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Single number that balances precision and recall. The harmonic mean penalizes extreme imbalance between the two | When you need one number and neither precision nor recall dominates in importance |

### Why accuracy is misleading with imbalanced classes

Recessions occur in roughly 7% of quarters. A model that always predicts "no recession" achieves 93% accuracy while being completely useless — it never detects the event you care about. In imbalanced settings, precision, recall, and F1 are far more informative than accuracy.

## Threshold-Free Metrics

Most classifiers output a probability (e.g., 0.73), and you choose a threshold to convert it to a binary prediction. The metrics above all depend on that threshold choice. These metrics evaluate the model across all possible thresholds:

| Metric | What it measures | Range | Perfect | Interpretation |
|---|---|---|---|---|
| **ROC-AUC** | Area under the ROC curve (True Positive Rate vs False Positive Rate at every threshold) | 0.5 (random) to 1.0 | 1.0 | Probability that the model ranks a random positive case higher than a random negative case |
| **PR-AUC** | Area under the Precision-Recall curve | Baseline = prevalence, to 1.0 | 1.0 | Summarizes the precision-recall trade-off; more sensitive to performance on the minority class than ROC-AUC |
| **Brier Score** | Mean squared difference between predicted probabilities and actual outcomes: $\frac{1}{n}\sum(p_i - y_i)^2$ | 0 to 1 | 0.0 | Measures calibration — a model that says "30% chance of recession" should be right about 30% of the time across all such predictions |

**When to use which**:
- **ROC-AUC**: Solid default for balanced or moderately imbalanced data. Measures the model's ability to discriminate between classes regardless of threshold.
- **PR-AUC**: Use when the positive class is rare (e.g., recessions at 7% prevalence, rare diagnoses). ROC-AUC can look optimistic in these settings because it rewards correctly classifying the large negative class; PR-AUC focuses on how well the model handles positives.
- **Brier Score**: Use when you care about the predicted probabilities themselves, not just the ranking. A well-calibrated model is essential if you feed probabilities into downstream decisions (e.g., expected cost calculations).

## The Precision-Recall Trade-off

Every threshold choice trades precision for recall:

- **Lowering the threshold** (e.g., from 0.50 to 0.20): The model flags more cases as positive. You catch more true positives (recall goes up), but you also generate more false alarms (precision goes down).
- **Raising the threshold** (e.g., from 0.50 to 0.80): The model is more conservative. Fewer false alarms (precision goes up), but you miss more true positives (recall goes down).

There is no free lunch — improving one side degrades the other. The right threshold depends on the relative costs of each type of error.

### Cost-Based Threshold Selection

If the cost of a false negative is $c_{FN}$ times the cost of a false positive, the decision-theoretically optimal threshold is:

$$
\text{threshold}^* = \frac{1}{1 + c_{FN}}
$$

**Worked example — recession prediction**:
- Missing a recession means portfolios are unhedged, fiscal responses are delayed. Suppose this costs 5x more than a false alarm (unnecessary tightening).
- $c_{FN} = 5$, so threshold $= \frac{1}{1 + 5} = 0.167$
- Flag a recession whenever $\hat p > 0.167$, much lower than the default 0.50.
- This catches more recessions (high recall) at the cost of more false alarms (lower precision) — exactly the trade-off you want when misses are expensive.

**Worked example — elective surgery screening**:
- Flagging a patient as high-risk leads to costly additional testing. A false alarm wastes resources but isn't dangerous. Missing a true high-risk patient has serious consequences. Suppose $c_{FN} = 3$.
- threshold $= \frac{1}{1 + 3} = 0.25$

## Calibration

A model is **well-calibrated** if its predicted probabilities match observed frequencies. If the model predicts $p = 0.30$ for a group of observations, roughly 30% of them should actually be positive.

**Why it matters**: ROC-AUC and F1 only measure ranking and classification — a model can rank cases perfectly but assign probabilities that are wildly wrong. If you use predicted probabilities for risk scoring, cost-benefit analysis, or communicating uncertainty to stakeholders, calibration matters.

**How to check**: Plot predicted probabilities (binned) on the x-axis vs observed frequency on the y-axis. A perfectly calibrated model follows the 45-degree line. `sklearn.calibration.calibration_curve` computes this.

**How to fix**: Platt scaling (logistic regression on the model's outputs) or isotonic regression can recalibrate a poorly calibrated model without changing its rankings.

## Python Quick Reference

```python
from src.evaluation import classification_metrics

# All metrics at once
m = classification_metrics(y_true, y_prob, threshold=0.5)
# Returns: threshold, accuracy, precision, recall, f1, roc_auc, pr_auc, brier

# Adjusting the threshold
m_sensitive = classification_metrics(y_true, y_prob, threshold=0.167)

# sklearn individual metrics
from sklearn.metrics import (
    confusion_matrix,         # confusion_matrix(y_true, y_pred)
    classification_report,    # classification_report(y_true, y_pred)
    roc_curve,                # fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    precision_recall_curve,   # prec, rec, thresholds = precision_recall_curve(...)
)

# Calibration
from sklearn.calibration import calibration_curve
fraction_of_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)
```

## Common Pitfalls

| Mistake | Why it's wrong | What to do instead |
|---|---|---|
| Using accuracy with imbalanced classes | A model predicting the majority class every time looks accurate but detects nothing | Use F1, PR-AUC, or recall depending on your cost structure |
| Optimizing threshold on test data | The threshold becomes tuned to the test set, overstating generalization performance | Tune threshold on a validation set or within cross-validation; evaluate final performance on a held-out test set |
| Ignoring calibration | High ROC-AUC with bad calibration means the probabilities are unreliable for decision-making | Check calibration plots; apply Platt scaling if needed |
| Reporting precision without recall (or vice versa) | Each metric alone hides the trade-off — you can get perfect precision by making only one very confident prediction | Always report both, or use F1 as a summary |
| Random train/test splits on time-series data | Future observations leak into the training set, producing unrealistically optimistic metrics | Use walk-forward validation: train on the past, test on the future |
| Comparing models at the default 0.50 threshold | The optimal threshold depends on class balance and costs, not on convention | Compare models using threshold-free metrics (ROC-AUC, PR-AUC) or at a cost-calibrated threshold |
