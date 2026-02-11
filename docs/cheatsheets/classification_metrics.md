# Cheatsheet: Classification Metrics

## The Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

## Core Metrics

| Metric | Formula | Intuition | When to care |
|---|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | % of all predictions correct | Balanced classes only |
| **Precision** | $\frac{TP}{TP + FP}$ | "Of those I flagged, how many were real?" | When false alarms are costly |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | "Of real positives, how many did I catch?" | When missing positives is costly |
| **Specificity** | $\frac{TN}{TN + FP}$ | "Of real negatives, how many did I correctly ignore?" | Complementary to recall |
| **F1 Score** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of precision & recall | When you need a single number balancing both |

## Threshold-Free Metrics

| Metric | What it measures | Range | Perfect |
|---|---|---|---|
| **ROC-AUC** | Area under the ROC curve (TPR vs FPR at all thresholds) | 0 to 1 | 1.0 |
| **PR-AUC** | Area under the Precision-Recall curve | 0 to 1 | 1.0 |
| **Brier Score** | Mean squared error of probability predictions | 0 to 1 | 0.0 |

**When to use which**:
- **ROC-AUC**: Good general-purpose metric. Can be misleading with highly imbalanced classes.
- **PR-AUC**: Better than ROC-AUC when the positive class is rare (e.g., recessions).
- **Brier Score**: Measures calibration — are your predicted probabilities trustworthy?

## The Precision-Recall Trade-off

Lowering the classification threshold:
- **Increases recall** (you catch more positives)
- **Decreases precision** (you also flag more false positives)

Raising the threshold does the opposite.

### Cost-Based Threshold Selection

If a false negative is $c_{FN}$ times worse than a false positive:

$$
\text{Optimal threshold} = \frac{1}{1 + c_{FN}}
$$

**Example**: Missing a recession ($c_{FN} = 5$) is 5x worse than a false alarm:

$$
\text{threshold} = \frac{1}{1 + 5} = 0.167
$$

Flag as positive when predicted probability exceeds 0.167 (much lower than the default 0.50).

## Python Quick Reference

```python
from src.evaluation import classification_metrics

# All metrics at once
m = classification_metrics(y_true, y_prob, threshold=0.5)
# Returns: accuracy, precision, recall, f1, roc_auc, pr_auc, brier

# sklearn individual metrics
from sklearn.metrics import (
    confusion_matrix,         # confusion_matrix(y_true, y_pred)
    classification_report,    # classification_report(y_true, y_pred)
    roc_curve,                # fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    precision_recall_curve,   # prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
)
```

## Common Pitfalls

| Mistake | Why it's wrong |
|---|---|
| Using accuracy with imbalanced classes | A model that always predicts "no recession" gets 93% accuracy but catches 0 recessions |
| Optimizing threshold on test data | Data leakage — use validation set for threshold tuning |
| Ignoring calibration | ROC-AUC can be high even if probabilities are poorly calibrated |
| Reporting only one metric | Precision without recall (or vice versa) hides the trade-off |
| Evaluating on random splits for time series | Future data in training set — use walk-forward validation |
