### Deep Dive: Calibration, Brier Score, and Decision Thresholds

In classification, you often want probabilities, not just labels.

> **Definition:** A model is **calibrated** if events predicted with probability 0.3 happen about 30% of the time.

#### Brier score (math)
> **Definition:** The **Brier score** is a proper scoring rule for probability forecasts.

For binary outcomes $y_i \in \{0,1\}$ and predicted probabilities $p_i$:

$$
\mathrm{Brier} = \frac{1}{n} \sum_{i=1}^n (p_i - y_i)^2
$$

Lower is better.

#### Why calibration matters for recession risk
A recession probability model is only useful if you can make decisions from its probabilities:
- allocate risk
- run stress tests
- change thresholds based on costs

If probabilities are not calibrated, "30%" and "70%" are not meaningful signals.

#### Calibration curve (reliability diagram)
A calibration curve groups predictions into bins and compares:
- average predicted probability in the bin
- actual fraction of positives in the bin

If the curve follows the diagonal, calibration is good.

#### Python demo: calibration and Brier (commented)
```python
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# y_true: 0/1 outcomes
# y_prob: predicted probabilities

# Example placeholders:
# y_true = np.array([...])
# y_prob = np.array([...])

# Brier score
# print('brier:', brier_score_loss(y_true, y_prob))

# Calibration curve
# prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
# print(prob_pred)
# print(prob_true)
```

#### Thresholds and decision costs
> **Definition:** A **decision threshold** converts probabilities to class labels (e.g., predict recession if p >= 0.4).

A good threshold depends on costs:
- false positives (crying wolf)
- false negatives (missing a recession)

A common pattern:
1. define a cost ratio (how bad is FN vs FP?)
2. choose threshold to minimize expected cost

#### Debug checklist
1. Always compute base rate (how rare is the positive class?).
2. Report PR-AUC and Brier score (not just accuracy).
3. Compare calibrated vs uncalibrated models.
4. Re-check calibration across eras (walk-forward).
