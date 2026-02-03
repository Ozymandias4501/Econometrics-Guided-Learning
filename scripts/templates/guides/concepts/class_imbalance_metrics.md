### Deep Dive: Class Imbalance and Why Accuracy Lies

Recessions are rare. That makes recession prediction an imbalanced classification problem.

#### Key terms (defined)
> **Definition:** The **base rate** is the prevalence of the positive class (fraction of recession quarters).

> **Definition:** **Imbalanced data** means one class is much rarer than the other.

> **Definition:** **Accuracy** is $(TP + TN) / (TP + TN + FP + FN)$.

> **Definition:** **Precision** is $TP / (TP + FP)$.

> **Definition:** **Recall** is $TP / (TP + FN)$.

Where $TP, TN, FP, FN$ are the confusion-matrix counts.

#### The accuracy trap
If recessions happen 10% of the time, predicting "no recession" always gives 90% accuracy.
That model is useless.

#### Metrics you should always report
- PR-AUC (rare-event focus)
- ROC-AUC (ranking)
- Brier score or log loss (probability quality)

#### Python demo: why PR-AUC can be more honest than ROC-AUC
```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(0)

# 10% positives
n = 500
y = rng.binomial(1, 0.1, size=n)

# A weak signal score
score = 0.2 * y + rng.normal(scale=1.0, size=n)

print('ROC-AUC:', roc_auc_score(y, score))
print('PR-AUC :', average_precision_score(y, score))
```

#### Baselines you should compute
- Majority class baseline
- Persistence baseline (predict next = current)
- Simple heuristic baseline (economic rule-of-thumb)

#### Decision framing
Ultimately you will pick a threshold and make decisions.
Define costs for false positives vs false negatives.
