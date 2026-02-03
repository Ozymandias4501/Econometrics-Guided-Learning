# Guide: 00_recession_classifier_baselines

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/00_recession_classifier_baselines.ipynb`.

This classification module predicts **next-quarter technical recession** from macro indicators.

### Key Terms (defined)
- **Logistic regression**: a linear model that outputs probabilities via a sigmoid function.
- **Log-odds**: `log(p/(1-p))`; logistic regression is linear in log-odds.
- **Threshold**: rule converting probability into class (e.g., 1 if p>=0.5).
- **Precision/Recall**: trade off false positives vs false negatives.
- **ROC-AUC / PR-AUC**: threshold-free ranking metrics.
- **Calibration**: whether predicted probabilities match observed frequencies.
- **Brier score**: mean squared error of probabilities (lower is better).


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Load data
- Complete notebook section: Define baselines
- Complete notebook section: Evaluate metrics
- Establish baselines before fitting any model.
- Fit at least one probabilistic classifier and evaluate ROC-AUC, PR-AUC, and Brier score.
- Pick a threshold intentionally (cost-based or metric-based) and justify it.

### Alternative Example (Not the Notebook Solution)
```python
# Toy logistic regression (not the notebook data):
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 3))
p = 1 / (1 + np.exp(-(0.2 + 1.0*X[:,0] - 0.8*X[:,1])))
y = rng.binomial(1, p)

clf = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])
clf.fit(X, y)
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Logistic Regression Mechanics
- Score: `z = β0 + β1 x1 + ...`
- Probability: `p = 1 / (1 + exp(-z))`
- Training minimizes **log loss** (cross-entropy), not squared error.

### Metrics: When to Use Which
- ROC-AUC: good for ranking; can be optimistic with heavy class imbalance.
- PR-AUC: focuses on the positive class; often more informative for rare recessions.
- Brier score: penalizes miscalibrated probabilities.

### Thresholds and Decision Costs
- If false positives are expensive (crying wolf), raise threshold.
- If missing a recession is expensive, lower threshold.


### Deep Dive: Class Imbalance and Why Accuracy Lies

Recessions are rare. That means classification is an imbalanced problem.

#### Key Terms (defined)
- **Base rate**: prevalence of the positive class (fraction of recession quarters).
- **Imbalanced data**: one class is much rarer than the other.
- **Precision**: among predicted positives, how many were true positives?
- **Recall**: among true positives, how many did we catch?
- **ROC-AUC**: ranking quality across thresholds (can look good even if precision is poor).
- **PR-AUC**: focuses on positive-class retrieval (often more honest for rare events).
- **Proper scoring rule**: rewards calibrated probabilities (log loss, Brier score).

#### The accuracy trap
- If recessions happen 10% of the time, a model that always predicts "no recession" has 90% accuracy.
- But it is useless.

#### Baselines you should always compute
- Majority class (always 0)
- Persistence (predict next = current)
- Simple heuristic (e.g., yield curve inversion rule)

#### Python demo: why PR matters
```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(0)
n = 500
y = rng.binomial(1, 0.1, size=n)  # 10% positives

# A weak signal score
score = 0.2*y + rng.normal(scale=1.0, size=n)

print('ROC-AUC:', roc_auc_score(y, score))
print('PR-AUC :', average_precision_score(y, score))
```

#### Decision framing
- Choose thresholds based on *costs* (false positives vs false negatives), not vibes.
- Calibration matters if you want probabilities you can act on.


### Common Mistakes
- Reporting only accuracy (can be misleading if recessions are rare).
- Picking threshold=0.5 by default without considering costs.
- Evaluating with random splits (time leakage).

<a id="summary"></a>
## Summary + Suggested Readings

You should now be able to build a recession probability model and explain:
- what the probability means,
- how you evaluated it, and
- why your chosen threshold makes sense.


Suggested readings:
- scikit-learn docs: classification metrics, calibration
- Murphy: Machine Learning (probabilistic interpretation)
- Applied time-series evaluation articles (walk-forward validation)
