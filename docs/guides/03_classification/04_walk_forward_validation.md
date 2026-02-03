# Guide: 04_walk_forward_validation

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/04_walk_forward_validation.ipynb`.

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
- Complete notebook section: Walk-forward splits
- Complete notebook section: Metric stability
- Complete notebook section: Failure analysis
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


### Deep Dive: Walk-Forward Validation (Stability Over Time)

**Walk-forward validation** repeatedly trains on the past and tests on the next time block.
It answers: "does my model work across multiple eras, or only in one?"

**Why it's important in economics**
- Relationships change (structural breaks).
- Policy regimes shift.
- A single split can hide fragility.

**Pseudo-code**
```python
# for each fold:
#   train = data[:t]
#   test  = data[t:t+h]
#   fit model
#   compute metrics
```

**Interpretation**
- If metrics vary widely across folds, your model is regime-sensitive.
- This can be a reason to retrain more frequently or include regime features.


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
