# Guide: 01_logistic_recession_classifier

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/01_logistic_recession_classifier.ipynb`.

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
- Complete notebook section: Train/test split
- Complete notebook section: Fit logistic
- Complete notebook section: ROC/PR
- Complete notebook section: Threshold tuning
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


### Deep Dive: Logistic Regression, Odds, and Interpreting Coefficients

Logistic regression models:
- score: `z = β0 + β1 x1 + ...`
- probability: `p = 1 / (1 + exp(-z))`

**Odds and log-odds**
- odds = `p / (1-p)`
- log-odds = `log(p/(1-p))`
- Logistic regression is linear in log-odds.

**Coefficient interpretation (key idea)**
- A +1 increase in feature x_j adds β_j to log-odds.
- `exp(β_j)` is the odds multiplier (holding other features fixed).

**Python demo: odds multipliers**
```python
import numpy as np
beta = 0.7
print('odds multiplier:', np.exp(beta))
```

**Scaling matters**
- If x is in large units, β will be small; if x is standardized, β is per 1 std dev.
- For interpretation, standardize or be explicit about units.


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
