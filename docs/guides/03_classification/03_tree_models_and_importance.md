# Guide: 03_tree_models_and_importance

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/03_tree_models_and_importance.ipynb`.

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
- Complete notebook section: Fit tree model
- Complete notebook section: Compare metrics
- Complete notebook section: Interpret importance
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


### Deep Dive: Tree Models + Feature Importance (What To Trust)

Tree models can capture non-linear relationships and interactions that linear models miss.
But they are easier to overfit and harder to interpret.

#### Key Terms (defined)
- **Decision tree**: splits data by thresholds on features.
- **Random forest**: averages many trees trained on bootstrapped samples.
- **Gradient boosting**: sequentially adds trees to correct errors.
- **Overfitting**: learning noise; appears as high train performance, low test performance.
- **Gini importance**: impurity-based importance from trees (can be biased).
- **Permutation importance**: importance from shuffling a feature and measuring performance drop.

#### Why tree feature importances can mislead
- Impurity-based importance can favor:
  - high-cardinality features,
  - noisy continuous features,
  - correlated features (importance can be split or concentrated unpredictably).

#### Python demo: impurity vs permutation importance
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
n = 800

# Two correlated features + one noise feature
x1 = rng.normal(size=n)
x2 = x1 * 0.9 + rng.normal(scale=0.5, size=n)
x3 = rng.normal(size=n)
X = np.column_stack([x1, x2, x3])
p = 1 / (1 + np.exp(-(0.5 + 1.0*x1)))
y = rng.binomial(1, p)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)
rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_tr, y_tr)
print('AUC:', roc_auc_score(y_te, rf.predict_proba(X_te)[:,1]))
print('gini importances:', rf.feature_importances_)

pi = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=0, scoring='roc_auc')
print('perm importances:', pi.importances_mean)
```

#### Practical interpretation
- Treat importance as "usefulness for prediction", not causal influence.
- Compare importances across eras (walk-forward) to see if drivers change.


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
