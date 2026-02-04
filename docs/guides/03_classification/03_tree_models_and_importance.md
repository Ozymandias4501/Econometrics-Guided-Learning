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

### Core Classification: Probabilities, Metrics, and Thresholds

In this project, classification is about predicting recession risk as a probability.

#### Logistic regression mechanics
Logistic regression models probabilities via log-odds:

$$
\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k
$$

Then:

$$
 p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots)}}
$$

#### Metrics you should treat as standard
- ROC-AUC: ranking quality across thresholds
- PR-AUC: often more informative when positives are rare
- Brier score (or log loss): probability quality

#### Thresholding is a decision rule
> **Definition:** A **threshold** converts probabilities into labels.

Default 0.5 is rarely optimal for imbalanced, cost-sensitive problems.
Pick thresholds based on:
- decision costs
- desired recall/precision tradeoff
- calibration quality

### Deep Dive: Tree Models + Feature Importance (What To Trust)

Tree models can capture non-linear relationships and interactions.
They can also overfit and they can be misinterpreted.

#### Key terms (defined)
> **Definition:** A **decision tree** predicts by splitting data using feature thresholds.

> **Definition:** A **random forest** averages many trees trained on bootstrapped samples.

> **Definition:** **Impurity-based importance** ("Gini importance") measures how much a feature reduces impurity across splits.

> **Definition:** **Permutation importance** measures how much performance drops when you shuffle a feature.

#### Why impurity-based importance can mislead
Impurity-based importance can be biased toward:
- features with many possible split points
- correlated features (importance can be split unpredictably)

Permutation importance is often more reliable for "usefulness" but still has caveats:
- correlated features can share importance
- shuffling breaks correlation structure

#### Python demo: impurity vs permutation importance (commented)
```python
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)

# Two correlated features + one noise feature
n = 800
x1 = rng.normal(size=n)
x2 = 0.9 * x1 + rng.normal(scale=0.5, size=n)
x3 = rng.normal(size=n)
X = np.column_stack([x1, x2, x3])

# Outcome depends mostly on x1
p = 1 / (1 + np.exp(-(0.5 + 1.0 * x1)))
y = rng.binomial(1, p)

# Split (random here only because this is toy IID data)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_tr, y_tr)
auc = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
print('AUC:', auc)
print('gini importances:', rf.feature_importances_)

pi = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=0, scoring='roc_auc')
print('perm importances:', pi.importances_mean)
```

#### Interpretation rule
Treat feature importance as "useful for prediction".
Do not treat it as causal influence.

### Project Code Map
- `src/evaluation.py`: classification metrics (ROC-AUC, PR-AUC, Brier, precision/recall)
- `scripts/train_recession.py`: training script that writes artifacts
- `scripts/predict_recession.py`: prediction script that loads artifacts
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

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
