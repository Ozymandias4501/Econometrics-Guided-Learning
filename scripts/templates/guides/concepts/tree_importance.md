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
