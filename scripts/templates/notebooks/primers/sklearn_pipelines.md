## Primer: sklearn Pipelines (How To Avoid Preprocessing Leakage)

### Why pipelines exist
A common ML mistake is fitting preprocessing (scalers, imputers) on the full dataset.
That leaks information from the test set into training.

A `Pipeline` enforces the correct order:
- fit preprocessing on training only
- apply preprocessing to test
- fit model on training only

### Key API concepts
- `fit(X, y)`: learn parameters from data (e.g., scaler means/standard deviations, model weights).
- `transform(X)`: apply learned parameters to new data (e.g., scale).
- `fit_transform(X, y)`: convenience that does both on the same data.

If you do `scaler.fit(X_all)` before splitting, you leaked test-set information.

### Example pattern
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=5000)),
])

# clf.fit(X_train, y_train)
# y_prob = clf.predict_proba(X_test)[:, 1]
```

### Mini demo: the leakage you're avoiding (toy example)
```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Pretend the last 20% of data comes from a different era with a different mean
rng = np.random.default_rng(0)
X_train = rng.normal(loc=0.0, scale=1.0, size=(100, 1))
X_test  = rng.normal(loc=2.0, scale=1.0, size=(25, 1))

# WRONG: fit scaler on train+test (leaks the future)
sc_wrong = StandardScaler().fit(np.vstack([X_train, X_test]))
X_test_wrong = sc_wrong.transform(X_test)

# RIGHT: fit scaler on train only
sc_right = StandardScaler().fit(X_train)
X_test_right = sc_right.transform(X_test)

print("test mean after wrong scaling:", float(X_test_wrong.mean()))
print("test mean after right scaling:", float(X_test_right.mean()))
```

### What to remember
- Always split by time first.
- Then fit the pipeline on train.
- Then evaluate on test.

If you need different preprocessing for different columns, look into:
- `sklearn.compose.ColumnTransformer`
