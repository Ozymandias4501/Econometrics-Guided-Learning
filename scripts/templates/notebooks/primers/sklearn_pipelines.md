## Primer: sklearn pipelines (how to avoid preprocessing leakage)

Pipelines prevent a common mistake: fitting preprocessing (scaling, imputation) using information from the test set.

### Why pipelines exist (in one sentence)

> A `Pipeline` ensures that transformations are fit on training data only, then applied to test data.

### The key APIs

- `fit(X, y)`: learn parameters (scaler mean/std, model weights) from training.
- `transform(X)`: apply learned transform to new data.
- `fit_transform(X, y)`: convenience for training data only.

### Minimal pattern (classification)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

clf = Pipeline([
  ("scaler", StandardScaler()),
  ("model", LogisticRegression(max_iter=5000)),
])

# clf.fit(X_train, y_train)
# y_prob = clf.predict_proba(X_test)[:, 1]
```

**Expected output / sanity check**
- you never call `scaler.fit` on the full dataset
- you split by time first, then fit the pipeline on train

### Mini demo: the leakage you’re avoiding (toy)

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X_train = rng.normal(loc=0.0, scale=1.0, size=(100, 1))
X_test  = rng.normal(loc=2.0, scale=1.0, size=(25, 1))

# WRONG: fit on train+test (leaks the future)
sc_wrong = StandardScaler().fit(np.vstack([X_train, X_test]))
X_test_wrong = sc_wrong.transform(X_test)

# RIGHT: fit on train only
sc_right = StandardScaler().fit(X_train)
X_test_right = sc_right.transform(X_test)

print("test mean after wrong scaling:", float(X_test_wrong.mean()))
print("test mean after right scaling:", float(X_test_right.mean()))
```

### Common pitfalls

- Splitting after preprocessing (leakage).
- Using random splits on time-indexed data (temporal leakage).
- Forgetting `ColumnTransformer` for mixed numeric/categorical columns (if needed).
