### Deep Dive: Bias/Variance and Overfitting (Why Test Performance Matters)

> **Definition:** **Overfitting** happens when a model learns patterns specific to the training data noise rather than general structure.

> **Definition:** **Bias** is systematic error from an overly simple model.

> **Definition:** **Variance** is error from sensitivity to the particular training sample.

High-level relationship:
- simple models: higher bias, lower variance
- flexible models: lower bias, higher variance

#### How it looks in practice
- Training error decreases as model complexity increases.
- Test error often decreases at first, then increases (the "U-shape" idea).

#### Python demo: simple vs flexible model (commented)
```python
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)

# Synthetic data: y is a smooth function of x + noise
n = 300
x = rng.uniform(-3, 3, size=n)
y = np.sin(x) + 0.3 * rng.normal(size=n)

# Feature matrix
X = x.reshape(-1, 1)

# Time-like split (just to keep the habit)
split = int(n * 0.8)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

lin = LinearRegression().fit(X_tr, y_tr)
tree = DecisionTreeRegressor(random_state=0).fit(X_tr, y_tr)

rmse_lin_tr = mean_squared_error(y_tr, lin.predict(X_tr), squared=False)
rmse_lin_te = mean_squared_error(y_te, lin.predict(X_te), squared=False)
rmse_tree_tr = mean_squared_error(y_tr, tree.predict(X_tr), squared=False)
rmse_tree_te = mean_squared_error(y_te, tree.predict(X_te), squared=False)

print({'lin_train': rmse_lin_tr, 'lin_test': rmse_lin_te})
print({'tree_train': rmse_tree_tr, 'tree_test': rmse_tree_te})
```

#### Interpretation
- A model with extremely low training error but high test error is overfitting.
- Regularization and simpler models can reduce variance.
