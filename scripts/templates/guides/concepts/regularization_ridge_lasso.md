### Deep Dive: Regularization (Ridge vs Lasso)

Regularization adds a penalty to reduce overfitting and stabilize coefficients.

> **Definition:** **Regularization** modifies the training objective to discourage overly complex models (often large coefficients).

#### The bias/variance tradeoff (why regularization can help)
- OLS can have low bias but high variance (coefficients jump around across samples), especially with correlated predictors.
- Regularization intentionally introduces some bias to reduce variance.

#### Objectives (math)
Let $y$ be your target and $X$ your feature matrix.

OLS minimizes:

$$
\min_{\beta} \; ||y - X\beta||_2^2
$$

Ridge (L2) minimizes:

$$
\min_{\beta} \; ||y - X\beta||_2^2 + \alpha ||\beta||_2^2
$$

Lasso (L1) minimizes:

$$
\min_{\beta} \; ||y - X\beta||_2^2 + \alpha ||\beta||_1
$$

- $||\beta||_2^2 = \sum_j \beta_j^2$
- $||\beta||_1 = \sum_j |\beta_j|$

#### What changes and what does not
- As $\alpha$ increases, coefficients shrink.
- Ridge typically shrinks all coefficients toward 0.
- Lasso can drive some coefficients exactly to 0 (feature selection).

#### Why standardization matters
> **Definition:** **Standardization** rescales features to mean 0 and standard deviation 1.

Regularization penalties depend on coefficient size. If features are on different scales, the penalty is applied unevenly.
Always use `StandardScaler` before ridge/lasso.

#### Python demo: ridge vs lasso (commented)
```python
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(0)

# Create correlated predictors
n = 300
x1 = rng.normal(size=n)
x2 = 0.98 * x1 + rng.normal(scale=0.2, size=n)
X = np.column_stack([x1, x2])

# Target depends on x1

y = 1.0 + 2.0 * x1 + rng.normal(scale=1.0, size=n)

# OLS (no penalty)
ols = LinearRegression().fit(X, y)

# Ridge + scaling
ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=5.0)),
]).fit(X, y)

# Lasso + scaling
lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso(alpha=0.1, max_iter=10000)),
]).fit(X, y)

print('ols  coef:', ols.coef_)
print('ridge coef:', ridge.named_steps['model'].coef_)
print('lasso coef:', lasso.named_steps['model'].coef_)
```

#### Interpretation cautions
- Regularized coefficients are biased by design.
- They can be excellent for prediction and stability.
- Do not interpret lasso-selected features as "the true causes".
- In correlated macro data, lasso may pick one variable from a group and ignore equally good substitutes.
