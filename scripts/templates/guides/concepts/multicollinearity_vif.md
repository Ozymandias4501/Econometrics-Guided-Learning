### Deep Dive: Multicollinearity and VIF (Why Coefficients Become Unstable)

> **Definition:** **Multicollinearity** means two or more predictors contain overlapping information (they are highly correlated).

Multicollinearity is especially common in macro data because many indicators move together (business cycle, policy regimes).

#### What multicollinearity does (and does not do)
- It often does **not** hurt prediction much.
- It **does** make individual coefficients unstable.
- It inflates standard errors, making p-values fragile.

#### Regression notation
We write a linear regression as:

$$
\mathbf{y} = \mathbf{X}\beta + \varepsilon
$$

- $\mathbf{y}$ is an $n \times 1$ vector of outcomes.
- $\mathbf{X}$ is an $n \times p$ matrix of predictors.
- $\beta$ is a $p \times 1$ vector of coefficients.
- $\varepsilon$ is an $n \times 1$ vector of errors.

When columns of $\mathbf{X}$ are nearly linearly dependent, $(\mathbf{X}'\mathbf{X})$ is close to singular, and coefficient estimates become unstable.

#### VIF (Variance Inflation Factor)
> **Definition:** The **variance inflation factor** for feature $j$ is:

$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^2}
$$

Where $R_j^2$ is from regressing $x_j$ on all the other predictors.

Interpretation:
- If $R_j^2$ is high, $x_j$ is well-explained by other predictors.
- Then $\mathrm{VIF}_j$ is high, meaning the variance of $\hat\beta_j$ is inflated.

Rules of thumb (not laws):
- VIF > 5 suggests notable collinearity.
- VIF > 10 suggests serious collinearity.

#### Python demo: correlated predictors -> unstable coefficients (commented)
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.econometrics import vif_table

rng = np.random.default_rng(0)

# Create two highly correlated predictors
n = 600
x1 = rng.normal(size=n)
x2 = 0.95 * x1 + rng.normal(scale=0.2, size=n)  # mostly the same information

# True outcome depends only on x1
# In the presence of collinearity, the model may "split" credit unpredictably.
y = 1.0 + 2.0 * x1 + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

# VIF quantifies how redundant each predictor is
print(vif_table(df, ['x1', 'x2']))

# Fit OLS with both predictors
X = sm.add_constant(df[['x1', 'x2']])
res = sm.OLS(df['y'], X).fit()

print('params:', res.params.to_dict())
print('std err:', res.bse.to_dict())
```

#### What to do about multicollinearity
> **Definition:** **Regularization** (like ridge) adds a penalty that stabilizes coefficients.

Practical options:
- Drop one variable from a correlated group.
- Combine variables (domain composite, PCA/factors).
- Use ridge regression to stabilize.
- Focus on prediction rather than coefficient interpretation.

#### Economics interpretation warning
In macro data, "holding other indicators fixed" can be an unrealistic counterfactual.
If two indicators are tightly linked, the idea of changing one while freezing the other is not economically meaningful.
Treat coefficients as conditional correlations unless you have a causal design.
