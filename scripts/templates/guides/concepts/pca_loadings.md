### Deep Dive: PCA and Loadings (Turning Many Indicators Into Factors)

> **Definition:** **Principal Component Analysis (PCA)** finds orthogonal directions (components) that explain the most variance in the data.

> **Definition:** A **component** is a linear combination of original variables.

> **Definition:** **Loadings** are the weights that map original variables into a component.

#### Why PCA is useful in macro
Macro indicators are often correlated (growth, inflation, rates). PCA can:
- compress many indicators into a few factors
- reduce multicollinearity
- create interpretable latent "macro factors"

#### Standardization is not optional
> **Definition:** **Standardization** rescales features to mean 0 and std 1.

Distance/variance depends on units. Without standardization, PCA mostly learns the biggest-unit variable.

#### The math (high level)
Let $X$ be a centered (and typically standardized) data matrix.
PCA finds vectors $w_k$ that maximize:

$$
\max_{w_k} \; \mathrm{Var}(X w_k) \quad \text{s.t. } ||w_k||_2 = 1 \text{ and } w_k \perp w_{k'}
$$

This leads to eigenvectors of the covariance (or correlation) matrix.

#### Interpreting loadings
A component score is:

$$
\text{factor}_k = X w_k
$$

- Large positive loading means the factor increases when that variable increases.
- Large negative loading means the factor increases when that variable decreases.

#### Python demo (commented)
```python
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# X: DataFrame of macro features
# X = ...

# 1) Standardize
# scaler = StandardScaler().fit(X)
# X_scaled = scaler.transform(X)

# 2) Fit PCA
# pca = PCA(n_components=3).fit(X_scaled)

# 3) Loadings: rows are components, columns are original features
# loadings = pd.DataFrame(
#     pca.components_,
#     columns=X.columns,
#     index=[f'PC{i+1}' for i in range(pca.n_components_)],
# )

# 4) Explained variance
# print(pca.explained_variance_ratio_)
```

#### Interpretation playbook
1. Inspect explained variance ratio to decide how many PCs matter.
2. For each PC, list the top positive and negative loadings.
3. Give the factor a name in economic terms.
4. Plot the factor through time and compare to known events.
