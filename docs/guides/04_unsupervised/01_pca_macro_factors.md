# Guide: 01_pca_macro_factors

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/04_unsupervised/01_pca_macro_factors.ipynb`.

This unsupervised module explores macro structure: factors, regimes, and anomalies.

### Key Terms (defined)
- **Unsupervised learning**: learning patterns without a labeled target.
- **PCA**: rotates correlated features into uncorrelated components (factors).
- **Loadings**: how strongly each original variable contributes to a component.
- **Clustering**: grouping similar periods into regimes.
- **Anomaly detection**: flagging unusual points (often crisis periods).


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Standardize
- Complete notebook section: Fit PCA
- Complete notebook section: Interpret loadings
- Standardize features before PCA/clustering (units matter).
- Interpret components/clusters economically (give them names).
- Compare regimes/anomalies to your recession labels (do they align?).

### Alternative Example (Not the Notebook Solution)
```python
# Toy PCA (not the notebook data):
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = np.random.randn(200, 5)
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(X)
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Core Unsupervised Learning: Describe Structure Before Predicting

Unsupervised methods help you understand structure in the indicators.
They do not require a target label.

#### Standardization matters
Many unsupervised methods rely on distances or variances.
If one variable has larger units, it dominates.

> **Definition:** **Standardization** rescales each feature to mean 0 and standard deviation 1.

#### Interpretation stance
Treat unsupervised outputs as:
- descriptions (factors, regimes, anomalies)
- hypotheses you can investigate

Avoid treating them as causal explanations.

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

#### Project touchpoints
- PCA is introduced in the unsupervised notebooks using `sklearn.decomposition.PCA`.
- You will typically start from `data/processed/panel_monthly.csv` (or `data/sample/panel_monthly_sample.csv`).

### Project Code Map
- `src/features.py`: feature engineering helpers (standardization happens in notebooks)
- `data/sample/panel_monthly_sample.csv`: offline dataset for experimentation
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Forgetting to standardize (PCA will just pick the biggest-unit variable).
- Interpreting a cluster label as a causal regime without validation.
- Using too many components/clusters and overfitting noise.

<a id="summary"></a>
## Summary + Suggested Readings

Unsupervised tools help you understand macro data structure even before prediction.
They are especially useful for detecting regime shifts and crisis periods.


Suggested readings:
- Jolliffe: Principal Component Analysis
- scikit-learn docs: PCA, clustering, anomaly detection
