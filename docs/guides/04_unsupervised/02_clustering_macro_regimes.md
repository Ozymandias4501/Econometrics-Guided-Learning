# Guide: 02_clustering_macro_regimes

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/04_unsupervised/02_clustering_macro_regimes.ipynb`.

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
- Complete notebook section: Clustering
- Complete notebook section: Choose k
- Complete notebook section: Relate to recessions
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

### Deep Dive: Clustering as Regime Discovery

Clustering groups time periods with similar indicator patterns.
You can treat clusters as candidate "macro regimes" and compare them to recession labels.

#### Key terms (defined)
> **Definition:** **Clustering** groups observations so points in the same cluster are similar.

> **Definition:** **k-means** finds k centroids and assigns each point to the closest centroid.

> **Definition:** **Inertia** is the k-means objective (within-cluster sum of squares). It always decreases with k.

> **Definition:** The **silhouette score** measures separation between clusters (higher is better).

#### k-means objective (math)
k-means solves:

$$
\min_{C_1,\ldots,C_k} \sum_{j=1}^k \sum_{i \in C_j} ||x_i - \mu_j||^2
$$

- $\mu_j$ is the centroid of cluster j.
- Distance depends on feature scale, so standardization matters.

#### Python demo: k-means + silhouette (commented)
```python
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# Toy feature matrix
X = rng.normal(size=(200, 4))

# Standardize before k-means
X_scaled = StandardScaler().fit_transform(X)

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_)
    print(k, 'inertia', km.inertia_, 'sil', sil)
```

#### Interpretation playbook
1. Standardize features.
2. Choose k using elbow + silhouette + interpretability.
3. Inspect cluster centroids in original units (undo scaling) for meaning.
4. Compare cluster assignments to recession labels.

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
