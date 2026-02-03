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

### PCA Intuition
- PCA finds directions that explain maximum variance.
- Components are orthogonal (uncorrelated by construction).
- Loadings help you interpret what each factor represents.

### Clustering Intuition
- k-means finds k centroids and assigns each point to the closest.
- Choosing k is a modeling decision; use elbow plots and interpretability.


### Deep Dive: Clustering as Regime Discovery

Clustering groups time periods with similar indicator patterns.
You can treat clusters as candidate "macro regimes" and then compare them to recessions.

#### Key Terms (defined)
- **k-means**: clustering method that minimizes within-cluster squared distances to centroids.
- **Inertia**: k-means objective value (lower is better, always decreases with k).
- **Silhouette score**: measures separation between clusters (higher is better).
- **Standardization**: required because distance depends on scale.

#### Choosing k
- There is no single "correct" k.
- Use elbow plots (inertia), silhouette scores, and interpretability.

#### Python demo: k-means + silhouette
```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))
X = StandardScaler().fit_transform(X)

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    sil = silhouette_score(X, km.labels_)
    print(k, 'inertia', km.inertia_, 'sil', sil)
```

#### Interpretation playbook
1. Compute cluster centroids in original units (undo scaling) for interpretability.
2. Name clusters in economic language ("high inflation/high rates", etc.).
3. Check whether certain clusters are recession-heavy.


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
