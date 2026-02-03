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
