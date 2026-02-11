# Guide: 02_clustering_macro_regimes

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/04_unsupervised/02_clustering_macro_regimes.ipynb`.

> **Note:** Clustering is a descriptive/exploratory tool. It is useful for data exploration in any setting (health data, macro data, survey data), but it does not establish causal relationships. In health economics, clustering can help segment patient populations or identify regional patterns, but the same caveats apply: clusters are hypotheses, not causal conclusions.

### Key Terms (defined)
- **Clustering**: grouping observations into categories based on similarity in feature space.
- **K-means**: the most common algorithm — assigns observations to $K$ clusters to minimize within-cluster distances.
- **Standardization**: scaling features to mean 0 and std 1 before clustering — essential because distance-based methods are sensitive to units.
- **Silhouette score**: a measure of how well-separated clusters are (ranges from $-1$ to $+1$; higher is better).
- **Inertia**: within-cluster sum of squares — decreases as $K$ increases (used for elbow plots).

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Standardize all features before clustering.
- Fit K-means for $K = 2, 3, 4, 5, 6$ and plot inertia and silhouette scores.
- Choose $K$ with justification (elbow, silhouette, interpretability).
- Label clusters by inspecting the mean of each indicator within each cluster.
- Compare cluster assignments to known recession dates — do they align?
- Re-fit with a different random seed and check assignment stability.

### Alternative Example (Not the Notebook Solution)
```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

rng = np.random.default_rng(42)
# Simulated features: unemployment change, GDP growth, inflation
X = rng.normal(size=(200, 3))
X[:50] += [1.5, -2, 0.5]  # simulate a "recession" cluster

X_scaled = StandardScaler().fit_transform(X)

# Fit K-means for multiple K
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_)
    print(f"K={k}: inertia={km.inertia_:.1f}, silhouette={sil:.3f}")
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### What clustering does

Clustering partitions observations into groups such that observations within a group are "similar" and observations across groups are "different." It is entirely descriptive — no target variable, no prediction.

### K-means objective

K-means chooses cluster centers $\mu_1, \dots, \mu_K$ and assignments $c_t \in \{1, \dots, K\}$ to minimize:

$$
\sum_{t=1}^{n} \|x_t - \mu_{c_t}\|_2^2.
$$

The algorithm alternates between (1) assigning each point to its nearest center and (2) recomputing centers as cluster means. It converges but may find a local minimum — always run with multiple random seeds (`n_init=10` or more).

### Why standardization is essential

K-means uses Euclidean distance. If one feature has range [0, 1000] and another has range [0, 1], the first feature dominates distance calculations regardless of its importance. Standardizing to zero mean and unit variance puts all features on equal footing:

$$
\tilde{x}_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}.
$$

### Choosing $K$

There is no "correct" $K$ — it depends on interpretation and purpose:

- **Elbow method:** Plot inertia vs $K$; look for a "bend" where additional clusters give diminishing returns.
- **Silhouette score:** Measures cluster separation. Higher is better. Values above 0.5 suggest reasonable structure; below 0.25 is weak.
- **Domain knowledge:** In macro, $K = 2$ or $3$ often maps to expansion/recession/crisis. In health data, think about clinical subgroups.

### Diagnostics checklist

1. **Standardization** — confirm features have mean $\approx 0$ and std $\approx 1$ after scaling.
2. **Stability across seeds** — do assignments change significantly? If so, the structure is weak.
3. **Cluster sizes** — very small clusters may be outliers, not regimes.
4. **Temporal coherence** — in time-series settings, do clusters appear in contiguous blocks or random scatter? Both can be informative.
5. **Sensitivity to $K$** — do conclusions persist across reasonable $K$ values?

### What this does NOT mean
- Clusters are not proof of causal regimes — they are data-driven groupings.
- K-means assumes roughly spherical clusters of similar size; violations (elongated or imbalanced clusters) can mislead.
- Over-interpreting cluster labels is a common trap — always validate externally.

#### Exercises

- [ ] Fit K-means for $K = 2 \dots 6$; plot inertia and silhouette. Choose $K$ with justification.
- [ ] Compare cluster periods to recession shading; interpret alignment and mismatches.
- [ ] Re-fit with a different seed and measure assignment stability (e.g., adjusted Rand index).

### Project Code Map
- `src/features.py`: feature engineering helpers
- `data/sample/panel_monthly_sample.csv`: offline dataset

### Common Mistakes
- Forgetting to standardize — the biggest-unit variable will dominate.
- Treating cluster labels as causal regime identifications.
- Using too many clusters and overfitting noise.
- Not checking stability across random seeds.

<a id="summary"></a>
## Summary + Suggested Readings

Clustering is a useful exploratory tool for finding structure in multivariate data. Standardize first, check multiple $K$ values, validate externally, and never over-interpret.

Suggested readings:
- James et al. (2021): *Introduction to Statistical Learning* — Ch. 12 (clustering)
- scikit-learn docs: KMeans, silhouette_score
