# Guide: 03_anomaly_detection

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/04_unsupervised/03_anomaly_detection.ipynb`.

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
- Complete notebook section: Fit detector
- Complete notebook section: Inspect anomalies
- Complete notebook section: Compare to recessions
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


### Deep Dive: Anomaly Detection (Crisis Periods)

Anomaly detection flags observations that look unusual relative to the bulk of the data.
In macro, anomalies often correspond to crises (e.g., 2008, 2020), but not always.

#### Key Terms (defined)
- **Outlier**: an observation far from typical behavior.
- **Anomaly score**: a numeric measure of "unusualness".
- **Isolation Forest**: detects anomalies by how easily points are isolated by random splits.
- **Contamination**: expected fraction of anomalies (hyperparameter).

#### Python demo: Isolation Forest intuition
```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X_normal = rng.normal(size=(300, 3))
X_anom = rng.normal(loc=6.0, scale=1.0, size=(10, 3))
X = np.vstack([X_normal, X_anom])
X = StandardScaler().fit_transform(X)

iso = IsolationForest(contamination=0.05, random_state=0).fit(X)
scores = -iso.score_samples(X)  # higher = more anomalous
scores[-10:].round(3)
```

#### Interpreting anomalies
- An anomaly is not automatically "bad" or "recession".
- It means the pattern of indicators is rare.
- Compare anomaly flags to your recession label to see overlaps and differences.

#### Debug tips
- Always standardize before distance/forest methods.
- Sensitivity-check the contamination parameter.
- Inspect which features contribute (e.g., via z-scores) for flagged periods.


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
