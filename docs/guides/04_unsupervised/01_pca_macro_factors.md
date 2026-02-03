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

### PCA Intuition
- PCA finds directions that explain maximum variance.
- Components are orthogonal (uncorrelated by construction).
- Loadings help you interpret what each factor represents.

### Clustering Intuition
- k-means finds k centroids and assigns each point to the closest.
- Choosing k is a modeling decision; use elbow plots and interpretability.


### Deep Dive: PCA and Loadings (Turning Many Indicators Into Factors)

**PCA** finds orthogonal directions that explain maximum variance.

**Loadings** tell you which original variables contribute to a component.
A component can often be interpreted as a macro "factor" (e.g., growth, inflation, rates).

**Critical step: standardization**
- Without standardization, PCA mostly learns the biggest-unit variable.

**Python demo**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# X_scaled = StandardScaler().fit_transform(X)
# pca = PCA(n_components=3).fit(X_scaled)
# loadings = pca.components_  # rows are components
```

**Interpretation playbook**
- Look at the largest positive/negative loadings.
- Give the factor a name.
- Check if that factor spikes during known historical episodes.


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
