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


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

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

### Core Unsupervised Learning: describe structure before predicting

Unsupervised methods are tools for *description* rather than *prediction*:
they help you understand structure in the indicators without a target label.

#### 1) Intuition (plain English)

In macro and finance, variables move together for many reasons:
- common shocks,
- regimes (expansions vs recessions),
- measurement changes.

Unsupervised methods help answer:
- “Are there a few latent factors driving many series?” (PCA)
- “Do observations cluster into regimes?” (clustering)
- “Which periods look unusual?” (anomaly detection)

These outputs are best treated as hypotheses you can investigate, not as causal explanations.

#### 2) Notation + setup (define symbols)

Let $X$ be an $n \\times p$ feature matrix:
- rows: time periods or entities,
- columns: standardized indicators.

Standardization is often essential:

$$
\\tilde x_{ij} = \\frac{x_{ij} - \\bar x_j}{s_j}
$$

**What each term means**
- $\\bar x_j$: mean of feature $j$,
- $s_j$: standard deviation of feature $j$,
- standardization prevents high-variance features from dominating.

#### 3) Assumptions (and what can go wrong)

Unsupervised methods assume:
- features are comparable after scaling,
- distance/variance summaries reflect meaningful structure.

Common failure modes:
- forgetting to standardize,
- including strong trends (nonstationarity) so “clusters” become time periods,
- over-interpreting factors as “the economy’s true drivers.”

#### 4) Estimation mechanics (high level)

- **PCA:** finds orthogonal directions that explain maximal variance.
- **Clustering (k-means):** partitions observations to minimize within-cluster distances.
- **Anomaly detection:** flags observations far from the “typical” pattern.

#### 5) Diagnostics + robustness (minimum set)

1) **Standardization check**
- confirm features have mean ~0 and std ~1 after scaling.

2) **Stability**
- do PCA loadings or cluster assignments change a lot if you change the sample period?

3) **Interpretability**
- can you label factors/clusters with economic meaning using external context?

4) **Sensitivity**
- try different numbers of components/clusters; do conclusions persist?

#### 6) Interpretation + reporting

Report:
- preprocessing (standardization, transformations),
- chosen hyperparameters (k, number of PCs),
- stability checks.

**What this does NOT mean**
- Clusters are not proof of causal regimes.
- PCA components are not “true” economic factors; they are variance summaries.

#### Exercises

- [ ] Standardize a feature matrix and verify means/stds.
- [ ] Fit PCA and interpret the top component using loadings.
- [ ] Cluster the dataset and compare cluster periods to known recessions.
- [ ] Try two different k values and explain how clustering changes.

### Deep Dive: Clustering regimes — grouping time periods into “states of the economy”

Clustering is a descriptive tool for finding patterns/regimes in multivariate macro data.

#### 1) Intuition (plain English)

Instead of predicting recession directly, you can ask:
- “Do the indicators naturally cluster into a few recurring patterns?”

These clusters can correspond to:
- expansions,
- recessions,
- high-inflation regimes,
- crisis periods.

But clusters are hypotheses, not causal regime proofs.

#### 2) Notation + setup (define symbols)

Let $x_t \\in \\mathbb{R}^p$ be the feature vector at time $t$.
K-means clustering chooses:
- cluster centers $\\mu_1,\\dots,\\mu_K$,
- assignments $c_t \\in \\{1,\\dots,K\\}$,
to minimize:

$$
\\sum_{t=1}^{n} \\|x_t - \\mu_{c_t}\\|_2^2.
$$

**What each term means**
- $K$: number of clusters (chosen by you).
- distance: usually Euclidean → scaling matters.

#### 3) Assumptions (and pitfalls)

Clustering assumes:
- a distance metric that reflects meaningful similarity,
- stable scaling across features (standardize),
- roughly “spherical” clusters for k-means.

Pitfalls:
- nonstationarity can dominate distance (clusters become time periods),
- correlated features can distort distances,
- k-means can be sensitive to initialization.

#### 4) Estimation mechanics

Practical steps:
1) standardize features,
2) choose $K$ candidates (e.g., 2–6),
3) fit clustering with multiple random seeds,
4) label clusters by inspecting indicator means and time periods.

#### 5) Inference: use stability, not p-values

Assess uncertainty by:
- stability across seeds,
- stability across subperiods,
- sensitivity to $K$.

#### 6) Diagnostics + robustness (minimum set)

1) **Silhouette / inertia trends**
- do you see diminishing returns as $K$ increases?

2) **Cluster size sanity**
- tiny clusters may be outliers, not regimes.

3) **Temporal coherence**
- do clusters appear in contiguous time blocks or random scatter? (both can be informative)

4) **Stability**
- re-fit with different seeds/subsamples and compare assignments.

#### 7) Interpretation + reporting

Report:
- preprocessing (standardization, transformations),
- chosen $K$ and how selected,
- stability evidence,
- cluster summaries (means of indicators).

#### Exercises

- [ ] Fit k-means for K=2..6 and plot inertia/silhouette; choose K with justification.
- [ ] Compare cluster assignments to recession shading and interpret alignment/mismatches.
- [ ] Re-fit with a different seed and measure assignment stability.

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
