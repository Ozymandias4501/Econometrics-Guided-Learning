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


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

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

Let $X$ be an $n \times p$ feature matrix:
- rows: time periods or entities,
- columns: standardized indicators.

Standardization is often essential:

$$
\tilde x_{ij} = \frac{x_{ij} - \bar x_j}{s_j}
$$

**What each term means**
- $\bar x_j$: mean of feature $j$,
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
- **Practical check:** fit PCA separately on the first half and second half of the sample. Compute the correlation between the PC1 loading vectors from each half. If the correlation exceeds ~0.8, the factor structure is reasonably stable. If it falls below ~0.5, the underlying relationships have shifted materially and you should investigate whether a structural break (e.g., the Great Moderation, COVID) is driving the change.

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

### Deep Dive: PCA and loadings — latent factors in economic indicators

Principal component analysis (PCA) is a standard way to summarize many correlated indicators with a few latent factors.

#### 1) Intuition (plain English)

Many macro indicators move together because they share common shocks (the “business cycle factor”).
PCA finds directions in feature space that explain the most variance.

Use PCA to:
- reduce dimensionality,
- build factor-like summaries,
- visualize structure.

Treat PCA as descriptive: it finds variance directions, not causal drivers.

#### 2) Notation + setup (define symbols)

Let $X$ be an $n \times p$ matrix of features:
- rows: observations (time periods),
- columns: indicators (standardized).

Standardization:
$$
\tilde X_{ij} = \frac{X_{ij} - \bar X_j}{s_j}.
$$

Sample covariance matrix (for standardized data):
$$
S = \frac{1}{n}\tilde X'\tilde X.
$$

PCA finds eigenvectors of $S$:
$$
S w_k = \lambda_k w_k,
$$
with eigenvalues $\lambda_1 \ge \lambda_2 \ge \cdots$.

PC scores:
$$
z_{ik} = \tilde x_i' w_k,
$$
where $\tilde x_i$ is row $i$ of $\tilde X$.

**What each term means**
- $w_k$: loading vector for component $k$ (how indicators combine).
- $\lambda_k$: variance explained by component $k$.
- $z_{ik}$: component score (latent factor value for observation $i$).

#### 3) Assumptions (and what can go wrong)

PCA assumes:
- scaling makes indicators comparable (standardize!),
- linear combinations capture meaningful structure.

Common pitfalls:
- forgetting to standardize,
- including strong trends (nonstationarity) so the first PC becomes “time”,
- over-interpreting components as causal factors.

#### 4) Estimation mechanics (SVD perspective)

Most implementations use SVD:
$$
\tilde X = U \Sigma V'.
$$

Then:
- loadings are columns of $V$,
- explained variance relates to singular values in $\Sigma$.

#### 5) Inference: uncertainty is about stability, not p-values

PCA does not come with simple p-values for “importance.”
Instead assess:
- stability across subperiods,
- sensitivity to feature sets,
- interpretability of loadings.

#### 6) Diagnostics + robustness (minimum set)

1) **Explained variance (scree plot)**
- how many components explain most variance?

2) **Loading interpretability**
- do top loadings match a coherent economic story?

3) **Stability**
- re-fit PCA on different time windows; do loadings and scores persist?

4) **Reconstruction check**
- can a few PCs reconstruct key series reasonably?

#### 7) Interpretation + reporting

Report:
- whether features were standardized,
- explained variance ratios,
- top loadings for the first few PCs,
- and stability checks.

**What this does NOT mean**
- PCA does not identify causal mechanisms.
- “First PC” is not automatically “the business cycle” unless evidence supports it.

#### Exercises

- [ ] Fit PCA on standardized macro features and plot the scree plot.
- [ ] Interpret the first PC using its top 5 loadings.
- [ ] Re-fit PCA on two subperiods and compare loadings; write 5 sentences on stability.

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
