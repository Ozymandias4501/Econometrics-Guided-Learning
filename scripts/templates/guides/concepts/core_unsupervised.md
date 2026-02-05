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
