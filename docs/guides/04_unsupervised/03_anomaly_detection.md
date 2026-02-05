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


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

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

### Deep Dive: Anomaly detection — finding “unusual” macro periods

Anomaly detection flags observations that look unusual relative to typical patterns.

#### 1) Intuition (plain English)

Crises (2008, 2020) look different in multivariate indicator space.
Anomaly detection can highlight these periods without a recession label.

Use it to:
- detect outliers,
- generate hypotheses,
- build monitoring dashboards.

#### 2) Notation + setup (define symbols)

Let $x_t$ be the feature vector at time $t$.
An anomaly detector produces a score:
$$
s_t = s(x_t),
$$
where larger (or smaller, depending on convention) means “more anomalous.”

Simple baseline: z-score on one feature:
$$
z_t = \\frac{x_t - \\bar x}{s}.
$$

Multivariate baselines:
- distance from mean (Mahalanobis distance),
- isolation forest score,
- reconstruction error from PCA.

#### 3) Assumptions

Anomaly detection assumes:
- features are scaled comparably,
- “typical” behavior exists and is represented in the data.

Structural breaks can shift what is “normal,” so anomaly thresholds should be treated as time-varying in real monitoring.

#### 4) Estimation mechanics (high level)

Common approaches:
- **PCA reconstruction:** anomalies have large reconstruction error using a few PCs.
- **Isolation forest:** anomalies are easier to isolate with random splits.
- **Distance-based:** far from center in standardized space.

#### 5) Inference: focus on false positives/negatives

Anomaly detection is not hypothesis testing; it is a scoring/ranking tool.
Validate by:
- checking whether known crises score high,
- inspecting the top anomalies qualitatively.

#### 6) Diagnostics + robustness (minimum set)

1) **Known-event sanity**
- do 2008/2020 periods appear as anomalies?

2) **Feature contribution**
- which features drive the anomaly score? (inspect deviations)

3) **Threshold sensitivity**
- how many anomalies do you flag under different thresholds?

4) **Stability**
- do top anomalies persist if you change feature set or standardization window?

#### 7) Interpretation + reporting

Report:
- anomaly method and preprocessing,
- how threshold was chosen,
- examples of top anomalies and what indicators drove them.

**What this does NOT mean**
- anomalies are not causal explanations; they are flags.

#### Exercises

- [ ] Fit an anomaly detector and list the top 10 anomalous dates; interpret at least 3.
- [ ] Compare two methods (PCA reconstruction vs isolation forest) and discuss differences.
- [ ] Vary the anomaly threshold and report how many periods are flagged.

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
