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
