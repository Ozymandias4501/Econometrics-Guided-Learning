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

Let $X$ be an $n \\times p$ matrix of features:
- rows: observations (time periods),
- columns: indicators (standardized).

Standardization:
$$
\\tilde X_{ij} = \\frac{X_{ij} - \\bar X_j}{s_j}.
$$

Sample covariance matrix (for standardized data):
$$
S = \\frac{1}{n}\\tilde X'\\tilde X.
$$

PCA finds eigenvectors of $S$:
$$
S w_k = \\lambda_k w_k,
$$
with eigenvalues $\\lambda_1 \\ge \\lambda_2 \\ge \\cdots$.

PC scores:
$$
z_{ik} = \\tilde x_i' w_k,
$$
where $\\tilde x_i$ is row $i$ of $\\tilde X$.

**What each term means**
- $w_k$: loading vector for component $k$ (how indicators combine).
- $\\lambda_k$: variance explained by component $k$.
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
\\tilde X = U \\Sigma V'.
$$

Then:
- loadings are columns of $V$,
- explained variance relates to singular values in $\\Sigma$.

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
