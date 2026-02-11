# Guide: 03_anomaly_detection

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/04_unsupervised/03_anomaly_detection.ipynb`.

> **Note:** Anomaly detection is a useful general-purpose tool for flagging unusual observations in any dataset — health outcomes, financial data, survey responses. In health economics, it can help identify outlier patients, suspicious billing patterns, or unusual geographic variation. The macro-specific examples below are illustrative; the methods are domain-agnostic.

### Key Terms (defined)
- **Anomaly / outlier**: an observation that is substantially different from the typical pattern in the data.
- **Anomaly score**: a numerical measure of how "unusual" an observation is — higher (or lower, depending on convention) = more anomalous.
- **Mahalanobis distance**: multivariate distance from the mean, scaled by the covariance matrix — accounts for correlation between features.
- **Isolation forest**: a tree-based method that identifies anomalies as observations that are "easy to isolate" (few random splits needed).
- **PCA reconstruction error**: fit PCA, reconstruct each observation from the top components, and measure the reconstruction error. Anomalies have high error.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Standardize features before applying any anomaly detection method.
- Fit at least two methods (e.g., Mahalanobis + Isolation Forest).
- List the top 10 anomalous periods; inspect what makes them unusual.
- Compare flagged anomalies to known crisis dates (2008, 2020).
- Vary the anomaly threshold and report how the number of flagged periods changes.

### Alternative Example (Not the Notebook Solution)
```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 4))
X[10] = [5, -4, 6, -3]  # inject an obvious outlier
X[100] = [4, 5, -5, 4]

X_scaled = StandardScaler().fit_transform(X)

# Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=0).fit(X_scaled)
scores = iso.decision_function(X_scaled)
print("Top 5 anomalous indices:", np.argsort(scores)[:5])

# Mahalanobis distance
cov = np.cov(X_scaled, rowvar=False)
cov_inv = np.linalg.inv(cov)
mean = X_scaled.mean(axis=0)
mah_dists = [mahalanobis(x, mean, cov_inv) for x in X_scaled]
print("Top 5 Mahalanobis:", np.argsort(mah_dists)[-5:])
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### What anomaly detection does

Anomaly detection assigns a score to each observation measuring how "unusual" it is relative to the bulk of the data. It is unsupervised — no labels required — and produces a ranking rather than a binary classification.

### Method comparison

| Method | Idea | Strengths | Weaknesses |
|---|---|---|---|
| **Mahalanobis distance** | Distance from multivariate mean, scaled by covariance | Fast, interpretable, principled for Gaussian data | Assumes elliptical distribution; sensitive to outliers in the covariance estimate |
| **Isolation forest** | Anomalies are "easy to isolate" with random splits | Handles nonlinear patterns; no distributional assumptions | Less interpretable — hard to explain *why* a point is anomalous |
| **PCA reconstruction error** | Fit PCA, measure $\|x - \hat{x}\|$ | Good when anomalies deviate from the main factor structure | Sensitive to number of components; misses anomalies along main PCs |

### Mahalanobis distance

For observation $x$ with sample mean $\bar{x}$ and covariance matrix $S$:

$$
d_M(x) = \sqrt{(x - \bar{x})' S^{-1} (x - \bar{x})}.
$$

Under multivariate normality, $d_M^2$ is approximately $\chi^2_p$ (where $p$ is the number of features), giving a natural threshold for flagging anomalies.

### Diagnostics checklist

1. **Known-event sanity** — do major crisis periods appear as anomalies?
2. **Feature contribution** — which features drive the anomaly score? Inspect the deviations.
3. **Threshold sensitivity** — how many anomalies do you flag at different thresholds?
4. **Method agreement** — do different methods flag the same observations?

### What this does NOT mean
- Anomalies are flags, not causal explanations.
- A point can be anomalous in one feature space and normal in another.
- Structural breaks can shift what is "normal," so fixed thresholds may not work across regimes.

#### Exercises

- [ ] Fit two anomaly detectors and list the top 10 anomalous dates from each; compare overlap.
- [ ] For the top 3 anomalies, identify which features drive the high score.
- [ ] Vary the contamination/threshold parameter and plot the number of flagged periods.

### Project Code Map
- `src/features.py`: feature engineering helpers
- `data/sample/panel_monthly_sample.csv`: offline dataset

### Common Mistakes
- Forgetting to standardize before computing distances.
- Using a single method without cross-checking with another.
- Over-interpreting anomalies as "something is wrong" — they are simply unusual observations.

<a id="summary"></a>
## Summary + Suggested Readings

Anomaly detection flags unusual observations in multivariate data. Use multiple methods, validate against known events, and treat results as hypotheses.

Suggested readings:
- scikit-learn docs: IsolationForest, outlier detection
- Aggarwal (2017): *Outlier Analysis*
