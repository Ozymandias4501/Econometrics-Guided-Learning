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

### Core Unsupervised Learning: Describe Structure Before Predicting

Unsupervised methods help you understand structure in the indicators.
They do not require a target label.

#### Standardization matters
Many unsupervised methods rely on distances or variances.
If one variable has larger units, it dominates.

> **Definition:** **Standardization** rescales each feature to mean 0 and standard deviation 1.

#### Interpretation stance
Treat unsupervised outputs as:
- descriptions (factors, regimes, anomalies)
- hypotheses you can investigate

Avoid treating them as causal explanations.

### Deep Dive: Anomaly Detection (Crisis Periods)

Anomaly detection flags observations that look unusual relative to the bulk of the data.
In macro, anomalies often correspond to crises, but not always.

#### Key terms (defined)
> **Definition:** An **outlier** is an observation far from typical behavior.

> **Definition:** An **anomaly score** measures "unusualness".

> **Definition:** **Isolation Forest** isolates points by random splits; anomalies isolate quickly.

> **Definition:** **Contamination** is the expected fraction of anomalies (a hyperparameter).

#### Python demo: Isolation Forest intuition (commented)
```python
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# Mostly normal points
X_normal = rng.normal(size=(300, 3))

# A few anomalous points
X_anom = rng.normal(loc=6.0, scale=1.0, size=(10, 3))

X = np.vstack([X_normal, X_anom])
X = StandardScaler().fit_transform(X)

iso = IsolationForest(contamination=0.05, random_state=0).fit(X)

# Higher score => more anomalous
scores = -iso.score_samples(X)
print(scores[-10:])
```

#### Interpretation cautions
- Anomaly does not mean recession.
- Anomaly means the pattern is rare.

#### Debug checklist
1. Standardize features.
2. Sensitivity-check contamination.
3. Inspect which indicators are extreme in anomalous periods.

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
