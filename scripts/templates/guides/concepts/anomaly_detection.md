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
