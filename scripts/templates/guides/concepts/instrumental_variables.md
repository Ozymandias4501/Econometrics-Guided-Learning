### Instrumental Variables (IV) and 2SLS

IV is used when a regressor is **endogenous** (correlated with the error term), often due to:
- omitted variables,
- reverse causality,
- measurement error.

> **Definition:** An **instrument** $Z$ must satisfy:
1) **Relevance:** $Z$ shifts the endogenous regressor $X$ (strong first stage).
2) **Exclusion:** $Z$ affects $Y$ only through $X$ (no direct path).

#### 2SLS mechanics (high level)
1. First stage: regress $X$ on $Z$ (and controls) to get predicted $\\hat{X}$.
2. Second stage: regress $Y$ on $\\hat{X}$ (and controls).

#### Weak instruments
If $Z$ barely predicts $X$, 2SLS can be badly behaved.
Always inspect first-stage strength (e.g., an F-statistic style check).

Interpretation caution:
- With heterogeneous treatment effects, IV often identifies a **local** effect (LATE) for compliers.

