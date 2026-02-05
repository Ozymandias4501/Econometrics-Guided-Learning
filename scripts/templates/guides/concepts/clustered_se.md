### Clustered Standard Errors (Why Robust SE Still Isn’t Enough)

Robust (HC) standard errors handle heteroskedasticity.
They do **not** handle correlation in errors across related observations.

> **Definition:** **Clustered standard errors** allow errors to be correlated within groups (clusters), but assume independence across clusters.

Why this matters in panels:
- Counties within the same state can share shocks (policy, labor markets).
- A county’s errors can be correlated over time (serial correlation).

Common choices:
- cluster by entity (county)
- cluster by higher-level geography (state)

Practical caution:
- With very few clusters, cluster-robust inference can be unreliable.
- Always report the number of clusters you used.

