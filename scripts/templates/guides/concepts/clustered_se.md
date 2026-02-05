### Clustered Standard Errors: Inference when observations move together

Clustered standard errors are about **honest uncertainty** when errors are correlated within groups.

#### 1) Intuition (plain English)

Regression formulas for standard errors often assume each row is “independent.”
In economics, that is frequently false.

**Story example:** Counties in the same state share:
- state policy changes,
- state-level business cycles,
- housing markets and migration flows.

If your residuals are correlated within states and you ignore that, your standard errors are often too small, making effects look “significant” when they are not.

#### 2) Notation + setup (define symbols)

Consider a linear regression in stacked form:

$$
\\mathbf{y} = \\mathbf{X}\\beta + \\mathbf{u}.
$$

Let:
- $n$ be the number of observations (rows),
- $K$ be the number of regressors,
- $g \\in \\{1,\\dots,G\\}$ index clusters (e.g., states),
- $\\mathbf{X}_g$ be the rows of $\\mathbf{X}$ in cluster $g$,
- $\\mathbf{u}_g$ be the residual vector in cluster $g$.

#### 3) What clustering assumes (and what it relaxes)

Cluster-robust SE relax the “independent errors” assumption **within** clusters:
- errors can be heteroskedastic and correlated in arbitrary ways within a cluster.

But they still assume something like **independence across clusters**:
- cluster $g$ shocks are independent of cluster $h$ shocks (for $g \\neq h$).

This is why **choosing the right cluster level is part of the design**, not a technical afterthought.

#### 4) Estimation mechanics: the cluster-robust covariance (sandwich form)

The OLS coefficient estimate $\\hat\\beta$ does not change when you change SE.
What changes is the estimated variance of $\\hat\\beta$.

The cluster-robust (one-way) covariance estimator is:

$$
\\widehat{\\mathrm{Var}}_{\\text{CL}}(\\hat\\beta)
= (\\mathbf{X}'\\mathbf{X})^{-1}
\\left(\\sum_{g=1}^{G} \\mathbf{X}_g' \\hat{\\mathbf{u}}_g \\hat{\\mathbf{u}}_g' \\mathbf{X}_g \\right)
(\\mathbf{X}'\\mathbf{X})^{-1}.
$$

**What each term means**
- “Bread”: $(\\mathbf{X}'\\mathbf{X})^{-1}$ is the same matrix you see in OLS.
- “Meat”: the sum over clusters aggregates within-cluster residual covariance.
- This estimator reduces to robust (HC) SE when each observation is its own cluster.

Many libraries also apply small-sample corrections (especially when $G$ is not huge).

#### 5) Mapping to code (statsmodels / linearmodels)

In `statsmodels`, you can request clustered SE via `cov_type='cluster'` with a `groups` vector.

In `linearmodels` (PanelOLS), clustering is common:
- build a `clusters` Series aligned to the panel index,
- use `cov_type='clustered'`.

#### 6) Inference pitfalls (important!)

1) **Few clusters problem**
- When the number of clusters $G$ is small (rule of thumb: < 30–50), cluster-robust inference can be unreliable.
- In serious applied work, people use remedies like wild cluster bootstrap; we do not implement that here, but you should know it exists.

2) **Wrong clustering level**
- If treatment is assigned at the state level, clustering below that (e.g., county) can be too optimistic.
- A common rule: cluster at the level of treatment assignment or the level of correlated shocks.

3) **Serial correlation in DiD**
- Classic results (e.g., Bertrand et al.) show naive SE can be severely biased in DiD with serial correlation.
- Clustering is often the minimum fix.

#### 7) Diagnostics + robustness (at least 3)

1) **Report cluster count**
- Always report $G$ (number of clusters). If $G$ is tiny, treat inference as fragile.

2) **Sensitivity to clustering level**
- Try clustering by county vs by state (or time). If SE change a lot, dependence is important.

3) **Residual dependence check**
- Plot residuals by cluster over time or compute within-cluster autocorrelation.

4) **Aggregation robustness (DiD-style)**
- As a sanity check, aggregate to the treatment-assignment level and see if conclusions are similar.

#### 8) Interpretation + reporting

Clustered SE change your uncertainty, not your point estimate.
So the right way to write results is:
- coefficient (effect size),
- clustered SE (uncertainty),
- cluster level and number of clusters,
- design assumptions.

**What this does NOT mean**
- Clustering does not fix bias from confounding or misspecification.
- Clustering does not “prove” causality; it only helps prevent overconfident inference.

#### Exercises

- [ ] In a panel regression, compute naive (HC) SE and clustered SE; compare the ratio for your main coefficient.
- [ ] Explain in words why state-level shocks make county-level rows dependent.
- [ ] Try clustering by entity vs by state; write 3 sentences about which is more defensible and why.
- [ ] If you had only 8 states, what would make you cautious about cluster-robust p-values?
