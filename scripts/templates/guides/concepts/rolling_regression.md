### Deep Dive: Rolling regressions — coefficient stability over time

Rolling regressions are a simple tool to see whether relationships drift across macro regimes.

#### 1) Intuition (plain English)

In macro data, relationships change:
- policy regimes shift,
- financial structure changes,
- measurement changes.

A single “full-sample” regression can hide instability.
Rolling regressions estimate coefficients repeatedly on moving windows to reveal drift.

#### 2) Notation + setup (define symbols)

Let window length be $W$ (in periods).
For each end time $t \\ge W$, define the window:
$$
\\{t-W+1, \\dots, t\\}.
$$

Estimate:
$$
\\hat\\beta_t = \\arg\\min_{\\beta} \\sum_{s=t-W+1}^{t} (y_s - x_s'\\beta)^2.
$$

**What each term means**
- $\\hat\\beta_t$ is the coefficient estimate using only the most recent $W$ observations ending at $t$.

#### 3) Assumptions and caveats

Rolling regressions assume:
- the relationship is approximately stable within each window,
- $W$ is large enough for estimation but small enough to detect changes.

Inference caveat:
- consecutive windows overlap heavily, so estimates are correlated.

#### 4) Mechanics (practical use)

Typical workflow:
1) choose a window (e.g., 40 quarters ≈ 10 years),
2) estimate $\\hat\\beta_t$ for each $t$,
3) plot $\\hat\\beta_t$ with CI bands,
4) compare drift to recession shading or known events.

#### 5) Diagnostics + robustness (minimum set)

1) **Window sensitivity**
- try multiple $W$; does the drift pattern persist?

2) **Residual diagnostics**
- within each window, check autocorrelation; HAC may still be needed.

3) **Regime interpretation**
- do coefficient changes align with known macro events? (recessions, policy changes)

#### 6) Interpretation + reporting

Rolling coefficients suggest instability but do not identify why.
Report:
- window length,
- SE choice,
- and a narrative linking drift to plausible regime changes.

#### Exercises

- [ ] Run a rolling regression and plot the coefficient path with CI.
- [ ] Compare two window lengths and explain the bias–variance trade-off.
- [ ] Identify one period where the coefficient changes sign and propose a macro explanation.
