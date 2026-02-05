### Cointegration + ECM: long-run equilibrium with short-run dynamics

Cointegration is the key “exception” to the rule that nonstationary levels regressions are spurious.

#### 1) Intuition (plain English)

Two series can trend over time but still be meaningfully linked:
- consumption and income,
- prices and money aggregates,
- wages and productivity (in some settings).

The idea:
- each series drifts,
- but some combination of them is stable in the long run.

Cointegration formalizes “move together over decades.”

#### 2) Notation + setup (define symbols)

Let $y_t$ and $x_t$ be time series.
Assume each is I(1) (nonstationary in levels, stationary in first differences).

They are **cointegrated** if there exists a parameter $\\beta$ such that:

$$
u_t = y_t - \\beta x_t
\\quad \\text{is I(0) (stationary).}
$$

**What each term means**
- $\\beta$: long-run relationship (“equilibrium” slope).
- $u_t$: deviation from the long-run equilibrium; should mean-revert if cointegration holds.

#### 3) Why cointegration matters

If $y_t$ and $x_t$ are cointegrated:
- a regression in levels can capture a real long-run relationship,
- but you must model short-run dynamics carefully.

If they are not cointegrated:
- levels regression can be spurious,
- differencing is usually safer.

#### 4) Engle–Granger two-step procedure (mechanics)

Step 1: estimate the long-run relationship in levels:

$$
y_t = a + \\beta x_t + e_t.
$$

Step 2: test whether the residuals are stationary:
$$
\\hat u_t = y_t - \\hat a - \\hat\\beta x_t.
$$

If $\\hat u_t$ is stationary, that supports cointegration.

Important practical note:
- residual-based cointegration tests use nonstandard critical values (packages like `statsmodels` handle this for you).

#### 5) Error Correction Model (ECM): connecting short-run changes to long-run gaps

If $y_t$ and $x_t$ are cointegrated, an ECM often makes sense:

$$
\\Delta y_t = c + \\alpha \\,(y_{t-1} - \\beta x_{t-1}) + \\Gamma \\Delta x_t + \\varepsilon_t.
$$

**What each term means**
- $\\Delta y_t$: short-run change in $y$.
- $(y_{t-1} - \\beta x_{t-1})$: last period’s equilibrium error (how far you were from long-run relationship).
- $\\alpha$: “speed of adjustment” (typically negative if deviations are corrected).
- $\\Gamma \\Delta x_t$: short-run effect of changes in $x$.

Interpretation of $\\alpha$:
- If $y$ is above its long-run equilibrium relative to $x$ (positive error),
  a negative $\\alpha$ pulls $\\Delta y_t$ downward to correct.

#### 6) Assumptions and practical caveats

Cointegration/ECM is most appropriate when:
- both series are integrated of the same order (often I(1)),
- the long-run relationship is stable over the sample,
- there are no major structural breaks (breaks can change $\\beta$).

If there are breaks (policy regime shifts, measurement changes), cointegration results can be misleading.

#### 7) Mapping to code (statsmodels)

Useful tools:
- `statsmodels.tsa.stattools.coint(y, x)` runs an Engle–Granger cointegration test.
- Build the ECM manually with `statsmodels.api.OLS`:
  - compute the lagged error term $(y_{t-1} - \\hat\\beta x_{t-1})$,
  - regress $\\Delta y_t$ on that error and $\\Delta x_t$ (and possibly lags).

#### 8) Diagnostics + robustness (minimum set)

1) **Confirm integration order**
- Test whether $x_t$ and $y_t$ are I(1) (ADF/KPSS on levels and differences).

2) **Residual stationarity**
- After Step 1, test whether $\\hat u_t$ is stationary; plot it too.

3) **ECM residual checks**
- Check autocorrelation of ECM residuals; add lags if needed.

4) **Stability**
- Re-estimate cointegration relationship on subperiods; do parameters drift?

#### 9) Interpretation + reporting

When reporting cointegration/ECM:
- clearly separate long-run relationship ($\\beta$) from short-run dynamics ($\\Gamma$),
- interpret $\\alpha$ as speed of adjustment (sign matters!),
- include plots of levels, residual (error-correction term), and differenced series.

**What this does NOT mean**
- Cointegration does not prove causality; it describes a stable long-run relationship.
- A significant $\\beta$ in levels is not meaningful if residuals are nonstationary.

#### Exercises

- [ ] Pick two trending macro series and test whether each is I(1).
- [ ] Run a cointegration test; interpret the result and plot the residual $\\hat u_t$.
- [ ] Fit an ECM and interpret $\\alpha$ in words (speed of adjustment).
- [ ] Try adding one lag of $\\Delta y_t$ and $\\Delta x_t$ and see whether residual autocorrelation improves.
