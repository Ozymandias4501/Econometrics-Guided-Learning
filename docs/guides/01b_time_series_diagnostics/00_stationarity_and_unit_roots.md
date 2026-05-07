# Guide: 00_stationarity_and_unit_roots

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01b_time_series_diagnostics/00_stationarity_and_unit_roots.ipynb`.

This is the **definitive reference** for stationarity, unit roots, the spurious-regression problem, and the differencing decision in this project. Later guides — especially `02_regression/04_inference_time_series_hac.md` — assume you have read this one.

### Key Terms (defined)
- **Stationarity (weak)**: a series whose mean, variance, and autocovariance at each lag do not depend on time. The OLS asymptotics you learned in §02 require some form of stationarity to hold.
- **Unit root**: a property of a stochastic process where shocks have permanent effects; the series is integrated of order one or higher.
- **Random walk**: $y_t = y_{t-1} + \varepsilon_t$. The canonical $I(1)$ process. Variance grows linearly with $t$.
- **Order of integration $d$**: the number of times you must difference a series to make it stationary. Most macro levels are $I(1)$; growth rates are usually $I(0)$.
- **Spurious regression**: a regression of one non-stationary series on another that produces small p-values and large t-statistics even when the series are independent.
- **ADF (Augmented Dickey–Fuller)** test: tests $H_0$: unit root vs $H_1$: stationary. Reject = stationary.
- **KPSS** test: tests $H_0$: stationary vs $H_1$: unit root. Reject = non-stationary. Complement to ADF.
- **Trend-stationary** vs **difference-stationary**: trend-stationary series can be made stationary by removing a deterministic trend; difference-stationary series need differencing. They look similar in plots and behave very differently in regressions.
- **Cointegration**: two or more $I(1)$ series whose linear combination is $I(0)$ — they share a long-run equilibrium even though each wanders.

### How To Read This Guide
- Use **Step-by-Step** for the notebook checklist and a self-contained spurious-regression demo.
- Use **Technical Explanations** for ADF/KPSS mechanics, the math of unit roots, the differencing decision, and one paragraph of cointegration.
- For OLS foundations and HAC inference, see [Guide 02_regression/04](../02_regression/04_inference_time_series_hac.md). HAC fixes inference under autocorrelation but **does not** fix the spurious-regression problem.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- [ ] Simulate white noise, a random walk, and a random walk with drift; eyeball them.
- [ ] Run the spurious-regression Monte Carlo: regress one independent random walk on another, 500 times, in levels and in first differences.
- [ ] Use `econometrics.stationarity_table` on the three caricatures and confirm the verdicts match your intuition.
- [ ] Re-test a trend-stationary series with `regression='ct'` (constant + trend) and see how the verdict changes.
- [ ] Apply `stationarity_table` to the macro panel in levels.
- [ ] Build log-differences for level series (GDPC1, CPIAUCSL, INDPRO, RSAFS) and first differences for rate series (UNRATE, FEDFUNDS); re-test.
- [ ] Decide and write down which transformations you will use as defaults in §02.

### Alternative Example: ADF on a Cointegrated Pair (Engle–Granger)

This demonstration shows that two non-stationary series *can* be regressed on each other safely if they cointegrate.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src import econometrics

rng = np.random.default_rng(0)
T = 500

# Common stochastic trend
z = rng.normal(size=T).cumsum()

# Two series that share the trend plus their own stationary noise
y = 1.0 + 2.0 * z + rng.normal(scale=0.5, size=T)
x = 0.5 + 1.0 * z + rng.normal(scale=0.5, size=T)

# Step 1: confirm both are I(1) in levels
print(econometrics.stationarity_table(
    pd.DataFrame({'y': y, 'x': x}), ['y', 'x']
))

# Step 2: regress in levels and inspect residuals
res = sm.OLS(y, sm.add_constant(x)).fit()
resid = pd.Series(res.resid)

# Step 3: ADF on residuals — if rejected, y and x are cointegrated
print(econometrics.adf_test(resid).as_dict())
```

You should find that y and x are each $I(1)$ in levels but the OLS residual is stationary (ADF rejects). That is Engle–Granger evidence of cointegration: the levels regression is recovering a real long-run relationship, not a spurious one. (Note: when you do this with real data, you also need to consult Engle–Granger critical values rather than ADF's tabulated ones, because the residual is estimated. For teaching purposes, the standard ADF p-value is fine.)

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Why stationarity is the first thing to check

OLS does not break "loudly" on non-stationary data. It quietly returns coefficients, t-statistics, and an $R^2$, all of which look reasonable. The Granger–Newbold (1974) result is that two **independent** random walks regressed on each other reject $H_0: \beta = 0$ at the 5% level roughly 75% of the time, with $R^2$ values that can exceed 0.5. Nothing in the OLS summary table flags the problem.

This is a different failure mode from the autocorrelation issue HAC fixes. HAC corrects the variance estimator when residuals are correlated. The spurious-regression problem is that the *coefficient* itself is converging to a non-degenerate distribution (not a fixed number), so the t-statistic is structurally invalid no matter what SE you use.

The fix is to model **stationary** quantities — usually growth rates or first differences. Or, if you really want to model levels, prove cointegration first.

### Stationarity, formally

A process $\{y_t\}$ is **strictly stationary** if its joint distribution is invariant under time shifts. We almost never need that. The useful concept is **weak (covariance) stationarity**:

1. $\mathbb{E}[y_t] = \mu$ for all $t$,
2. $\mathrm{Var}(y_t) = \sigma^2 < \infty$ for all $t$,
3. $\mathrm{Cov}(y_t, y_{t-k}) = \gamma(k)$ depends only on $k$, not on $t$.

If any of these change with $t$, OLS asymptotics break.

### Unit roots, in two equations

Consider the AR(1) model $y_t = \rho y_{t-1} + \varepsilon_t$.
- If $|\rho| < 1$, $y_t$ is **stationary** with mean reversion.
- If $\rho = 1$, $y_t$ is a **random walk**: shocks accumulate forever, variance grows with $t$. We say $y_t$ has a **unit root**.
- If $|\rho| > 1$, $y_t$ is **explosive** — extremely rare in macro data.

The ADF test re-parameterizes the AR(1) as $\Delta y_t = (\rho - 1) y_{t-1} + \varepsilon_t$ and tests $H_0: (\rho - 1) = 0$ (unit root) against $H_1: (\rho - 1) < 0$ (stationary). The "augmented" part adds lags of $\Delta y_t$ to soak up serial correlation in residuals. The catch is that the test statistic does **not** follow a standard t-distribution under the null — it has its own (Dickey–Fuller) distribution, which is why the critical values look unfamiliar.

### KPSS test, mechanically

KPSS flips the null. It models $y_t$ as a sum of a random walk and a stationary residual and tests whether the random-walk variance is zero. Reject = the random-walk component is non-zero = non-stationary.

Why bother running both? **Power**: ADF and KPSS each have low power in different regions, so they disagree on borderline cases (e.g., a stationary AR(1) with $\rho = 0.95$). Running them together gives you a four-cell decision rule that is more honest than relying on either alone.

| ADF rejects? | KPSS rejects? | Verdict |
|---|---|---|
| Yes | No | Stationary |
| No | Yes | Non-stationary (unit root) |
| Yes | Yes | Conflicting — likely **trend-stationary** or borderline |
| No | No | Inconclusive — low power, not enough data |

### Choosing the regression spec for ADF

ADF takes a `regression` argument. Pick deliberately:
- `"n"` — no constant. Use only when you have a clear prior the series is mean-zero.
- `"c"` — constant only (default). Use for most growth-rate series.
- `"ct"` — constant + linear trend. Use when the series clearly grows on average (real GDP levels, CPI levels).
- `"ctt"` — constant + linear + quadratic trend. Rare.

If you mis-specify the trend, ADF can fail to reject when it should, or reject for the wrong reason. The notebook walks through this with the trend-stationary caricature.

### The differencing decision (a recipe)

For a column you intend to use as a feature or target in §02:

1. Plot it. If it visibly drifts up or down for years, it is almost certainly $I(1)$.
2. Run `stationarity_table` on it.
3. If verdict is `stationary`, leave it alone.
4. If verdict is `non-stationary (unit root)`:
   - For positive-valued levels (prices, GDP, CPI, employment): take **log differences** ($\Delta \log y_t = \log y_t - \log y_{t-1}$). This is approximately the percent change and is symmetric in growth and decline.
   - For rates and spreads (UNRATE, FEDFUNDS, T10Y2Y): take **first differences** ($\Delta y_t = y_t - y_{t-1}$).
5. Re-test the transformed series. If it is now `stationary`, use it. If it is still flagged, you may need a second difference, or you may have a structural break causing trouble (see §04 for the structural-break angle).
6. If verdict is `conflicting`, try ADF with `regression='ct'`; that often resolves it as trend-stationary, in which case you can de-trend instead of differencing.

**Do not** difference reflexively. Over-differencing introduces a non-invertible MA(1) term and inflates noise. The point is the lowest level of differencing that achieves stationarity.

### Cross-Reference: When HAC is not enough

The HAC guide (`02_regression/04`) covers what to do when residuals are autocorrelated but the series are stationary. This guide covers what to do when the series themselves are non-stationary. **Both can be true at once**, and they require different fixes:

| Symptom | Tool |
|---|---|
| Residuals autocorrelated, series stationary | HAC SE |
| Series non-stationary, even after differencing residuals look fine | Difference (this guide) |
| Series non-stationary, but two of them cointegrate | Levels regression + Engle–Granger residual ADF |
| Series non-stationary, residuals autocorrelated, no cointegration | Difference, then HAC if needed |

### Cointegration, the one paragraph version

If $y$ and $x$ are both $I(1)$ but there exists a constant $\beta$ such that $y - \beta x$ is $I(0)$, then $y$ and $x$ are **cointegrated** with cointegrating vector $(1, -\beta)$. Economically: they share a common stochastic trend and revert to a long-run equilibrium. Statistically: the OLS regression of $y$ on $x$ in levels is *not* spurious; in fact, the OLS estimator of $\beta$ is **super-consistent** (converges faster than $\sqrt{T}$) under cointegration. The two-step Engle–Granger test is exactly that recipe: regress in levels, ADF the residuals. If the residuals reject the unit-root null (using Engle–Granger critical values for proper inference), you have cointegration. This curriculum sticks to growth-rate models and does not exploit cointegration further, but you should recognize the term.

### Project Code Map
- `src/econometrics.py`: `adf_test`, `kpss_test`, `stationarity_table`, `chow_test` — all wrappers over `statsmodels.tsa.stattools` plus a small Chow implementation.
- `src/features.py`: `add_diff_features`, `add_log_diff_features`, `add_pct_change_features` for building stationary transforms.
- `src/macro.py`: `gdp_growth_qoq`, `gdp_growth_qoq_annualized`, `gdp_growth_yoy` — the macro-specific stationary transforms used elsewhere in the curriculum.

### Common Mistakes
- Running only ADF and treating non-rejection as proof of non-stationarity. ADF has notoriously low power on samples under 200 observations; it under-rejects.
- Differencing a series that was already stationary. Over-differencing introduces a unit-root MA component and amplifies noise.
- Using ADF with `regression='c'` on a clearly trending series and concluding "non-stationary" when the right verdict is "trend-stationary."
- Treating a cointegrated pair as separate $I(1)$ series and differencing both. The level relationship contains real information that differencing destroys.
- Trusting the OLS summary table on levels regressions in macro work. The t-statistics are not what they appear to be.

<a id="summary"></a>
## Summary + Suggested Readings

This guide gives you the minimum vocabulary to avoid the most common time-series mistake: regressing non-stationary series on each other and reading the t-stat table. You should now be able to:

- define stationarity in plain language and in symbols,
- run the spurious-regression Monte Carlo and explain the rejection rate,
- use ADF and KPSS together and interpret the four-cell decision rule,
- pick the right `regression` argument (`c` vs `ct`) for ADF,
- decide between log-differencing and first-differencing for a given macro series,
- recognize the term "cointegration" and know when to dig further.

**Companion guides:**
- [Guide 02_regression/04](../02_regression/04_inference_time_series_hac.md): HAC SE for autocorrelation in residuals (different problem from this guide; sometimes both apply).
- [Guide 02_regression/02](../02_regression/02_single_factor_regression_macro.md): the first macro regression — uses growth rates, not levels, for exactly the reasons in this guide.

**Suggested readings:**
- Granger and Newbold (1974), "Spurious Regressions in Econometrics" — the original demonstration.
- Hamilton, *Time Series Analysis*, Ch. 17 (univariate processes with unit roots), Ch. 19 (cointegration). Standard graduate reference.
- Wooldridge, *Introductory Econometrics*, Ch. 18 (advanced time series — stationarity, unit roots, cointegration in plain language).
- Stock and Watson, *Introduction to Econometrics*, Ch. 14–15 (stationarity, ADF, forecasting).
- statsmodels docs: `statsmodels.tsa.stattools.adfuller`, `statsmodels.tsa.stattools.kpss`.
