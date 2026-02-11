# Guide: 02_single_factor_regression_macro

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/02_single_factor_regression_macro.ipynb`.

Macro regression applies OLS to **time-series** economic data. The mechanics of OLS are the same as in cross-section (see [Guide 00](00_single_factor_regression_micro.md#technical) for the full treatment), but the data-generating process is fundamentally different: observations are ordered in time, errors are serially correlated, and many macro variables are non-stationary. This guide focuses on what changes when you move from cross-section to time series.

### Key Terms (defined)
- **Time series**: observations indexed by time; ordering matters.
- **Stationarity**: a process whose statistical properties (mean, variance, autocovariance) do not change over time. Required for standard regression inference.
- **Autocorrelation (serial correlation)**: errors at time $t$ are correlated with errors at $t-k$. Common in macro data.
- **Newey-West / HAC SE**: standard errors that remain valid when errors are heteroskedastic and autocorrelated. See [Guide 04](04_inference_time_series_hac.md#technical) for the definitive treatment.
- **Spurious regression**: regressing one non-stationary series on another produces misleadingly "significant" results.
- **Macro indicators**: aggregate economic variables (GDP growth, unemployment, interest rates, inflation) observed at quarterly or monthly frequency.

### How To Read This Guide
- Use **Step-by-Step** for the notebook checklist and an alternative worked example.
- Use **Technical Explanations** for what makes time-series regression different from cross-section.
- For OLS foundations (derivation, assumptions, interpretation), see [Guide 00](00_single_factor_regression_micro.md#technical).
- For hypothesis testing and HAC details (Newey-West formula, maxlags), see [Guide 04](04_inference_time_series_hac.md#technical).

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- [ ] Load macro data (e.g., FRED series via `src/data.py`).
- [ ] Plot the series; visually assess whether levels look stationary or trending.
- [ ] If levels are non-stationary, transform to growth rates or differences before regressing.
- [ ] Fit a plain OLS regression of one macro variable on another.
- [ ] Inspect the residual ACF/PACF to check for serial correlation.
- [ ] Re-fit with HAC SE (`cov_type='HAC'`) and compare naive vs HAC standard errors.
- [ ] Interpret the coefficient: units, sign, magnitude, and what it does *not* imply causally.

### Alternative Example (Not the Notebook Solution)

Regress the change in unemployment on the Treasury yield spread (10Y minus 3M), a classic recession-forecasting relationship.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(42)
T = 160  # ~40 years of quarterly data

# Simulated yield spread (stationary) and unemployment change
spread = rng.normal(loc=1.5, scale=1.0, size=T)
# Negative spread predicts rising unemployment (stylized)
d_unemp = -0.3 * spread + 0.7 * rng.normal(size=T)
# Inject AR(1) serial correlation into the errors
for t in range(1, T):
    d_unemp[t] += 0.4 * (d_unemp[t-1] + 0.3 * spread[t-1])

df = pd.DataFrame({'d_unemp': d_unemp, 'spread': spread})
X = sm.add_constant(df[['spread']])
res = sm.OLS(df['d_unemp'], X).fit()

# Compare naive vs HAC SE
res_hac = res.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': 4})
print("Naive SE:", res.bse['spread'].round(4))
print("HAC SE:  ", res_hac.bse['spread'].round(4))
print("Coefficient:", res.params['spread'].round(4))
```

Notice how HAC SE differ from naive SE when residuals are autocorrelated.

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Cross-References (no duplication)

**OLS foundations** -- derivation, assumptions, coefficient interpretation, diagnostics -- are covered in full in [Guide 00: Single-Factor Regression (Micro)](00_single_factor_regression_micro.md#technical). Read that first if you have not already.

**Hypothesis testing and HAC / Newey-West** -- p-values, confidence intervals, the Newey-West formula, Bartlett weights, maxlags guidance, and the SE comparison table -- are covered in [Guide 04: Inference and Time-Series HAC](04_inference_time_series_hac.md#technical). That guide is the definitive reference for all HAC-related content.

This section covers what is *unique* to macro time-series regression.

---

### Time-Series Regression: What is Different from Cross-Section

#### 1) Why macro data violates classical SE assumptions

Cross-section data (e.g., individuals in a survey) can often be treated as independent draws. Macro time series cannot:

- **Serial correlation.** Economic shocks persist: a recession quarter is followed by more recession quarters, not by a random draw. This means $\mathrm{Cov}(u_t, u_{t-k}) \neq 0$, violating the classical SE assumption. Naive SE are typically *too small*, inflating t-stats and producing spurious significance.

- **Conditional heteroskedasticity.** Volatility clusters in macro data (e.g., GDP growth variance spikes during crises). The error variance $\mathrm{Var}(u_t)$ is not constant. This further distorts naive SE.

- **Small effective sample sizes.** Even 50 years of quarterly data gives only $T = 200$. After differencing, lags, and burn-in, you may have far fewer effective observations than a cross-section with thousands of individuals.

The fix for inference is to use **HAC standard errors** (see [Guide 04](04_inference_time_series_hac.md#technical)). But the fix for *specification* -- making sure you are estimating something meaningful -- requires thinking about stationarity and identification.

#### 2) The stationarity requirement: why trending variables produce garbage

If you regress one I(1) variable (e.g., the price level) on another I(1) variable (e.g., nominal GDP), OLS will nearly always find a "significant" relationship, even when the two series are generated independently. This is **spurious regression** (Granger and Newbold, 1974): the $R^2$ converges to 1, the t-stats diverge, and the Durbin-Watson statistic converges to 0 as $T \to \infty$.

**Practical rule:** Before running any macro regression, check whether your variables are stationary. If they are not:
- Use growth rates, first differences, or log-differences.
- Or use cointegration methods (error-correction models) if you believe a long-run equilibrium relationship exists.

For a detailed treatment of unit root tests and stationarity, see [Guide 08: Stationarity and Unit Roots](../07_time_series_econ/00_stationarity_unit_roots.md).

#### 3) Interpretation challenges: association vs prediction vs causation

Macro regressions can serve three very different purposes, and conflating them is the most common mistake:

- **Association.** "Unemployment is correlated with the yield spread." This is descriptive; OLS gives you this.
- **Prediction / forecasting.** "The yield spread today predicts next-quarter unemployment." This is the basis for many leading-indicator models. OLS can be useful here, but you must evaluate out-of-sample and guard against look-ahead bias.
- **Causal identification.** "A 1pp increase in the Fed funds rate *causes* unemployment to rise by X pp." This requires a credible identification strategy (e.g., instrumental variables, natural experiments, structural VARs). OLS alone does not give you this.

A classic example: the yield curve (10Y-3M spread) is a strong predictor of recessions, but the regression coefficient does not tell you "what happens if we change the spread." It tells you about the joint behavior of interest rates, expectations, and the business cycle.

#### 4) Common macro regression pitfalls

**Data mining with many indicators.** FRED publishes thousands of series. If you screen 100 indicators for "significance" at the 5% level, you expect 5 false positives even under the null. Always distinguish between exploratory analysis and confirmatory testing.

**Look-ahead bias.** Many macro series are revised after initial release. Using revised data to "predict" the past overstates forecast accuracy. Use real-time vintage data when evaluating forecasting models.

**Structural breaks.** Macro relationships change over time. The Phillips curve slope, the Taylor rule coefficients, the yield-curve signal -- all have shifted across decades. A regression fit on 1960-2020 may not describe 2020-2025. Rolling regressions or subsample splits can reveal instability.

**Confounding from common trends.** Two series that both trend upward (e.g., healthcare spending and GDP) will show high $R^2$ even with no direct relationship. Always detrend or difference before interpreting correlations.

#### 5) When HAC is enough vs when you need more

HAC standard errors fix *inference* (SE, CI, p-values) under serial correlation and heteroskedasticity. They do not fix:
- **Non-stationarity.** If your regression is spurious, HAC cannot rescue it.
- **Endogeneity.** If your regressor is correlated with the error term, HAC gives you consistent SE for an inconsistent coefficient -- still biased.
- **Strong persistence.** When autocorrelation is extreme (near unit root in errors), HAC with fixed bandwidth may still underperform. Consider GLS/FGLS, Cochrane-Orcutt, or Prais-Winsten corrections, which model the error structure directly.
- **Multiple equation dynamics.** When feedback between variables matters (e.g., interest rates affect unemployment and unemployment affects interest rates), a single-equation OLS + HAC is insufficient. Consider VARs or structural approaches.

**Rule of thumb:** HAC is the right first step for time-series inference. If residual autocorrelation is mild (ACF dies off within a few lags), HAC with a reasonable `maxlags` is adequate. If autocorrelation is strong or persistent, you need to rethink the specification, not just the SE.

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hac`) + multicollinearity (`vif_table`)
- `src/macro.py`: GDP + labels (`gdp_growth_*`, `technical_recession_label`)
- `src/features.py`: time-series feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`)

### Common Mistakes
- Regressing non-stationary levels on each other (spurious regression).
- Using naive SE on autocorrelated residuals and over-trusting p-values.
- Interpreting a forecasting relationship as a causal effect.
- Ignoring structural breaks: fitting one regression to 60 years of macro data as if the world never changed.
- Data-mining dozens of FRED indicators without adjusting for multiple comparisons.

<a id="summary"></a>
## Summary + Suggested Readings

Macro time-series regression uses the same OLS mechanics as cross-section, but the data structure introduces serial correlation, non-stationarity, and interpretation challenges that require additional care. You should now be able to:
- check stationarity before regressing,
- use HAC SE for valid inference under serial correlation,
- distinguish association, prediction, and causation in macro contexts, and
- recognize common pitfalls (spurious regression, data mining, structural breaks).

**Companion guides:**
- [Guide 00](00_single_factor_regression_micro.md): OLS foundations (derivation, assumptions, diagnostics)
- [Guide 04](04_inference_time_series_hac.md): Hypothesis testing and HAC (Newey-West formula, maxlags, SE comparison)
- [Guide 08](../07_time_series_econ/00_stationarity_unit_roots.md): Stationarity and unit root tests

Suggested readings:
- Wooldridge: *Introductory Econometrics*, Ch. 10-12 (time-series regression)
- Hamilton: *Time Series Analysis* (stationarity, spurious regression, cointegration)
- Stock & Watson: *Introduction to Econometrics* (forecasting with macro data)
- Angrist & Pischke: *Mostly Harmless Econometrics* (why regression is not identification)
