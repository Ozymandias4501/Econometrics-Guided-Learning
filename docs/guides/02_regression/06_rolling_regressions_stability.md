# Guide: 06_rolling_regressions_stability

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/06_rolling_regressions_stability.ipynb`.

Economic relationships are not permanent. A regression estimated over the full sample assumes the coefficients are constant from the first observation to the last. Rolling regressions test that assumption by re-estimating the model on moving windows, revealing when and how coefficients drift. This guide covers both the rolling regression technique and the broader question of structural breaks and parameter instability.

### Key Terms (defined)
- **Rolling window**: a fixed-length subsample that slides forward through time. At each position, you re-estimate the regression using only the observations in the window.
- **Structural break**: a discrete shift in the data-generating process -- coefficients change at a specific date (or narrow period) rather than drifting gradually.
- **Parameter instability**: the general phenomenon where regression coefficients are not constant over time, whether due to gradual drift or discrete breaks.
- **Regime change**: a shift in the economic environment (e.g., monetary policy regime, regulatory change) that alters the relationship between variables.
- **Chow test**: a formal hypothesis test for whether regression coefficients are the same in two subsamples split at a candidate break date.
- **CUSUM (cumulative sum)**: a test that tracks cumulative forecast errors over time; a departure from the expected range signals instability.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** for rolling regression mechanics and structural break analysis.
- For OLS derivation, assumptions, and standard error formulas, see [Guide 00](00_single_factor_regression_micro.md#technical).

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Choose a rolling window length $W$ with justification (e.g., 40 quarters for macro data). Document why this length balances precision and sensitivity.
- Fit a rolling regression of your outcome on one or more predictors. Store the coefficient estimates $\hat\beta_t$ for each window.
- Plot the coefficient path over time with confidence interval bands. Add recession shading or known event markers.
- Identify periods where the coefficient changes sign or magnitude substantially. Propose an economic explanation.
- Compare at least two different window lengths (e.g., 20 and 60 quarters). Assess whether the drift pattern is robust to window choice.
- (Optional) Conduct a Chow test at a candidate structural break date and interpret the result.

### Alternative Example (Not the Notebook Solution)
```python
# Rolling regression of GDP growth on yield spread,
# showing coefficient drift around recessions.
# (Simulated data, not the notebook solution.)
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(7)
n = 160  # 40 years of quarterly data

# Simulate a yield spread with a regime shift at t=80 (mid-sample)
yield_spread = rng.normal(loc=1.5, scale=0.8, size=n)

# True coefficient changes: 0.5 in first half, 0.1 in second half
beta_true = np.where(np.arange(n) < 80, 0.5, 0.1)
gdp_growth = 2.0 + beta_true * yield_spread + rng.normal(scale=0.8, size=n)

df = pd.DataFrame({
    'gdp_growth': gdp_growth,
    'yield_spread': yield_spread,
}, index=pd.date_range('1984-01-01', periods=n, freq='QS'))

# Rolling regression with W=40
W = 40
rolling_beta = []
rolling_ci_lo = []
rolling_ci_hi = []

for t in range(W, n):
    window = df.iloc[t-W:t]
    X = sm.add_constant(window['yield_spread'])
    res = sm.OLS(window['gdp_growth'], X).fit()
    rolling_beta.append(res.params['yield_spread'])
    ci = res.conf_int().loc['yield_spread']
    rolling_ci_lo.append(ci[0])
    rolling_ci_hi.append(ci[1])

roll_df = pd.DataFrame({
    'beta': rolling_beta,
    'ci_lo': rolling_ci_lo,
    'ci_hi': rolling_ci_hi,
}, index=df.index[W:])

# Plot: you would see the coefficient decline from ~0.5 to ~0.1 around 2004
print(roll_df.describe())
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

> **OLS foundations.** For the OLS derivation, assumptions, standard errors, and core diagnostics, see [Guide 00: Single-Factor Regression](00_single_factor_regression_micro.md#technical). This guide focuses on what happens when those coefficients are not constant over time.

### Deep Dive: Rolling regressions — coefficient stability over time

Rolling regressions are a simple tool to see whether relationships drift across macro regimes.

#### 1) Intuition (plain English)

In macro data, relationships change:
- policy regimes shift,
- financial structure changes,
- measurement changes.

A single "full-sample" regression can hide instability.
Rolling regressions estimate coefficients repeatedly on moving windows to reveal drift.

#### 2) Notation + setup (define symbols)

Let window length be $W$ (in periods).
For each end time $t \ge W$, define the window:
$$
\{t-W+1, \dots, t\}.
$$

Estimate:
$$
\hat\beta_t = \arg\min_{\beta} \sum_{s=t-W+1}^{t} (y_s - x_s'\beta)^2.
$$

**What each term means**
- $\hat\beta_t$ is the coefficient estimate using only the most recent $W$ observations ending at $t$.

#### 3) Assumptions and caveats

Rolling regressions assume:
- the relationship is approximately stable within each window,
- $W$ is large enough for estimation but small enough to detect changes.

Inference caveat:
- consecutive windows overlap heavily, so estimates are correlated.

#### 4) Mechanics (practical use)

**Choosing window length $W$:** The window must be long enough for reliable estimation (rule of thumb: at least 5-10 observations per coefficient) but short enough to detect structural change. A common starting point for quarterly macro data is 40 quarters (~10 years), which balances estimation precision with sensitivity to regime shifts. Always try multiple window lengths (e.g., 20, 40, 60 quarters) to check whether your conclusions are sensitive to this choice. If the drift pattern persists across different $W$, it is more credible.

Typical workflow:
1) choose a window (e.g., 40 quarters ≈ 10 years),
2) estimate $\hat\beta_t$ for each $t$,
3) plot $\hat\beta_t$ with CI bands,
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
- [ ] Compare two window lengths and explain the bias-variance trade-off.
- [ ] Identify one period where the coefficient changes sign and propose a macro explanation.

### Deep Dive: Structural Breaks and Parameter Instability

Rolling regressions visualize instability; formal tests quantify it. This section covers why coefficients drift, how to test for discrete breaks, and what instability means for your model.

#### 1) Why coefficients drift

Economic relationships are products of institutions, policies, technologies, and behavioral norms -- all of which evolve. Common sources of instability include:

- **Policy regime shifts.** A change in monetary policy framework (e.g., the Volcker disinflation of the early 1980s, the shift to inflation targeting, the zero lower bound era after 2008) can alter the relationship between interest rates and output. The yield spread may predict recessions differently when the Fed operates under different rules.
- **Structural transformation.** The shift from manufacturing to services, the rise of the gig economy, or the growth of employer-sponsored health insurance all change how macro indicators relate to outcomes. A regression of health spending on GDP growth estimated in the 1970s may not apply in the 2010s.
- **Financial innovation.** New financial instruments (securitization, derivatives) change how credit conditions transmit to the real economy, altering regression coefficients that involve financial variables.
- **Measurement changes.** Statistical agencies periodically revise definitions and methods (e.g., CPI methodology changes, GDP accounting revisions), which can create artificial breaks in estimated relationships.

The key insight is that a "constant coefficient" assumption is a strong claim for any macro regression spanning multiple decades. Rolling regressions are one way to check it; formal break tests are another.

#### 2) The Chow test: testing for a break at a known date

The Chow test asks: "Are the regression coefficients the same before and after date $\tau$?"

**Setup.** Split the sample into two subsamples at candidate break date $\tau$:
- Subsample 1: observations $1, \dots, \tau$ (before the break).
- Subsample 2: observations $\tau+1, \dots, n$ (after the break).

Estimate three regressions:
- Full sample: get $SSR_{full}$ (sum of squared residuals).
- Subsample 1: get $SSR_1$.
- Subsample 2: get $SSR_2$.

**Test statistic:**

$$
F = \frac{(SSR_{full} - SSR_1 - SSR_2) / K}{(SSR_1 + SSR_2) / (n - 2K)}
$$

where $K$ is the number of parameters (including intercept). Under the null of no break, $F \sim F(K, n-2K)$.

**Interpretation.** A large $F$ (small p-value) means the coefficients are significantly different across the two subsamples. The regression relationship has changed.

**Important limitations:**
- The Chow test requires you to specify the break date $\tau$ in advance. If you choose $\tau$ by looking at the data (e.g., picking the date where rolling coefficients change most), the test is invalid because you have used the data twice.
- Each subsample must have enough observations relative to the number of parameters.
- The test assumes homoskedastic, non-autocorrelated errors within each subsample. For macro data, use a robust variant or complement with other diagnostics.

#### 3) CUSUM: detecting breaks without specifying a date

The CUSUM (cumulative sum) test does not require you to pre-specify a break date. It works by tracking cumulative one-step-ahead forecast errors as you move through the sample:

1. Estimate the model on the first $K$ observations (minimum for estimation).
2. Forecast observation $K+1$ and record the error.
3. Add observation $K+1$ to the estimation sample, forecast $K+2$, and so on.
4. Cumulate the (standardized) forecast errors.

Under the null of stable coefficients, the cumulative sum should stay within bounds (typically plotted as a pair of straight lines diverging from zero). If the CUSUM crosses the boundary, it signals a structural break.

CUSUM is complementary to rolling regressions: rolling coefficients show *where* instability occurs visually, while CUSUM provides a formal significance assessment.

#### 4) What instability means for your model

If you find evidence of structural breaks or drifting coefficients, the implications depend on your goal:

**For forecasting:** A full-sample regression averages over different regimes. If the most recent regime differs from earlier ones, the full-sample coefficients may produce poor forecasts. Options include:
- Use only recent data (shorter estimation window), accepting higher variance.
- Fit separate models for different eras and use the most recent model for forecasting.
- Use models that allow for time-varying parameters (e.g., state-space models, Bayesian time-varying parameter regressions).

**For inference:** If the relationship between $X$ and $Y$ has changed, there is no single "true" coefficient to estimate. A full-sample estimate is a weighted average of the different regimes, which may not correspond to the current relationship. Report the instability explicitly rather than pretending the coefficient is constant.

**For policy analysis:** Parameter instability is a fundamental challenge for policy evaluation. If the relationship between a policy instrument and an outcome has shifted (the Lucas critique in macro), historical regressions cannot reliably predict the effects of future policy changes.

#### 5) Health economics example

The relationship between hospital spending per capita and patient outcomes (e.g., 30-day mortality) may shift after major policy changes:
- **Pre-ACA vs post-ACA (2010):** The Affordable Care Act expanded insurance coverage and changed reimbursement incentives. A regression of outcomes on spending estimated before 2010 may not apply after.
- **Medicare payment reforms (e.g., bundled payments, value-based purchasing):** These change the incentive structure, potentially altering how marginal spending translates to outcomes.
- **COVID-19 pandemic:** The 2020-2021 period represents a massive structural break in nearly all health and economic relationships.

A rolling regression of outcomes on spending would reveal whether the coefficient is stable across these policy eras. A Chow test at the ACA implementation date could formally test for a break.

#### Exercises

- [ ] Pick a candidate structural break date (e.g., the start of a recession) and conduct a Chow test. Interpret the F-statistic and p-value.
- [ ] Compare the Chow test result to the rolling coefficient plot around the same date. Are they consistent?
- [ ] Estimate separate regressions before and after the break. How do the coefficients differ in magnitude and sign?
- [ ] Discuss: if you find a structural break, should you use the full sample or only the post-break sample for forecasting? What are the trade-offs?

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`) + multicollinearity (`vif_table`)
- `src/macro.py`: GDP + labels (`gdp_growth_*`, `technical_recession_label`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)

### Common Mistakes
- **Using a single full-sample regression without checking for structural breaks.** A full-sample regression assumes constant coefficients. If the relationship has changed, the full-sample estimate is an unreliable average. Always plot rolling coefficients or run a formal break test.
- **Choosing the rolling window length arbitrarily.** The window length $W$ involves a bias-variance trade-off. Always try multiple values and check whether your conclusions are robust.
- **Picking a break date by looking at the data, then running a Chow test at that date.** This invalidates the test because you used the data to choose where to test. Use CUSUM or a sup-F test if you do not have a pre-specified break date.
- **Ignoring overlap in rolling estimates.** Consecutive rolling windows share most of their data, so rolling coefficient estimates are highly correlated. Do not interpret small wiggles as meaningful -- focus on persistent, large-magnitude shifts.
- **Treating instability as a problem to "fix" rather than a finding to report.** If coefficients drift, that is economically informative. Report it and discuss what regime changes might explain it.

<a id="summary"></a>
## Summary + Suggested Readings

Rolling regressions and structural break tests address a fundamental question: is the relationship you estimated stable over time?

You should now be able to:
- implement rolling regressions with justified window lengths and interpret the coefficient path,
- identify periods of instability and connect them to economic events or policy changes,
- understand the Chow test for known break dates and CUSUM for unknown break dates,
- decide whether to use the full sample, a subsample, or a time-varying model when instability is detected.

Key takeaway: a regression that ignores parameter instability may produce misleading coefficient estimates and poor forecasts. Checking for stability is not optional -- it is a core diagnostic.

Suggested readings:
- Stock & Watson: Introduction to Econometrics, Chapter 14 (time-series regression, structural breaks)
- Hansen: "Tests for Parameter Instability in Regressions with I(1) Processes" (formal break tests)
- Andrews: "Tests for Parameter Instability and Structural Change With Unknown Change Point" (sup-F test for unknown break dates)
- Wooldridge: Introductory Econometrics, Chapter 10 (time-series regression basics)
- Zeileis et al.: "Testing and Dating of Structural Changes in Practice" (practical implementation in R/Python)
