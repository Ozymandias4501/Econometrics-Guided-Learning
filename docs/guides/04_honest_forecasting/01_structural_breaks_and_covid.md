# Guide: 01_structural_breaks_and_covid

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/04_honest_forecasting/01_structural_breaks_and_covid.ipynb`.

It assumes you have done [Guide 00](00_walk_forward_backtest.md) and have a walk-forward evaluation already set up. This guide is what you reach for when the rolling-error plot from the previous notebook tells you the model failed badly during a specific period.

### Key Terms (defined)
- **Structural break**: a point in time after which the parameters of a regression model change. Coefficients, intercept, or noise variance can all break.
- **Chow test**: an F-test for whether OLS coefficients are equal across two sub-samples split at a known break date.
- **Quandt-Andrews test**: extension of the Chow test for an *unknown* break date. Out of scope for this guide.
- **CUSUM test**: tracks cumulative recursive residuals over time. Useful as a visual diagnostic for breaks at unknown dates.
- **Crisis dummy**: an indicator variable that flags break-period observations. Usually 1 during the crisis quarters and 0 otherwise.
- **Regime-switching**: explicit modeling of two or more regimes with a Markov process governing transitions. Out of scope here; mentioned for vocabulary.

### How To Read This Guide
- Use **Step-by-Step** for the notebook checklist and the strategy decision tree.
- Use **Technical Explanations** for the math of the Chow test and the practical tradeoffs of the three mitigation strategies.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- [ ] Plot the target with vertical lines at hypothesized break dates (2008-Q3, 2020-Q1).
- [ ] Plot OLS residuals from a single full-sample fit; look for clusters of large residuals around the candidate dates.
- [ ] Run `chow_test` at each candidate date. Note which p-values are small.
- [ ] Fit OLS separately on pre-break and post-break sub-samples; compare coefficients side by side.
- [ ] Train a model on pre-break data only; evaluate on post-break data; observe how badly it fails.
- [ ] Apply each of the three mitigation strategies (crisis dummy, drop period, post-only refit).
- [ ] Pick a default and write down what assumption it commits you to.

### Strategy Decision Tree

If walk-forward error spikes during a specific window:

1. **Is it a few quarters or a multi-year regime change?**
   - Few quarters → crisis dummy.
   - Multi-year → consider refitting on post-break data.

2. **Do you have enough post-break data to refit?**
   - Less than ~30 observations → don't refit on post-break only; you'll be parameter-starved.
   - 30+ observations → refit on post-break is on the table.

3. **Is your audience a forecasting consumer or a research peer?**
   - Forecasting consumer → crisis dummy is usually the most defensible (keeps all the data, makes the break visible).
   - Research peer → drop the break window, report the analysis, and address the COVID period as a separate case study.

4. **Are the coefficient changes consistent with theory?**
   - Yes → a structural break may be the right interpretation; consider regime-switching.
   - No (random sign flips, etc.) → the 'break' may be an artifact of small post-break samples; treat with skepticism.

### Alternative Example: CUSUM for Unknown Break Dates

When you don't have a strong prior on *where* the break is, CUSUM is the right tool.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import recursive_olsresiduals
import matplotlib.pyplot as plt

X = sm.add_constant(panel[x_cols], has_constant='add')
y = panel[y_col]
res = sm.OLS(y, X).fit()
recres, sigma, mu, p_low, p_up, p_low2, p_up2 = recursive_olsresiduals(res)

cusum = np.cumsum(recres) / sigma
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(panel.index, cusum)
ax.fill_between(panel.index, p_low, p_up, alpha=0.2, label='5% bounds')
ax.set_title('CUSUM of recursive residuals')
ax.legend()
```

If the CUSUM line crosses outside the bands, that's evidence of instability. Where it crosses tells you roughly when. (CUSUM doesn't pinpoint a single date; treat it as a diagnostic, not a precise estimator.)

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### The Chow test, mechanically

Set up the regression:

$$
y_t = x_t' \beta + u_t, \quad t = 1, \dots, n.
$$

A candidate break at time $t^*$ partitions the sample into two sub-samples of sizes $n_1$ and $n_2$ with $n = n_1 + n_2$.

Three OLS fits:
- **Pooled** (one model on all data): residual sum of squares $RSS_P$.
- **Pre-break** ($t < t^*$): $RSS_1$.
- **Post-break** ($t \geq t^*$): $RSS_2$.

Under $H_0$ (coefficients equal across sub-samples), the Chow F-statistic is:

$$
F = \frac{(RSS_P - (RSS_1 + RSS_2)) / k}{(RSS_1 + RSS_2) / (n - 2k)}
$$

where $k$ is the number of regression parameters (including the constant). Under $H_0$, $F \sim F_{k, n-2k}$. Reject for large $F$ (small p).

**Assumptions:**
- The break date $t^*$ is specified *before* looking at the data.
- The error variance is constant across regimes (otherwise use a Wald test with HC-robust SE).
- The model specification is the same in both regimes — only the coefficients differ.

**What it does not do:**
- Tell you which coefficients changed.
- Tell you when the break happened (you supply the date).
- Account for the data-mining cost of trying many candidate dates.

### Three mitigation strategies, with tradeoffs

**Strategy 1: Crisis dummy** introduces $D_t \in \{0, 1\}$ for break-period observations and adds it as a regressor:

$$
y_t = \alpha + \beta x_t + \delta D_t + u_t.
$$

This soaks up the **level shift** during the crisis. It does **not** absorb a slope change. If the relationship between $y$ and $x$ changed, you need an interaction $D_t \cdot x_t$ as well, which doubles the parameter count and erodes statistical power.

**When to use:** the break is short (a few quarters), the level shift is the obvious feature, and you want to keep all the data.

**Strategy 2: Drop the break period** simply removes the offending observations:

$$
\text{Fit on } \{t : t \notin \text{break window}\}.
$$

This is the most honest move when you believe the model has nothing useful to say about the break period. Coefficient estimates are not distorted by the break. The cost is throwing out potentially informative observations and pretending you cannot answer questions about that window.

**When to use:** the break period is contaminating coefficients, and the audience accepts that you don't model that period.

**Strategy 3: Refit on post-break data only** uses only observations after the break.

$$
\text{Fit on } \{t : t \geq t^*\}.
$$

This commits you to the position that the structural change is permanent and that pre-break data is no longer informative. It also halves (or worse) your sample size, which can make coefficient estimates noisy.

**When to use:** you have a strong prior that the regime change is durable (post-1971 dollar dynamics, post-2008 monetary policy regime), and you have enough post-break data.

### What the COVID episode taught macro forecasters

The Q2 2020 GDP collapse was a 5-sigma event under any pre-COVID model. Three lessons that have stuck:

1. **Crisis dummies for the 2020 quarters are now standard practice** in central-bank forecasting models. The dummy doesn't pretend the model handles COVID; it surgically excises the influence of those quarters.
2. **Some models needed regime-switching frameworks** because the relationships themselves shifted. Inflation–unemployment dynamics in 2021–2022 looked nothing like 2010–2019.
3. **Backtests now report metrics with and without the COVID quarters separately.** A model evaluated only on 1990–2019 may look great and still be useless going forward.

### Project Code Map
- `src/econometrics.py`: `chow_test` (this notebook).
- `src/econometrics.py`: `fit_ols` for sub-sample fitting.
- `src/evaluation.py`: `regression_metrics` to score the pre/post forecasts.

### Common Mistakes
- Searching over many candidate break dates and reporting the smallest p-value as if you tested only one. Use Quandt-Andrews if you really want unknown-break inference.
- Using a crisis dummy and concluding 'the model handles COVID now.' The dummy makes the model fit better in-sample; it does not give the model new ability to forecast the next crisis.
- Refitting on post-break data alone with only 5 observations and reporting the coefficient as if it were precise.
- Confusing 'large residual' with 'structural break.' One outlier is a noise event; sustained pattern change is a break.

<a id="summary"></a>
## Summary + Suggested Readings

After this notebook you should be able to:

- explain the difference between an outlier and a structural break,
- run a Chow test correctly (including being honest about pre-specifying the date),
- compare pre- and post-break OLS coefficients and read the change,
- demonstrate how a pre-break model fails on post-break data,
- apply at least one mitigation strategy and write down what it assumes.

**Companion guides:**
- [Guide 00 — Walk-forward backtest](00_walk_forward_backtest.md): the rolling-error plot is what surfaces breaks worth investigating.
- [Guide 01b/00 — Stationarity](../01b_time_series_diagnostics/00_stationarity_and_unit_roots.md): rules out the 'pseudo-break' you can get from non-stationary data.

**Suggested readings:**
- Chow (1960), "Tests of Equality Between Sets of Coefficients in Two Linear Regressions" — the original.
- Andrews (1993), "Tests for Parameter Instability and Structural Change with Unknown Change Point" — the modern sup-F approach.
- Hamilton, *Time Series Analysis*, Ch. 22 (regime-switching).
- Stock and Watson (2003), "Has the Business Cycle Changed and Why?" NBER macro annual — accessible primer on the Great Moderation break.
- Federal Reserve Bank of Minneapolis Q1 2021 working papers on macro forecasting through COVID.
