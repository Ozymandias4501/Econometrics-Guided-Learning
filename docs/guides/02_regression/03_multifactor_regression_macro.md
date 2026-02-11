# Guide: 03_multifactor_regression_macro

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/03_multifactor_regression_macro.ipynb`.

Multifactor macro regression uses several macroeconomic indicators simultaneously to predict or explain an outcome like GDP growth, inflation, or recession risk. The central challenge is that macro indicators are fundamentally entangled: they all respond to the business cycle, making it difficult to isolate the contribution of any single variable.

### Key Terms (defined)
- **Macro indicator**: an aggregate economic time series (yield spread, unemployment rate, industrial production, etc.) used as a predictor.
- **Multicollinearity (in macro)**: the near-universal correlation among macro indicators because they co-move with the business cycle. VIF values of 5-20 are typical, not exceptional.
- **Feature selection**: choosing which indicators to include in a regression. In macro, this is driven by economic theory, VIF diagnostics, and out-of-sample performance.
- **In-sample vs out-of-sample**: in-sample $R^2$ measures fit on the data used for estimation; out-of-sample $R^2$ measures forecasting accuracy on new data. The gap between them reveals overfitting.
- **Overfitting**: a model that captures noise in the training sample and forecasts poorly. Particularly dangerous in macro because sample sizes are small (often < 300 quarterly observations).
- **Forecast horizon**: how far ahead you are predicting (1 quarter, 4 quarters, etc.). Different horizons often require different models.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** for the challenges unique to multifactor macro regression.
- For OLS mechanics, assumptions, and VIF formulas, see [Guide 00](00_single_factor_regression_micro.md#technical).

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Choose 3-5 macro features guided by economic theory (e.g., yield spread, unemployment change, industrial production growth).
- Compute the correlation matrix across your chosen features. Identify pairs with $|r| > 0.7$.
- Compute a VIF table. Flag any variable with VIF > 10 and decide whether to drop or combine it.
- Fit a multifactor OLS regression with HAC standard errors. Report coefficients with CI.
- Check coefficient stability: re-estimate on the first half and second half of the sample. Do signs or magnitudes change substantially?
- Compare in-sample $R^2$ to out-of-sample $R^2$ using a walk-forward or expanding-window split. Quantify the gap.

### Alternative Example (Not the Notebook Solution)
```python
# Multifactor macro regression: predict GDP growth from three indicators
# (This is a toy example with simulated data, not the notebook solution.)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

rng = np.random.default_rng(42)
n = 120  # 30 years of quarterly data

# Simulate correlated macro indicators
cycle = np.cumsum(rng.normal(size=n))  # common business cycle factor
yield_spread = 0.6 * cycle + rng.normal(scale=0.8, size=n)
unemp_change = -0.5 * cycle + rng.normal(scale=0.7, size=n)
ip_growth = 0.4 * cycle + rng.normal(scale=0.9, size=n)

gdp_growth = 2.0 + 0.3 * yield_spread - 0.4 * unemp_change + 0.2 * ip_growth \
             + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({
    'gdp_growth': gdp_growth,
    'yield_spread': yield_spread,
    'unemp_change': unemp_change,
    'ip_growth': ip_growth,
})

# 1) Correlation matrix
print(df[['yield_spread', 'unemp_change', 'ip_growth']].corr().round(2))

# 2) VIF table
X = sm.add_constant(df[['yield_spread', 'unemp_change', 'ip_growth']])
vifs = pd.Series(
    [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])],
    index=['yield_spread', 'unemp_change', 'ip_growth'],
)
print('\nVIF:\n', vifs.round(1))

# 3) Fit with HAC SE
res = sm.OLS(df['gdp_growth'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(res.summary())

# 4) Coefficient stability: first half vs second half
mid = n // 2
res_1st = sm.OLS(df['gdp_growth'].iloc[:mid], X.iloc[:mid]).fit()
res_2nd = sm.OLS(df['gdp_growth'].iloc[mid:], X.iloc[mid:]).fit()
print('\nFirst-half coefs:\n', res_1st.params.round(3))
print('Second-half coefs:\n', res_2nd.params.round(3))
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

> **OLS and VIF foundations.** For the OLS derivation, assumptions, standard error formulas, VIF definition, and core diagnostics, see [Guide 00: Single-Factor Regression](00_single_factor_regression_micro.md#technical). This guide builds on that foundation and focuses on the challenges that arise specifically when combining multiple macro indicators in one regression.

### Multifactor Regression with Macro Data: Special Challenges

#### 1) The multicollinearity problem in macro

In microeconomic cross-section data you can sometimes find predictors that vary independently: education and region of birth, for instance, have limited correlation in many samples. Macro time-series data is fundamentally different. Nearly every indicator responds to the same underlying business cycle:

- When the economy expands, unemployment falls, industrial production rises, consumer confidence climbs, and yield spreads widen.
- When the economy contracts, these indicators reverse together.

The result is that pairwise correlations of 0.5-0.8 among standard macro indicators are normal, not pathological. VIF values in the 5-20 range are routine for any multifactor macro model with more than three indicators. This does **not** mean the model is "wrong." It means:

- **Individual coefficient interpretation is unreliable.** The regression cannot reliably separate the yield spread's contribution from the unemployment contribution because they move together in the data. Coefficients may flip sign or change magnitude across sample periods.
- **Prediction can still work.** The combined model may forecast reasonably even though you cannot attribute the forecast to any single input.
- **Standard errors inflate.** Confidence intervals on individual coefficients will be wide, and many individually "insignificant" variables may jointly predict well (use an F-test for joint significance).

This is the central tension in multifactor macro regression: you gain forecasting power by combining indicators, but you lose the ability to tell clean stories about individual coefficients.

**Health economics connection:** Hospital utilization, insurance coverage rates, regional income, and physician supply are all correlated through the same local economic conditions. A multifactor regression of health spending on these variables will face the same entanglement.

#### 2) Feature selection strategies

Adding more macro indicators generally improves in-sample fit but can degrade out-of-sample forecasts. Systematic feature selection is essential.

**Start with economic theory.** Before looking at data, identify which variables *should* predict your outcome based on economic reasoning. For GDP growth forecasting, the yield spread (term structure) has a long theoretical and empirical track record. Unemployment changes proxy for labor-market slack. Industrial production captures real activity. Starting from theory limits the risk of data mining.

**Check VIF and drop obvious redundancies.** After choosing candidate features, compute the VIF table. If two variables are measuring essentially the same thing (e.g., both the unemployment rate and the employment-to-population ratio), keep the one with the clearest theoretical link to your outcome and drop the other. There is no single VIF cutoff that works universally, but VIF > 10 means that more than 90% of that variable's variance is explained by the other predictors -- very little unique information remains.

**Use regularization for systematic selection.** When you have more candidate features than theory can narrow down, ridge or lasso regression provides a disciplined way to shrink or zero out coefficients. See [Guide 05: Regularization](05_regularization_ridge_lasso.md) for details. Ridge is particularly useful in macro because it handles correlated predictors gracefully without forcing hard inclusion/exclusion decisions.

**Use time-aware cross-validation.** This is critical and non-negotiable for macro forecasting. Never use random train/test splits on time-series data. Random splits allow future information to leak into the training set, producing artificially good performance. Always use:
- **Walk-forward (rolling origin) validation:** train on data up to time $t$, forecast time $t+h$, then advance $t$ and repeat.
- **Expanding-window validation:** same idea, but the training set grows over time instead of rolling.

These methods respect the temporal ordering of the data and give honest estimates of forecast accuracy.

#### 3) In-sample vs out-of-sample performance

This is arguably the most important concept for applied macro forecasting, and where the econometric and ML perspectives converge.

**Why high in-sample $R^2$ is almost meaningless for macro forecasting.** With enough predictors, you can always achieve a high in-sample $R^2$ on a short macro sample. Quarterly U.S. GDP data since 1960 gives you roughly 250 observations. A model with 20 indicators will "explain" much of the variation in-sample but will often forecast worse than a simple model with 2-3 well-chosen indicators. This is the **kitchen-sink problem**: throwing in every available indicator improves in-sample fit mechanically but captures noise that does not recur out of sample.

**Measuring out-of-sample performance.** The standard metric is the out-of-sample $R^2_{OOS}$:

$$
R^2_{OOS} = 1 - \frac{\sum_t (y_t - \hat{y}_t)^2}{\sum_t (y_t - \bar{y}_{t-1})^2}
$$

where $\hat{y}_t$ is the model's forecast and $\bar{y}_{t-1}$ is the historical mean (a simple benchmark). A positive $R^2_{OOS}$ means the model beats the "predict the average" benchmark. Many macro models that look excellent in-sample produce $R^2_{OOS}$ near zero or negative.

**Walk-forward validation as the gold standard.** In each step:
1. Estimate the model using data from the beginning through time $t$.
2. Forecast at horizon $h$: predict $y_{t+h}$.
3. Record the forecast error.
4. Advance $t$ by one period and repeat.

This simulates what a real-time forecaster would experience and provides an honest assessment.

**Forecast combination.** A robust finding in the forecasting literature is that averaging forecasts from several simple models often outperforms a single complex model. If you have three candidate specifications (e.g., one with yield spread only, one with unemployment change only, one with both), averaging their forecasts can reduce variance and improve reliability. This is especially valuable in macro where no single model dominates across all time periods.

#### 4) Interpreting coefficients in multifactor macro models

Understanding what you can and cannot say about individual coefficients in a multifactor macro regression is essential for honest reporting.

**What individual coefficients do not mean.** Suppose you estimate:

$$
\widehat{GDP}_t = 1.2 + 0.3 \cdot \text{yield\_spread}_t - 0.4 \cdot \text{unemp\_change}_t + 0.2 \cdot \text{ip\_growth}_t
$$

The coefficient 0.3 on yield spread does **not** mean "a 1-unit increase in the yield spread causes GDP growth to rise by 0.3 percentage points." There are two problems:

1. **Multicollinearity obscures attribution.** If yield spread is correlated at 0.7 with the Fed funds rate (which is not in the model) and at 0.5 with industrial production (which is in the model), the 0.3 coefficient is an unstable compromise. Re-estimate on a different subsample or add one more variable, and the coefficient may change substantially. The yield spread is "sharing credit" with correlated indicators in an arbitrary way.

2. **No causal identification.** Even without multicollinearity, this is an observational time-series regression with no exogenous variation. The coefficient reflects correlation, not causation. Policy conclusions ("if we widen the yield spread, GDP will grow") require a causal design, not a forecasting regression. See [Guide 01: Controls and OVB](01_multifactor_regression_micro_controls.md) for the omitted variable bias framework.

**What the model as a whole can do.** If the model's out-of-sample $R^2$ is positive and stable across evaluation windows, you can say: "This combination of indicators has historically forecast GDP growth better than the historical average." That is a useful statement for forecasting. It does not support stories about which specific indicator "drives" growth.

**Practical guidance for reporting.** When presenting multifactor macro results:
- Report the full model's $R^2_{OOS}$ alongside $R^2$ in-sample. The gap is informative.
- Report coefficient estimates with robust (HAC) standard errors, but flag that individual coefficient interpretation is limited due to multicollinearity.
- If you must discuss individual coefficients, show their sensitivity: re-estimate with different subsets of features and different sample periods. Stable coefficients are more credible.
- Never claim causal effects from a forecasting regression without an identification strategy.

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`) + multicollinearity (`vif_table`)
- `src/macro.py`: GDP + labels (`gdp_growth_*`, `technical_recession_label`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)

### Common Mistakes
- **Using random train/test splits on time-series data (look-ahead bias).** Random splits allow future observations into the training set, producing artificially optimistic results. Always use walk-forward or expanding-window evaluation for macro forecasting.
- **Interpreting individual macro coefficients as if they were causal.** A forecasting regression with correlated indicators cannot support "X causes Y" claims. Report the model's predictive accuracy, not coefficient stories.
- **Adding too many indicators without checking out-of-sample performance.** More features improve in-sample $R^2$ mechanically but often degrade forecasts. Compare your model's $R^2_{OOS}$ against a simple benchmark.
- **Ignoring multicollinearity diagnostics.** Always compute VIF and the correlation matrix before interpreting individual coefficients. VIF > 10 means the coefficient is almost entirely determined by the other predictors.
- **Using naive (non-HAC) standard errors on quarterly macro data.** Macro residuals are typically autocorrelated. HAC standard errors (Newey-West) are the minimum for honest inference.

<a id="summary"></a>
## Summary + Suggested Readings

Multifactor macro regression is a powerful forecasting tool but requires discipline:
- macro indicators are fundamentally correlated through the business cycle, making individual coefficient interpretation unreliable,
- feature selection should start from economic theory and be validated out-of-sample,
- in-sample $R^2$ is almost meaningless without out-of-sample confirmation, and
- forecasting regressions do not support causal claims without identification.

You should now be able to:
- build a multifactor macro regression with appropriate diagnostics (VIF, correlation matrix),
- evaluate it honestly using walk-forward validation,
- report results that distinguish between what the model predicts and what it explains.

Suggested readings:
- Stock & Watson: "Forecasting Using Principal Components from a Large Number of Predictors" (the classic on macro forecasting and dimensionality)
- Wooldridge: Introductory Econometrics, Chapter 10 (time-series regression basics)
- Diebold: "Forecasting in Economics, Business, Finance and Beyond" (practical forecasting methodology)
- Clark & West: "Approximately Normal Tests for Equal Predictive Accuracy in Nested Models" (testing out-of-sample forecast accuracy)
- Angrist & Pischke: Mostly Harmless Econometrics (causal thinking -- why forecasting coefficients are not structural)
