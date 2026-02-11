# Technical Deep Dive: FRED + Macro ML (Ultra-Detailed)

This document is the technical reference for the tutorial. It explains the Python patterns, the data processing pipeline, and (mostly) the statistics/ML used for training and interpreting models on economic time-series.

Note: the project now also includes **one guide per notebook** under `docs/guides/` and a navigation hub at `docs/index.md`. Prefer those guides for notebook-specific deep dives; treat this file as an optional monolithic reference.

## Table of Contents
- [1. How To Use This Guide](#toc-how-to-use)
- [2. Notebook Map (What This Explains)](#toc-notebook-map)
- [3. Data and Preprocessing (FRED, Caching, Monthly Alignment)](#toc-data-prep)
- [4. Time-Series Modeling Framing (Forecasting vs Nowcasting, Leakage)](#toc-framing)
- [5. Feature Engineering (Lags, Diffs, Rolling Stats)](#toc-features)
- [6. Regression Deep Dive (OLS, Assumptions, Regularization)](#toc-regression)
- [7. Classification Deep Dive (Logistic Regression, Thresholds, ROC/PR)](#toc-classification)
- [8. Evaluation for Time Series (Splits, Walk-Forward, Metrics)](#toc-eval)
- [9. Interpretation and Diagnostics (Coefficients, Residuals, Importance)](#toc-interpret)
- [10. Macro-Specific Pitfalls (Revisions, Structural Breaks, Nonstationarity)](#toc-macro)
- [11. Practical Use Cases and Extensions](#toc-extensions)
- [12. Quick Reference Glossary (Short)](#toc-glossary)

---

<a id="toc-how-to-use"></a>
## 1) How To Use This Guide

This repo is designed to be hands-on. The notebooks are intentionally TODO-driven, and each ends with a collapsed **Solutions (Reference)** section for self-checking. This guide is where the deeper explanations live.

Recommended workflow:
1. Open the notebook for the step you are on.
2. When you hit a TODO, try to implement it.
3. If you get stuck, search this guide for the matching section.
4. If you can make it work but do not understand *why*, read the interpretation sections.

A key theme: macro data is time-series data. Many "standard" ML defaults break if you ignore time ordering.

---

<a id="toc-notebook-map"></a>
## 2) Notebook Map (What This Explains)

- `notebooks/00_foundations/00_setup.ipynb`: environment, paths, using sample vs API data.
- `notebooks/00_foundations/01_time_series_basics.ipynb`: resampling, lags, rolling windows, leakage intuition.
- `notebooks/00_foundations/02_stats_basics_for_ml.ipynb`: correlation vs causation, multicollinearity, bias/variance.
- `notebooks/01_data/00_fred_api_and_caching.ipynb`: calling an API, caching, converting JSON -> DataFrame.
- `notebooks/01_data/01_build_macro_monthly_panel.ipynb`: frequency alignment (daily/monthly) into a clean monthly panel.
- `notebooks/01_data/02_gdp_growth_and_recession_label.ipynb`: GDP growth math + technical recession label (computed).
- `notebooks/01_data/03_build_macro_quarterly_features.ipynb`: monthly -> quarterly aggregation + lagged macro features.
- `notebooks/02_regression/*`: OLS, robust SE (HC3/HAC), ridge/lasso, rolling regressions.
- `notebooks/03_classification/*`: recession classifiers, calibration, trees, walk-forward validation.
- `notebooks/04_unsupervised/*`: PCA factors, clustering regimes, anomaly detection.
- `notebooks/05_model_ops/*`: reproducible CLI runs and saved artifacts.
- `notebooks/08_capstone/*`: end-to-end project output + report + dashboard.

---

<a id="toc-data-prep"></a>
## 3) Data and Preprocessing (FRED, Caching, Monthly Alignment)

### Key Terms (with detail)

**API (Application Programming Interface)**
- An API is a contract: "If you call this endpoint with these parameters, you will get data back in this structure." The entire point is predictability.
- In practice: you build a thin client (like `src/fred_api.py`) that translates Python calls into HTTP requests.

**HTTP**
- The protocol that carries your request to the server and the server's response back.
- Your code typically does: `requests.get(url, params=..., timeout=...)`.

**Endpoint**
- A specific path under a base URL.
- In FRED, `series/observations` is an endpoint.

**JSON**
- A text format that looks like nested dictionaries/lists.
- FRED returns values as strings; you must cast them (and handle missing values like `"."`).

**Schema**
- The expected structure of the JSON payload.
- Why it matters: if you depend on a field name (like `observations`), schema changes break code.

**Cache**
- A local copy of data to avoid repeated requests.
- Caching is not just speed. It is reproducibility: you can re-run the same analysis without the API changing under you.

### 3.1 Why "Raw" vs "Processed" Data?
- `data/raw/` holds cached JSON responses. This is your closest representation of "what the API returned".
- `data/processed/` holds tabular CSVs. These are designed for analysis, ML, and reuse.

The conceptual separation matters because debugging is easier:
- If a model looks wrong, you can trace it back to features.
- If features look wrong, you can trace back to the raw panel.
- If panel looks wrong, you can trace back to the API payload.

### 3.2 Converting JSON -> DataFrame Correctly
FRED returns `date` and `value` as strings.

Typical conversion steps:
1. Parse `date` into a `datetime` type.
2. Convert `value` to numeric.
3. Coerce placeholders like `"."` to missing (`NaN`).
4. Sort by date and set the index.

Why this matters:
- `DatetimeIndex` unlocks resampling.
- Numeric dtypes unlock vectorized operations and scikit-learn models.

### 3.3 Missing Values: What They Mean
Missing values can happen because:
- The series began later than others.
- The series is reported at a different frequency.
- There are gaps or discontinued series.

There is no universally correct fix. Common strategies:
- Forward fill (reasonable for slow-moving monthly levels).
- Drop rows (simplest, but can throw away a lot of early history).
- Add missingness indicators (advanced; can help if missingness is informative).

### 3.4 Monthly Alignment: Resampling and Forward Fill
**Resampling** converts frequency.
- Example: daily -> monthly.

Our helper uses something like:
- `.resample('M').last()`

Interpretation:
- You are saying: "For each month, represent the month by its final observation."

**Forward fill** (`ffill`) means:
- "If the value is missing this month, assume it is unchanged from the last available month."

Risks:
- It can hide within-month movements for higher-frequency series.
- It can create artificial stability.

When forward fill is defensible:
- The indicator truly only updates monthly.
- You are explicitly modeling at monthly frequency.

---

<a id="toc-framing"></a>
## 4) Time-Series Modeling Framing (Forecasting vs Nowcasting, Leakage)

### Key Terms (with detail)

**Time series**
- A sequence ordered by time.
- The order is not cosmetic; it encodes causality constraints (you cannot use the future to predict the past).

**Forecasting vs Nowcasting**
- Forecasting: predict a future value (e.g., next month's inflation).
- Nowcasting: estimate "right now" using partial or high-frequency signals.

**Leakage**
- When your features include information that would not be available at prediction time.
- Leakage produces unrealistically good test performance.

### 4.1 The Supervised Learning Framing
We aim to learn a mapping:
- features at time t -> target at time t+h

Where:
- t is a month
- h is the horizon (1 month ahead by default)

### 4.2 Why Random Splits Are Usually Wrong Here
Random train/test splits mix future and past.

If your test contains dates that occur *before* some training dates, your model has effectively "seen the future" during training. Even if the leakage is indirect, it inflates performance.

Time-based split:
- Train: earlier portion
- Test: later portion

### 4.3 What You Are Actually Predicting
A critical macro concept: prediction is not causality.

Your model can be useful for forecasting even if:
- features are correlated, not causal
- relationships are unstable

But you must interpret coefficients carefully, because macro features are highly correlated.

---

<a id="toc-features"></a>
## 5) Feature Engineering (Lags, Diffs, Rolling Stats)

This project uses mostly "classical" time-series features that keep interpretation straightforward.

### Key Terms (with detail)

**Feature**
- A numeric input to a model.
- In time series, features are often transformations of past values.

**Lag**
- A feature that equals a past value.
- Example: `UNRATE_lag3` is unemployment 3 months ago.

**Difference (diff)**
- A feature that equals the change in a value over time.
- Example: `FEDFUNDS_diff1` is the month-over-month change.

**Rolling window**
- A moving slice used to compute local statistics.
- Example: 12-month rolling mean.

### 5.1 Why Lags Are the Default Baseline
Macro effects often appear with delays:
- policy changes take time to propagate
- labor markets adjust slowly

Lags let the model learn delayed relationships.

### 5.2 Why Diffs Help With Trends
Many macro series trend.
- Trending levels can cause the model to learn spurious relationships.

Differencing helps by focusing on changes rather than levels.

Important nuance:
- Differencing reduces "level" information.
- Sometimes level matters (e.g., high interest rate regimes).

This is why the tutorial includes both levels and diffs, then relies on regularization.

### 5.3 Rolling Mean and Rolling Std
Rolling mean:
- captures local trend / regime level

Rolling std (volatility proxy):
- captures stability vs turbulence

Why macro volatility matters:
- model relationships often shift during high-volatility regimes
- volatility can signal uncertainty

### 5.4 Feature Explosion and the Need for Regularization
If you have:
- 6 base series
- 3 lags
- diffs
- rolling mean/std across 3 windows

You quickly create dozens of features. Many will be correlated.

That is why ridge/lasso are important here.

---

<a id="toc-regression"></a>
## 6) Regression Deep Dive (OLS, Assumptions, Regularization)

Regression is used when the target is continuous (like inflation in percent).

### Key Terms (with detail)

**Regression**
- Predict a real number.

**Design matrix (X)**
- The table of features arranged as rows (samples) x columns (features).

**OLS (Ordinary Least Squares)**
- The standard linear regression objective: minimize squared errors.

**Residual**
- `residual = y_true - y_pred`

**Heteroskedasticity**
- When residual variance changes over time or with the level of y.
- Common in macro series (crises have larger errors).

**Autocorrelation**
- Residuals are correlated over time.
- Violates common textbook assumptions and makes naive inference unreliable.

### 6.1 Linear Regression: The Model
Linear regression assumes:
- y = b0 + b1*x1 + ... + bp*xp + error

Where:
- b0 is the intercept
- bi are coefficients

### 6.2 OLS Objective
OLS chooses coefficients to minimize:
- sum_t (y_t - yhat_t)^2

Why squared error?
- it is smooth and easy to optimize
- it penalizes large errors more strongly

### 6.3 Interpreting Coefficients (Carefully)
Coefficient interpretation depends on scale:
- If `x` is in dollars and `y` is in percent, the coefficient mixes units.

Two useful interpretations:
1. Raw coefficient: change in y per 1 unit of x.
2. Standardized coefficient (after scaling): change in y per 1 standard deviation of x.

Macro warning:
- correlated indicators can make coefficients unstable (multicollinearity).

### 6.4 Bias-Variance Tradeoff
- High-variance models fit noise (overfit).
- High-bias models miss signal (underfit).

Regularization pushes you toward higher bias / lower variance.

### 6.5 Ridge and Lasso: Regularization
Regularization adds a penalty term.

Ridge (L2) minimizes:
- SSE + alpha * sum(b_i^2)

Lasso (L1) minimizes:
- SSE + alpha * sum(|b_i|)

Intuition:
- Ridge shrinks coefficients smoothly.
- Lasso can set coefficients to zero (feature selection).

### 6.6 Why Scaling Is Non-Negotiable for Regularization
If one feature is measured in thousands and another in decimals, the penalty impacts them differently.

Standard scaling:
- subtract mean
- divide by standard deviation

Effect:
- penalty treats coefficients more fairly
- coefficient magnitudes become comparable

### 6.7 Metrics: MAE, RMSE, R^2
**MAE** (Mean Absolute Error)
- average absolute error
- robust-ish to outliers compared to RMSE

**RMSE** (Root Mean Squared Error)
- penalizes large errors strongly
- highlights crisis-period blowups

**R^2**
- proportion of variance explained
- can be misleading in time-series with regime shifts

---

<a id="toc-classification"></a>
## 7) Classification Deep Dive (Logistic Regression, Thresholds, ROC/PR)

Classification is used when the target is categorical (like "inflation up" vs "not up").

### Key Terms (with detail)

**Classification**
- Predict a discrete label.

**Log-odds**
- `log(p / (1-p))`, where p is probability.

**Sigmoid / logistic function**
- maps any real number to (0,1)

**Calibration**
- whether predicted probabilities match observed frequencies

**Imbalanced classes**
- one class occurs much more often than another

### 7.1 Logistic Regression Model
Logistic regression models probability:
- p(y=1 | x) = sigmoid(b0 + b1*x1 + ...)

Sigmoid:
- sigmoid(z) = 1 / (1 + exp(-z))

Interpretation:
- coefficients add linearly to z
- z is then squashed into a probability

### 7.2 Coefficients as Odds Multipliers
A 1-unit increase in feature i changes log-odds by b_i.

Odds are:
- odds = p / (1-p)

So exp(b_i) is an odds multiplier (holding other features fixed).

### 7.3 Thresholding: Turning Probabilities Into Labels
You typically choose a threshold:
- predict 1 if p >= 0.5

But 0.5 is not special. It is a default.

Threshold choice depends on:
- cost of false positives vs false negatives
- class imbalance

### 7.4 Confusion Matrix
Counts:
- true positives (TP)
- false positives (FP)
- true negatives (TN)
- false negatives (FN)

This is the foundation for most classification metrics.

### 7.5 ROC Curve and AUC
ROC curve plots:
- TPR = TP / (TP + FN)
- FPR = FP / (FP + TN)

AUC summarizes ranking quality:
- 1.0 is perfect
- 0.5 is random guessing

Macro note:
- if positives are rare, ROC-AUC can look decent even when precision is poor.

### 7.6 Precision/Recall and Why You May Prefer It
Precision:
- TP / (TP + FP)

Recall:
- TP / (TP + FN)

If you care about "when I say up, I want to be right", you care about precision.
If you care about "catch most ups", you care about recall.

---

<a id="toc-eval"></a>
## 8) Evaluation for Time Series (Splits, Walk-Forward, Metrics)

### Key Terms (with detail)

**Holdout test set**
- A final period you do not touch during training/tuning.

**Walk-forward validation**
- A repeated evaluation where you train on past data and test on the next chunk.

**Baseline**
- A simple model you must beat to justify complexity.

### 8.1 Why One Split Is Not Enough
A single train/test split can be fragile:
- performance depends on which years ended up in the test set
- macro regimes differ (low inflation era vs high inflation era)

Walk-forward helps you see stability across time.

### 8.2 Baselines You Should Try
Regression baselines:
- predict the last observed value (persistence)
- predict rolling mean

Classification baselines:
- always predict the majority class

If your model cannot beat baselines, it may not be learning signal.

### 8.3 Hyperparameter Tuning Without Cheating
Rule of thumb:
- tune on a validation set
- evaluate once on the final test set

Common mistake:
- repeatedly tuning based on test performance (test leakage).

---

<a id="toc-interpret"></a>
## 9) Interpretation and Diagnostics (Coefficients, Residuals, Importance)

### Key Terms (with detail)

**Interpretability**
- the ability to explain why the model made a prediction.

**Multicollinearity**
- features are highly correlated.
- common in macro data (many indicators move together).

**Permutation importance**
- feature importance measured by shuffling a feature and observing performance drop.

**Influence**
- whether a small number of points heavily affect the fitted coefficients.

### 9.1 Coefficients vs Feature Importance
Coefficients:
- tell you direction and magnitude in a linear model
- can be unstable under multicollinearity

Permutation importance:
- tells you whether the model uses a feature
- still can be misleading if two features are redundant (shuffling one might not hurt much)

### 9.2 Residual Diagnostics (Regression)
Questions residuals can answer:
- Are errors larger during crises? (heteroskedasticity)
- Are errors correlated month-to-month? (autocorrelation)
- Do we systematically underpredict at high inflation? (model misspecification)

Practical checks:
- histogram of residuals
- residuals vs time
- residuals vs predictions

### 9.3 Error Analysis by Date
Macro series have historical episodes:
- recessions
- policy regime changes
- supply shocks

A good habit:
- list the worst errors
- map them to dates
- ask what changed economically

### 9.4 Interpreting Signs in an Economic Context
If you see a sign that conflicts with intuition:
- check scaling
- check multicollinearity
- check whether features are levels vs diffs
- check whether the target definition matches your intuition

---

<a id="toc-macro"></a>
## 10) Macro-Specific Pitfalls (Revisions, Structural Breaks, Nonstationarity)

### Key Terms (with detail)

**Data revision**
- historical data can be updated after initial publication.

**Vintage data**
- the value as it was known at that time.

**Structural break**
- relationships change.

**Regime**
- a period with different macro dynamics (e.g., low inflation vs high inflation).

### 10.1 Revisions and Real-Time Forecasting
Many indicators are revised. If you train on revised data, you may overestimate real-time performance.

Advanced extension:
- use vintage data sources and evaluate using only information available at the time.

### 10.2 Structural Breaks
If relationships change, a single global model can struggle.

Symptoms:
- performance is good in some eras, terrible in others
- residuals spike around major events

Potential responses:
- shorter training window
- regime features
- walk-forward retraining

### 10.3 Nonstationarity
Trending series can cause spurious relationships.

Tools:
- differences
- percent changes
- including levels + regularization

---

<a id="toc-extensions"></a>
## 11) Practical Use Cases and Extensions

Ideas that stay within the tutorial's spirit (interpretable, hands-on):
- Add a recession indicator and predict recession probability.
- Change the horizon from 1 month to 3 months ahead.
- Implement walk-forward validation and plot performance over time.
- Add more indicators and compare ridge vs lasso stability.

Ideas that go beyond (still useful):
- Gradient boosting models (less interpretable, often stronger).
- SHAP explanations for complex models.
- Real-time/vintage forecasting.

---

<a id="toc-glossary"></a>
## 12) Quick Reference Glossary (Short)

This is intentionally short. Full explanations are in the sections.

- API: interface for requesting data.
- Cache: stored copy to avoid refetching.
- Feature: model input.
- Target: model output to predict.
- Lag: past value used as a feature.
- Regularization: penalty that shrinks coefficients.
- Residual: y_true - y_pred.
- Leakage: using future info accidentally.
- Walk-forward: repeated time-ordered validation.
