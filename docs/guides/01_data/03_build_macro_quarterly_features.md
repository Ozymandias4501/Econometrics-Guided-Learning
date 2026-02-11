# Guide: 03_build_macro_quarterly_features

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01_data/03_build_macro_quarterly_features.ipynb`.

**Prerequisites:** [Data engineering foundations](00_fred_api_and_caching.md) — core data principles (tidy format, caching, validation, reproducibility).

This guide focuses on **feature engineering for time-series forecasting**: turning
raw economic indicators into lag, rolling-window, and percent-change features that
respect the information set available at prediction time. Building features correctly
is the bridge between a clean dataset and a trustworthy model.

### Key Terms (defined)
- **Lag feature**: a past value of a variable used as a predictor (e.g., unemployment at $t-1$).
- **Rolling statistic**: a summary (mean, std, min, max) computed over a trailing window of fixed length.
- **Percent change**: the relative change in a variable over a fixed horizon (month-over-month, year-over-year).
- **Lookahead bias (leakage)**: using information that would not have been available at prediction time.
- **Feature naming convention**: a systematic column-naming scheme that encodes the variable, transform, and horizon.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Aggregate monthly indicators to quarterly frequency (`.resample('QE').last()` or `.mean()`).
- Create lag features (1 through 4 quarters) for each macro indicator.
- Create rolling statistics (mean, std) over 3- and 4-quarter trailing windows.
- Create percent-change features (quarter-over-quarter and year-over-year).
- Apply a consistent naming convention: `{var}_lag{n}`, `{var}_pct{n}q`, `{var}_roll{k}q_mean`, etc.
- Merge features with GDP growth / recession label; confirm target is shifted forward.
- Drop rows with NaN introduced by lags/rolls and save the final modeling table.
- Spot-check at least 3 rows to verify no lookahead: every feature at row $t$ uses data from $\le t$.

### Alternative Example (Not the Notebook Solution)
```python
# Toy lag + rolling feature construction (not the real notebook data):
import pandas as pd, numpy as np

idx = pd.date_range('2015-03-31', periods=20, freq='QE')
np.random.seed(42)
df = pd.DataFrame({
    'unrate': np.random.normal(5.0, 0.5, 20),
    'cpi':    np.random.normal(250, 5, 20),
}, index=idx)

# --- Lag features ---
for col in ['unrate', 'cpi']:
    for lag in [1, 2, 3, 4]:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# --- Percent-change features ---
df['unrate_pct1q'] = df['unrate'].pct_change(1)       # quarter-over-quarter
df['cpi_pct4q']    = df['cpi'].pct_change(4)           # year-over-year

# --- Rolling features (trailing window, no lookahead) ---
df['unrate_roll4q_mean'] = df['unrate'].rolling(4, min_periods=4).mean()
df['cpi_roll4q_std']     = df['cpi'].rolling(4, min_periods=4).std()

# Drop NaN rows created by lags/rolls, then inspect
df_model = df.dropna()
print(df_model.head())
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Why lags matter for time-series prediction

In cross-sectional regression, features and the outcome are measured at the same
moment. In time-series forecasting the question is different: *given what we know
today, what will happen next quarter?* This means every feature must come from the
**information set available at prediction time** — the set of values that have
already been observed when the forecast is made.

Lag features are the simplest way to operationalize this idea. If you want to
predict $y_{t+1}$ (e.g., whether the next quarter is a recession), you can use
$x_t, x_{t-1}, x_{t-2}, \ldots$ but never $x_{t+1}$.

Lags also capture **persistence**: many macro series are autocorrelated, so
knowing the recent trajectory provides genuine predictive signal.

**Practical starting point.** For quarterly macro data, lags 1 through 4 (one full
year of history) are a sensible default for each indicator. You can prune later
with feature-selection methods.

### Rolling statistics and their smoothing properties

A rolling statistic summarizes a trailing window of $k$ observations:

$$
\overline{x}_{t,k} = \frac{1}{k}\sum_{j=0}^{k-1} x_{t-j}
\quad\text{(rolling mean)}
$$

$$
s_{t,k} = \sqrt{\frac{1}{k-1}\sum_{j=0}^{k-1}(x_{t-j} - \overline{x}_{t,k})^2}
\quad\text{(rolling std)}
$$

Rolling means smooth out short-run noise — useful when an indicator is volatile
quarter to quarter but the underlying trend matters. Rolling standard deviations
capture recent **volatility**, which itself can predict regime changes (calm vs
turbulent periods).

You can also compute rolling min and rolling max to capture the range of recent
outcomes.

**Window-length trade-off.** Shorter windows (2–3 quarters) track turning points
quickly but are noisy. Longer windows (8–12 quarters) are smoother but respond
slowly. A practical recipe: compute both a short and a long window, then let the
model select which is more useful.

### Percent-change features

Percent changes convert levels into growth rates, which are closer to stationary
and more directly interpretable:

$$
\%\Delta_h x_t = \frac{x_t - x_{t-h}}{x_{t-h}}
$$

Common horizons:

| Horizon ($h$) | Name | Column suffix |
|---|---|---|
| 1 quarter | Quarter-over-quarter | `_pct1q` |
| 4 quarters | Year-over-year | `_pct4q` |
| 12 months (monthly data) | Year-over-year | `_pct12m` |

Year-over-year changes are especially popular because they remove seasonal patterns
without explicit seasonal adjustment.

**When levels are near zero** (e.g., interest rates at 0.1%), percent changes
blow up and become misleading. In such cases prefer simple differences
($\Delta x_t = x_t - x_{t-h}$) instead.

### Feature naming conventions

Consistent naming prevents confusion when you have dozens of derived columns.
The convention used in this project:

| Transform | Pattern | Example |
|---|---|---|
| Lag | `{var}_lag{n}` | `unrate_lag1` |
| Percent change | `{var}_pct{h}q` or `{var}_pct{h}m` | `cpi_pct4q` |
| Rolling mean | `{var}_roll{k}q_mean` | `unrate_roll4q_mean` |
| Rolling std | `{var}_roll{k}q_std` | `spread_roll8q_std` |
| Difference | `{var}_diff{h}` | `fedfunds_diff1` |

Reading any column name should immediately tell you: *which raw variable, which
transform, and which horizon*.

### The lookahead trap

Lookahead bias is the single most common data-engineering mistake in time-series
work. It means a feature at time $t$ contains information from $t+1$ or later.

**How it happens in practice:**

1. **Wrong shift direction.** `df['x'].shift(-1)` pulls future values into the
   current row.
2. **Centered rolling windows.** `df['x'].rolling(5, center=True)` uses two
   future observations.
3. **Fitting a scaler on the full dataset** and then splitting by time — the
   scaler's mean/std incorporate future data.
4. **Revised data.** Using the final-vintage GDP number as a feature when, in real
   time, only a preliminary estimate was available.

**Prevention checklist:**

- Always use `.shift(+n)` with positive $n$ for lag features.
- Always use `center=False` (the default) for rolling windows.
- Fit any scaler/encoder on the training set only.
- After building the modeling table, pick 3–5 random rows and manually verify that
  every feature value could have been observed before the target date.

### Practical recipe: building a feature matrix

For each raw indicator $x$ in your dataset:

1. **Lags 1–4:** `x_lag1`, `x_lag2`, `x_lag3`, `x_lag4`.
2. **Rolling windows (mean and std):** 3-quarter and 4-quarter trailing windows.
   Optionally add 8-quarter windows for slow-moving indicators.
3. **Percent changes:** quarter-over-quarter (`pct1q`) and year-over-year (`pct4q`).

This yields roughly 10–12 features per raw indicator. With 5 indicators, you start
with ~50–60 features — enough to let a model learn patterns, but small enough
that regularization (Ridge/Lasso) can handle it without exotic dimensionality
reduction.

After construction:

- Drop the first $k_{\max}$ rows (where $k_{\max}$ is the longest window) to
  remove NaN-filled observations.
- Merge the recession label shifted one quarter forward (`target_recession_next_q`).
- Confirm no NaN remains.
- Save to `data/processed/macro_quarterly.csv`.

### Health economics connection: lag features in clinical prediction

The logic of lag features extends beyond macro forecasting. In health economics,
**hospital readmission models** use the same principle: prior utilization patterns
(number of ER visits in the past 6 months, medication fill history, prior
inpatient days) are lag features that capture a patient's recent health trajectory.
The discipline is identical — every feature must precede the prediction date, and
rolling summaries smooth out episode-level noise to reveal underlying acuity.

### Notation summary

| Symbol | Meaning |
|---|---|
| $x_t$ | Value of indicator $x$ at time $t$ |
| $x_{t-j}$ | Lag $j$ of indicator $x$ |
| $\overline{x}_{t,k}$ | Trailing $k$-period rolling mean of $x$ |
| $s_{t,k}$ | Trailing $k$-period rolling standard deviation of $x$ |
| $\%\Delta_h x_t$ | Percent change of $x$ over $h$ periods |
| $y_{t+1}$ | Target label one period ahead |

### Diagnostics and robustness checks

1. **Spot-check rows.** Print a few timestamps and verify that `var_lag1` at row
   $t$ equals `var` at row $t-1$.
2. **Compare transforms.** Fit a simple model with levels only vs. differences
   only vs. full feature set. If the full feature set massively outperforms levels
   in-sample but not out-of-sample, suspect overfitting.
3. **Overfitting guard.** More features raise variance; always evaluate with
   time-aware splits (walk-forward CV, expanding-window).
4. **Leakage audit.** Intentionally shift one feature the wrong way (`.shift(-1)`)
   and re-train. If accuracy jumps, your evaluation caught the leak. If it does
   not, your evaluation itself may be leaking.

### Exercises

- [ ] Create lag and rolling features for two macro indicators and verify alignment by printing rows.
- [ ] Compare a model using levels vs differences; interpret the change in accuracy.
- [ ] Create a leakage feature intentionally (shift the wrong way) and show how it inflates performance.
- [ ] Write a 5-line data dictionary for the final feature matrix (column, meaning, units, frequency, source).

### Project Code Map
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/macro.py`: GDP + labels (`gdp_growth_qoq`, `gdp_growth_yoy`, `technical_recession_label`, `monthly_to_quarterly`)
- `src/fred_api.py`: FRED client (`fetch_series_meta`, `fetch_series_observations`, `observations_to_frame`)
- `scripts/build_datasets.py`: end-to-end dataset builder
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Using `.shift(-1)` when you mean `.shift(1)` — this pulls future values into the current row.
- Using `center=True` in a rolling window, which peeks into the future.
- Forgetting to `dropna()` after creating lags/rolls, leaving NaN rows that crash or bias the model.
- Merging quarterly GDP with monthly predictors without explicit aggregation (silent misalignment).
- Creating dozens of features without checking whether the model generalizes (overfitting to noise).

<a id="summary"></a>
## Summary + Suggested Readings

You now have a feature matrix built from raw macro indicators using lags, rolling
statistics, and percent changes — all constructed with strict lookahead discipline.
This matrix feeds directly into the classification and regression guides that follow.

**Key takeaways:**
- Lags encode "what was known when" and capture persistence.
- Rolling statistics smooth noise (mean) or capture volatility (std).
- Percent changes approximate stationarity and remove seasonal patterns.
- Every feature must pass the lookahead test: is this value observable before the target date?
- Consistent naming (`var_lag1`, `var_pct4q`, `var_roll4q_mean`) prevents confusion downstream.

Suggested readings:
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (Ch. 7–8, feature-based forecasting)
- pandas documentation: `.shift()`, `.rolling()`, `.pct_change()`
- Christoffersen & Diebold, "Financial Asset Returns, Direction-of-Change Forecasting, and Volatility Dynamics" (on rolling-window features)
- Hamilton, *Time Series Analysis*, Ch. 1–3 (lag operators and autoregressive structure)
