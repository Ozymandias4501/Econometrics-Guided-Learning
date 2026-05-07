# Guide: 00_walk_forward_backtest

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/04_honest_forecasting/00_walk_forward_backtest.ipynb`.

This is the **definitive reference** for time-series backtesting in this project. The classification version (§03 walk-forward validation) cross-references this one for the underlying mechanics.

### Key Terms (defined)
- **Backtest**: simulating how a model would have performed if you had been running it in real time on historical data.
- **Walk-forward**: a backtesting protocol where you re-fit the model at multiple points in time, each time using only data available up to that point.
- **Expanding window**: the training window starts at observation 0 and grows over time. Mimics 'I have all my data so far.'
- **Rolling window**: the training window has fixed length and slides forward over time. Mimics 'recent data is more relevant.'
- **Pseudo-out-of-sample**: predictions made on data the model never saw during fitting, but where the data is still historical (you know the answer).
- **Refit cadence**: how often the model is retrained inside the walk-forward loop. Smaller cadence = more refits = more realistic + more compute.
- **Look-ahead bias**: any way the model can see information that wouldn't have been available at the prediction time. Lethal in backtests; almost always silent.
- **Ensemble**: a forecast combining predictions from multiple models. The simplest version is the mean.

### How To Read This Guide
- Use **Step-by-Step** for the notebook checklist and the rolling-vs-expanding decision.
- Use **Technical Explanations** for the mechanics, look-ahead bias, and the rationale for ensemble combining.
- Use the [structural-breaks guide](01_structural_breaks_and_covid.md) for what to do when the walk-forward error plot reveals a break.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- [ ] Pick the target carefully: `df['gdp_growth_qoq'].shift(-1)` to forecast next-quarter growth.
- [ ] Pick predictors that are *known at time t* (lagged features, never contemporaneous).
- [ ] Implement two single train/test splits with different cutoffs to confirm the result is sensitive to the cutoff.
- [ ] Use `walk_forward_splits(initial_train_size, test_size)` from `src.evaluation`.
- [ ] Wrap your model in a `model_fn(X_train, y_train, X_test) -> y_pred` so the same backtest loop drives OLS, RF, and XGBoost.
- [ ] Collect predictions and realizations into a single DataFrame indexed by date.
- [ ] Report MAE, RMSE, R² aggregated over the whole walk-forward window.
- [ ] Plot rolling absolute error over time so you see *when* each model fails, not just the average.
- [ ] Build a mean ensemble and check whether it beats the best individual model.

### Alternative Example: Rolling Window Backtest

If you suspect old data is misleading (regime change), use a rolling window instead of expanding.

```python
from collections import deque

def rolling_walk_forward(panel, x_cols, y_col, model_fn, window=60, test_size=4):
    out = []
    for start in range(0, len(panel) - window - test_size + 1, test_size):
        train = panel.iloc[start:start + window]
        test = panel.iloc[start + window:start + window + test_size]
        yhat = model_fn(train[x_cols], train[y_col], test[x_cols])
        out.append(pd.DataFrame({'y_true': test[y_col].to_numpy(), 'y_pred': yhat}, index=test.index))
    return pd.concat(out)
```

The expanding-window version uses *all* prior data; the rolling version uses only the most recent `window` observations. On macro data, expanding usually wins on average RMSE because data is precious; rolling can win during regime changes.

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Why a single train/test split flatters time-series models

Suppose you do an 80/20 split: train on the first 80% of observations, test on the last 20%. Two problems:

1. The test window is one specific historical regime. If it happens to be a calm period, the model looks better than it is. If it is across a crisis, it looks worse.
2. The model is not actually 're-fit at every point in time' — it has the entire pre-2017 dataset in its training window when predicting 2024. That is more information than a real deployed model would have.

Walk-forward fixes both. By averaging over many test windows, you get a regime-distributed estimate of error. By re-fitting at each step, you accurately reflect that early predictions had less data to work with.

### Mechanics of the expanding-window walk-forward

For a panel of length $n$ with `initial_train_size = T_0` and `test_size = h`, the protocol generates splits:

| Iteration | Train | Test |
|---|---|---|
| 1 | $[0, T_0)$ | $[T_0, T_0 + h)$ |
| 2 | $[0, T_0 + h)$ | $[T_0 + h, T_0 + 2h)$ |
| ... | ... | ... |
| $K$ | $[0, T_0 + (K-1)h)$ | $[T_0 + (K-1)h, n)$ |

The number of refits is $K = \lfloor (n - T_0) / h \rfloor$. With $n = 156$ quarters, $T_0 = 40$, $h = 4$: $K = 29$ refits.

Refit cadence ($h$) is a tradeoff. Setting $h = 1$ refits at every quarter (most realistic, but the model only gets one new data point between refits). Setting $h = 4$ matches an analyst who refits annually. Setting $h$ to the test_size you care about (one year, four quarters) keeps the test windows non-overlapping and easy to interpret.

### Look-ahead bias: the failure mode of every backtest

A backtest is only as honest as its feature engineering. Three classic ways to leak the future:

1. **Using contemporaneous features.** If your target is GDP growth in quarter $t+1$, the predictors should be known at time $t$ — meaning lagged macro indicators (`UNRATE_lag1`, `T10Y2Y_lag1`, etc.). Using `UNRATE` itself in a regression that targets next-quarter growth is using information that wouldn't be available at the prediction point.
2. **Standardizing on the full sample.** If you compute a column's mean and std on the entire dataset and use it to standardize, the test rows have been touched by training-set statistics. The fix: standardize *inside* each train fold.
3. **Hyperparameter tuning on the test data.** Common in ML pipelines. If you grid-search the regularization parameter on the same window you report OOS RMSE on, the OOS RMSE is contaminated. Use a nested split.

The walk-forward loop in this notebook avoids #1 by lagging features. It does not solve #2 or #3 — those are the user's responsibility.

### Why an unweighted ensemble often wins

If three models make errors that are partly uncorrelated, averaging cancels noise. Mathematically, with model errors $e_i$ each having variance $\sigma^2$ and pairwise correlation $\rho$:

$$
\mathrm{Var}\left(\frac{1}{m}\sum_{i=1}^m e_i\right) = \frac{\sigma^2}{m}\bigl(1 + (m-1)\rho\bigr).
$$

For $m = 3$ and $\rho = 0.7$ (similar models on the same data, plausible value), the ensemble error variance is $\sigma^2 \cdot 0.8$ — a 10% reduction in standard error for free. The lower the correlation between model errors, the more the ensemble helps. Where it stops working: when all the models share the same structural blind spot (e.g., none of them have ever seen COVID).

### Reading the rolling-error plot

The aggregate RMSE compresses the most useful information. Plot rolling absolute errors over time and look for:

- **Spikes during crises**: usually expected. A spike above the median across multiple models = a regime the cohort cannot handle.
- **Persistent shifts in the level of error**: a model that used to be good and is now worse than its peers. Often a sign of structural change in the relationships.
- **One model going wild while others are calm**: an idiosyncratic failure mode of that model class. RF can spike on out-of-distribution feature values; OLS can spike when residuals stop being mean-zero.

### Project Code Map
- `src/evaluation.py`: `walk_forward_splits`, `time_train_test_split_index`, `regression_metrics`.
- `src/econometrics.py`: `fit_ols`, `fit_ols_hac` for the OLS arm.
- The notebook builds its own `backtest()` helper rather than using one from `src/`. The helper is small and explicit; users should be able to read every line.

### Common Mistakes
- Re-tuning hyperparameters on the test window. Defeats the purpose.
- Using `cross_val_score` from sklearn out of the box on time-series data. Default CV shuffles, which is catastrophic. Use `TimeSeriesSplit` or this notebook's explicit walk-forward.
- Reporting only the aggregate RMSE. The time series of errors is much more informative.
- Reporting walk-forward RMSE that is *lower* than the single-split RMSE you started with. That is suspicious — usually a sign of look-ahead bias somewhere.

<a id="summary"></a>
## Summary + Suggested Readings

This guide is the playbook for honest time-series evaluation in this project. After working through the notebook you should be able to:

- explain why a single train/test split is misleading on time-series data,
- implement an expanding-window walk-forward from scratch in 15 lines,
- decide between expanding and rolling based on the data,
- spot the three canonical look-ahead biases in your own pipeline,
- build a mean ensemble and explain mathematically why it helps,
- read the rolling-error plot for evidence of a structural break.

**Companion guides:**
- [Guide 01 — Structural Breaks](01_structural_breaks_and_covid.md): what to do once the walk-forward plot reveals a regime change.
- [Guide 03/04 — walk-forward classification](../03_classification/04_walk_forward_validation.md): same protocol applied to a probabilistic target.

**Suggested readings:**
- Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.), Ch. 5: time-series cross-validation.
- Diebold, *Forecasting in Economics, Business, Finance and Beyond*, Ch. 9–11: realistic forecast evaluation.
- López de Prado, *Advances in Financial Machine Learning*, Ch. 7: combinatorial walk-forward and the dangers of look-ahead bias (financial bias, but the lessons transfer).
- scikit-learn user guide: `TimeSeriesSplit` for an off-the-shelf walk-forward iterator.
