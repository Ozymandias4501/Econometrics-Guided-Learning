# Guide: 00_ml_regression_setup_and_baselines

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02b_ml_regression/00_ml_regression_setup_and_baselines.ipynb`.

ML notebooks tend to gloss over the baseline. This notebook does the opposite: it locks in the dataset, the target, the train/test split, the OLS baseline, and the cross-validation strategy *before* a single tree is fit. Everything in notebooks 01 (Random Forest) and 02 (XGBoost) is compared against the numbers you record here.

### Key Terms (defined)
- **OLS baseline**: a plain ordinary least squares fit on the training set, scored on the held-out test set. The benchmark that the ML models must beat.
- **HAC standard errors (Newey–West)**: a covariance estimator robust to heteroskedasticity *and* autocorrelation. Required for time-series inference. Does not change point estimates.
- **Time-aware split**: a chronological train/test split — train on the earlier rows, test on the later rows — to avoid lookahead leakage.
- **Walk-forward CV**: rolling time-aware folds where the training window grows (or slides) and the next chunk is the test set. Implemented in `src/evaluation.py:walk_forward_splits`.

### How To Read This Guide
- Use **Step-by-Step** as a checklist while you work the notebook.
- Use **Technical Explanations** for the formal definitions and pitfalls.
- Bookmark the OLS / HAC numbers you record in the notebook — you will need them again in the next two notebooks.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Load the macro panel (`data/sample/macro_quarterly_sample.csv` if no API key) and select target `gdp_growth_qoq` plus five lagged predictors.
- Build a chronological 80/20 train/test split using `src.evaluation.time_train_test_split_index`.
- Fit OLS via `src.econometrics.fit_ols` and read off out-of-sample RMSE / MAE / R² with `src.evaluation.regression_metrics`.
- Fit OLS-HAC via `src.econometrics.fit_ols_hac(maxlags=4)`. Verify the **point estimates** are unchanged from plain OLS (only standard errors differ).
- Run walk-forward CV with `src.evaluation.walk_forward_splits(initial_train_size=0.5*n, test_size=8, step_size=8)`. Record the mean and std of fold RMSE.
- Write the OLS test RMSE / MAE / R² in a table you will reuse in notebooks 01 and 02.

### Alternative Example (Not the Notebook Solution)
```python
# Build a tiny baseline-comparison harness that any model can plug into.
import numpy as np
import pandas as pd
from src.evaluation import time_train_test_split_index, regression_metrics

def evaluate(model, X, y, test_size=0.2):
    sp = time_train_test_split_index(len(X), test_size=test_size)
    Xtr, ytr = X[sp.train_slice], y[sp.train_slice]
    Xte, yte = X[sp.test_slice], y[sp.test_slice]
    model.fit(Xtr, ytr)
    return regression_metrics(yte, model.predict(Xte))

# usage:
# evaluate(LinearRegression(), X, y)
# evaluate(RandomForestRegressor(...), X, y)
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### The OLS baseline as a benchmark

Out-of-sample, OLS is harder to beat than people think on macro data. Three reasons:

1. **Linearity is a strong prior.** Most macro relationships (Okun's law, Phillips curve in steady state) are approximately linear. Trees waste degrees of freedom learning that fact.
2. **Sample size.** ~150 quarters is small. Trees overfit quickly when the data-generating process is nearly linear.
3. **Bias-variance.** A linear model with five regressors has very low variance (see `00a/08`). Tree ensembles trade higher bias for lower variance, but if OLS is already low-bias, the trade is unfavorable.

This is why the OLS baseline is not a formality — it is often the answer.

### Why HAC standard errors

For time-series regressors, the OLS error term is rarely i.i.d. — residuals are typically autocorrelated, and variance can shift across regimes. The HAC (Newey–West) estimator handles both:

$$
\hat V_{\text{HAC}} = (X'X)^{-1} \hat \Omega (X'X)^{-1}
$$

where $\hat \Omega$ sums weighted lagged outer products of $X_i e_i$. The bandwidth `maxlags=4` is a reasonable default for quarterly data (roughly one year of autocorrelation). Point estimates are unchanged; t-stats are typically *deflated* (more honest).

### Walk-forward CV vs random K-fold

For time series, `KFold` from sklearn is wrong: it puts future quarters in the training set and past quarters in the test set, which destroys the test's meaning.

`walk_forward_splits(n, initial_train_size, test_size, step_size)` instead yields chronological folds:

```
fold 1: train [0:80]   test [80:88]
fold 2: train [0:88]   test [88:96]
fold 3: train [0:96]   test [96:104]
...
```

Each test window comes strictly after its training window. The output is a *distribution* of fold RMSE values — mean and std are both informative (a model with great mean but huge variance is fragile).

### Common Pitfalls
- Comparing OLS to RF on different splits — apples to oranges.
- Using R² alone to compare. Always report RMSE *and* R² (and MAE if outliers matter).
- Tuning hyperparameters on the test set. Use walk-forward CV on the training portion, then evaluate the chosen model exactly once on the test set.
- Forgetting that HAC fixes inference, not prediction. The point estimates are identical — RMSE will not change.

<a id="summary"></a>
## Summary + Suggested Readings

You now have a baseline (OLS test RMSE / MAE / R²) and a robust evaluator (walk-forward CV) that the next two notebooks will reuse. Recording these numbers carefully is the single most important thing in this section — without them, "ML beat OLS" is unfalsifiable.

### Suggested Readings
- Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*, ch. 7 (model assessment).
- Diebold, *Forecasting in Economics, Business, Finance and Beyond*, ch. on out-of-sample evaluation.
- scikit-learn user guide: ["Cross-validation: time series"](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split).
