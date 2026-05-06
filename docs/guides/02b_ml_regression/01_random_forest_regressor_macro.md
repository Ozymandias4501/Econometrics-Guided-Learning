# Guide: 01_random_forest_regressor_macro

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies `notebooks/02b_ml_regression/01_random_forest_regressor_macro.ipynb`.

`RandomForestRegressor` is the workhorse ML regressor for tabular data: it is fast to train, has very few critical hyperparameters, and tends to give a strong baseline with almost no tuning. It works by averaging many decision trees, each grown on a bootstrap sample with a random feature subset at each split. The averaging is what reduces variance.

### Key Terms (defined)
- **Bagging (Bootstrap AGGregating)**: train each tree on a bootstrap sample of rows.
- **Feature subsampling**: at every split, only consider $\sqrt{p}$ features for regression. Decorrelates the trees.
- **`n_estimators`**: number of trees. More is rarely harmful (just slower).
- **`max_depth`**: maximum tree depth. `None` = grow until pure. Controls bias.
- **`min_samples_leaf`**: minimum observations in a leaf. Strong regularization knob.
- **Permutation importance**: shuffle a feature on the *test* set and measure the RMSE drop.
- **Built-in (impurity) importance**: total impurity reduction across all splits using the feature, summed across trees. Biased toward high-cardinality features.

### How To Read This Guide
- Use **Step-by-Step** as a checklist while running the notebook.
- Use **Technical Explanations** for the math and the most common tuning mistakes.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Load the same macro panel and split as notebook `00`.
- Fit a default `RandomForestRegressor(n_estimators=500, min_samples_leaf=3)` and record test RMSE / MAE / R².
- Run a small grid over `max_depth ∈ {3, 6, None}` × `min_samples_leaf ∈ {1, 3, 5}` using `walk_forward_splits` to score each combo.
- Refit the best combo on the full training set; score on the test set.
- Plot predicted vs actual and residuals over time.
- Compute permutation importance with `sklearn.inspection.permutation_importance(n_repeats=30)`.
- Compare RF to OLS on a single summary table.

### Alternative Example (Not the Notebook Solution)
```python
# Use OOB (out-of-bag) error as a free generalization estimate
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=500,
    oob_score=True,            # request OOB R² automatically
    min_samples_leaf=3,
    random_state=0,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
print('OOB R²:', rf.oob_score_)  # ~unbiased generalization estimate, no holdout needed
```

OOB is convenient for quick diagnostics on small datasets, but does not replace a proper time-aware test set when the data is a time series.

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Why averaging reduces variance

Suppose each individual tree has variance $\sigma^2$ on a single prediction. If the trees were perfectly independent, averaging $B$ of them would give variance $\sigma^2 / B$ — i.e. variance shrinks linearly with the number of trees. Trees on bootstrap samples are not perfectly independent (they share data), so the realized variance is

$$
\rho \sigma^2 + \frac{(1 - \rho) \sigma^2}{B}
$$

where $\rho$ is the average pairwise correlation between trees. The random feature subset at each split **decorrelates** the trees, lowering $\rho$ — which is the whole game. With $\rho = 1$ (identical trees) averaging buys you nothing.

### Hyperparameters that matter (in order)

1. **`min_samples_leaf`** — the regularization knob for small datasets. Default is 1; raising it to 3–10 prevents leaves with one observation, which is necessary on ~100-quarter macro panels.
2. **`max_depth`** — caps individual tree complexity. `None` (default) grows until pure; for small datasets, capping at 6 or 10 is often better.
3. **`n_estimators`** — bigger is rarely harmful. 300–500 is the sweet spot. 100 is too few; 5000 is wasted compute.
4. **`max_features`** — controls feature subsampling at each split. The default `1.0` (all features) is slightly worse than `'sqrt'` for regression in practice; try both.

### Permutation importance, again

The notebook reports both built-in and permutation importance. They will often disagree on macro features because:

- The 10y–2y spread (`T10Y2Y_lag1`) is highly informative around recessions but flat in expansions. Built-in importance counts every split it participates in; permutation importance only credits it when removing it actually hurts test performance — and on a non-recession-heavy test slice, permutation importance can be much smaller.
- Lag features for the same series are correlated, so trees pick one at random and split credit between them. Permutation importance for any one lag can look small even when the *group* is critical.

When permutation importance for an individual feature looks suspiciously low, group correlated features and permute them together (`permutation_importance(...)` with a custom function).

### Common Mistakes
- Forgetting `random_state` and getting different results every time you re-run.
- Using random K-fold for time series — see `00`.
- Tuning on the test set. Use walk-forward CV on the training portion only.
- Reading too much into permutation importance on a small test set. Confidence intervals are wide; report `importances_std` alongside `importances_mean`.

<a id="summary"></a>
## Summary + Suggested Readings

You should now be able to fit, tune, and read the output of `RandomForestRegressor` on a small macro panel, and tell whether it beats OLS on RMSE. The next notebook does the same exercise with XGBoost and adds a final three-way comparison table.

### Suggested Readings
- Breiman (2001), "Random Forests" — the foundational paper.
- Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*, ch. 15.
- scikit-learn user guide: ["Forests of randomized trees"](https://scikit-learn.org/stable/modules/ensemble.html#forest).
- Christodoulou et al. (2019), "A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models" — a sobering counterpoint that applies just as well to macro forecasting.
