# Guide: 02_xgboost_regressor_macro

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies `notebooks/02b_ml_regression/02_xgboost_regressor_macro.ipynb`.

`XGBoost` is the canonical gradient-boosted decision tree library. Where Random Forest fits independent trees and averages them, XGBoost fits trees **sequentially** — each new tree is trained to fix the residual errors of the ensemble built so far. In practice it almost always beats RF on tabular data when tuned, and is a Kaggle / industrial workhorse.

### Key Terms (defined)
- **Boosting**: sequential ensemble where each model corrects the running ensemble's residuals.
- **`learning_rate` ($\eta$)**: how aggressively each new tree is added. Small $\eta$ + many trees generalizes better than large $\eta$ + few trees.
- **`n_estimators`**: number of boosting rounds (= number of trees).
- **Early stopping**: stop adding trees when validation RMSE / log-loss has not improved for *N* rounds.
- **`tree_method='hist'`**: histogram-based splitter that bins continuous features. Much faster than the exact splitter, and the right default.
- **`subsample` / `colsample_bytree`**: row / column subsampling fractions per tree. Stochasticity that reduces overfitting.
- **`gain` importance**: XGBoost's built-in importance metric (average loss reduction per split using the feature).

### How To Read This Guide
- Use **Step-by-Step** as a checklist while running the notebook.
- Use **Technical Explanations** for the boosting math and tuning strategy.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Load the same panel / split as notebooks `00` and `01`.
- Carve a validation slice out of the training set (last ~15–20%) for early stopping.
- Fit a default `XGBRegressor(n_estimators=2000, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, tree_method='hist', early_stopping_rounds=50)`.
- Walk-forward search over `max_depth × learning_rate`. Each fold uses early stopping internally.
- Refit the best combo on the full training set with early stopping on the inner validation slice; record test RMSE / MAE / R² and the best iteration count.
- Plot predicted vs actual and residuals over time.
- Compute gain importance and permutation importance side by side.
- Build the final OLS vs RF vs XGBoost comparison table including training time.

### Alternative Example (Not the Notebook Solution)
```python
# scale_pos_weight matters even for regression with extreme target imbalance
# (e.g., predicting recessionary vs non-recessionary GDP growth)

from xgboost import XGBRegressor

# Up-weight observations with negative growth (recession-like)
sample_weight = (y_train < 0).astype(float) * 4 + 1  # 5x weight on negative growth

m = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                 tree_method='hist', random_state=0)
m.fit(X_train, y_train, sample_weight=sample_weight)
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### How gradient boosting works

Initialize the prediction with a constant $F_0(x) = \bar y$. At each round $m = 1, 2, \dots, M$:

1. Compute the negative gradient of the loss with respect to the current predictions: $r_i = -\partial L(y_i, F_{m-1}(x_i)) / \partial F$.
2. Fit a small (shallow) tree $h_m(x)$ to the **gradients** $r_i$.
3. Update the ensemble: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$.

For squared error, the negative gradient is just the residual $y_i - F_{m-1}(x_i)$, so each new tree is fit to the current residuals. This is "gradient" boosting in the same sense that gradient descent steps in the direction that reduces the loss most.

### `learning_rate` × `n_estimators`: the central tradeoff

Halving $\eta$ roughly doubles the number of trees you need to reach a given training fit. Small $\eta$ + many trees is essentially always better for generalization — each step is cautious, and the model averages over many small adjustments. The cost is compute time. Typical defaults: $\eta = 0.03$–$0.1$ with `n_estimators = 500`–$2000$ and early stopping.

### Early stopping is the right way to pick `n_estimators`

Cross-validating `n_estimators` is wasteful — instead, hand XGBoost an `eval_set` and let it stop when validation RMSE has not improved for `early_stopping_rounds` rounds. The trained model exposes `best_iteration` so you know exactly how many trees were used. You then *do not* tune `n_estimators`; you tune everything else and let early stopping pick `n_estimators` for free.

### Tuning order

1. Set `learning_rate=0.05` and `n_estimators=2000` with `early_stopping_rounds=50`. Forget about `n_estimators` from now on.
2. Tune `max_depth ∈ {3, 4, 6, 8}`. Shallow (3–6) is the boosting sweet spot.
3. Tune `min_child_weight ∈ {1, 3, 5, 10}` (XGBoost's analog to `min_samples_leaf`).
4. Tune `subsample` and `colsample_bytree` together over `{0.6, 0.8, 1.0}`.
5. Optionally, drop `learning_rate` to 0.03 or 0.01 in the final retrain — buys a little extra performance for compute.

Avoid grid-searching all five at once. Sequential tuning gives 90% of the lift at 5% of the cost.

### Gain importance vs permutation importance

`xgb.feature_importances_` reports **gain**: the average improvement in loss when the feature is used in a split. It is the gold standard among XGBoost's three flavors (gain, weight, cover) but it is still computed on training data and still biased toward features with many split candidates. As with Random Forest, run `permutation_importance` on the test set and compare.

### Common Mistakes
- Tuning `n_estimators` directly instead of using early stopping — slow and brittle.
- Forgetting `tree_method='hist'` — the default exact splitter is slower for no benefit on tabular data.
- Not setting `eval_metric` explicitly — modern XGBoost warns; old code logs nothing.
- Comparing RMSE across models trained on different splits / preprocessing pipelines.
- Reading too much into a single test-set RMSE — always pair with walk-forward CV mean / std.

<a id="summary"></a>
## Summary + Suggested Readings

You should now be able to:
- Fit `XGBRegressor` with early stopping on a held-out validation slice.
- Tune the depth × learning-rate combination via walk-forward CV.
- Read gain importance and permutation importance side by side.
- Produce a final OLS vs Random Forest vs XGBoost RMSE / MAE / R² / training-time table.

If XGBoost won, document by how much. If OLS won, document why — small samples and approximately-linear macro relationships are the most common reason. Both outcomes are interesting.

### Suggested Readings
- Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System" — the foundational paper.
- Friedman (2001), "Greedy Function Approximation: A Gradient Boosting Machine."
- Official XGBoost docs: ["XGBoost Parameters"](https://xgboost.readthedocs.io/en/stable/parameter.html).
- scikit-learn user guide: ["Histogram-Based Gradient Boosting"](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) — sklearn's built-in alternative if you want to drop the `xgboost` dependency.
