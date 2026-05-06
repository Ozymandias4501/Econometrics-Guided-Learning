# Guide: 08_metrics_and_accuracy_vs_precision

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00a_statistics_primer/08_metrics_and_accuracy_vs_precision.ipynb`.

Every regression and ML output prints a battery of numbers — RMSE, R², F-statistic, accuracy, precision, recall, F1, AUC. The notebook teaches you to read each of them in plain English. This guide is the reference: definitions, the formulas behind them, and the traps that catch most students.

### Key Terms (defined)
- **MSE / RMSE**: average (root) squared residual. RMSE is in the same units as the target.
- **MAE**: average absolute residual. Outlier-robust alternative to RMSE.
- **R²**: fraction of variance in $y$ that the model explains, relative to predicting the mean.
- **Adjusted R²**: R² with a penalty for the number of regressors $k$.
- **Bias**: systematic error from a model that is too simple (underfitting).
- **Variance**: sensitivity to the specific training sample (overfitting).
- **Accuracy (statistics sense)**: closeness to the true value, on average.
- **Precision (statistics sense)**: closeness of repeated estimates to each other.
- **Confusion matrix**: 2×2 table of TP/FP/TN/FN counts.
- **Accuracy (classification metric)**: $(TP+TN)/(TP+FP+TN+FN)$.
- **Precision (classification metric)**: $TP/(TP+FP)$ — of predicted positives, fraction truly positive.
- **Recall / Sensitivity**: $TP/(TP+FN)$ — of true positives, fraction we caught.
- **F1**: harmonic mean of precision and recall.
- **ROC curve**: TPR vs FPR across all thresholds.
- **AUC**: area under the ROC curve. Threshold-free ranking quality.

### How To Read This Guide
- Use **Step-by-Step** for the implementation checklist when you forget what to compute.
- Use **Technical Explanations** for the formulas and the most common pitfalls.
- Bookmark the Rosetta Stone table in the notebook — it maps every metric to where it appears in `statsmodels`, `sklearn`, and `xgboost` output.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Compute MSE, RMSE, MAE manually from residuals; verify against `sklearn`.
- Inject one large outlier into your residuals and confirm RMSE jumps far more than MAE.
- Compute R² and adjusted R² manually; verify against `r2_score`.
- Run the bias–variance simulation: 30 fits of a degree-1 vs degree-12 polynomial on small samples drawn from $\sin(x) + \varepsilon$.
- Plot the four-panel dartboard for accuracy × precision combinations.
- Build a synthetic recession-prediction example with class imbalance; compute the confusion matrix, accuracy, precision, recall, F1.
- Plot a ROC curve and compute AUC; identify why AUC is the right metric when no threshold has been chosen yet.

### Alternative Example (Not the Notebook Solution)
```python
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

rng = np.random.default_rng(7)
y = rng.normal(0, 1, 500)
yhat = y + rng.normal(0, 0.4, 500)  # noisy predictions

print('MAE :', mean_absolute_error(y, yhat))
print('RMSE:', root_mean_squared_error(y, yhat))
print('R²  :', r2_score(y, yhat))

# Inject a single large mistake
y2 = y.copy(); yhat2 = yhat.copy()
yhat2[0] = y2[0] + 10
print('RMSE after outlier:', root_mean_squared_error(y2, yhat2))  # jumps a lot
print('MAE  after outlier:', mean_absolute_error(y2, yhat2))      # barely moves
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Regression metrics

$$\text{MSE} = \tfrac{1}{n}\sum_{i=1}^n (y_i - \hat y_i)^2 \qquad \text{RMSE} = \sqrt{\text{MSE}} \qquad \text{MAE} = \tfrac{1}{n}\sum_{i=1}^n |y_i - \hat y_i|$$

The choice between RMSE and MAE is about **how you want errors weighted**. Squaring penalizes large errors disproportionately, which matters when a single bad miss is much worse than several small ones (forecasting GDP turning points, predicting tail events). MAE is the median's friend: robust, easier to communicate.

$$R^2 = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2} \qquad \bar R^2 = 1 - (1 - R^2)\frac{n-1}{n-k-1}$$

R² in time series is notoriously misleading: trending variables tend to have R² near 1 even when the model is doing nothing economically meaningful. Always plot residuals (section 02 covers this) and compare against simple naive baselines (last observation, seasonal lag).

### Classification metrics

From the confusion matrix:

|              | Predicted Pos | Predicted Neg |
|--------------|---------------|---------------|
| Actual Pos   | TP            | FN            |
| Actual Neg   | FP            | TN            |

$$\text{Accuracy} = \frac{TP+TN}{TP+FP+TN+FN}\quad\text{Precision} = \frac{TP}{TP+FP}\quad\text{Recall} = \frac{TP}{TP+FN}\quad F_1 = \frac{2\,P\,R}{P+R}$$

The harmonic mean for F1 is not arbitrary: it equals zero whenever precision or recall is zero, which is the right behavior for a single-number summary on imbalanced data.

### Bias–variance decomposition (sketch)

For a fixed $x$, the expected squared prediction error decomposes as

$$\mathbb{E}[(y - \hat f(x))^2] = (\text{Bias}[\hat f(x)])^2 + \text{Var}[\hat f(x)] + \sigma^2_{\text{noise}}$$

This is the formal version of the dartboard analogy: total error has a systematic component (bias), a sample-to-sample component (variance), and an irreducible noise floor.

### Where each metric lies

| Metric | `statsmodels.OLS.fit()` | sklearn / xgboost |
|---|---|---|
| t-stat / p-value | `results.tvalues`, `results.pvalues` | n/a in core API |
| F-statistic | `results.fvalue` | n/a |
| R² / Adj R² | `results.rsquared`, `results.rsquared_adj` | `r2_score` (R² only) |
| Residual MSE | `results.mse_resid` | `mean_squared_error` |
| RMSE | $\sqrt{\text{mse\_resid}}$ | `root_mean_squared_error` |
| MAE | n/a in summary | `mean_absolute_error` |
| Confusion matrix | n/a | `confusion_matrix` |
| Accuracy / Precision / Recall / F1 | n/a | `*_score` family |
| ROC-AUC | n/a | `roc_auc_score` |

### Common Pitfalls
- Reporting only accuracy on imbalanced data — a model that always says "no" can score 95%.
- Comparing R² across models that use different transformations of $y$ (level vs log vs differenced). The denominators are different; the comparison is meaningless.
- Comparing R² across nested models where the larger has more regressors — use adjusted R² or cross-validated MSE.
- Confusing the *statistics* sense of "precision" (= low variance / repeatability) with the *classification* sense (= TP / (TP+FP)). They sound the same and mean opposite things.
- Picking a threshold by eyeballing accuracy on the training set. Use ROC analysis or an explicit cost function.

<a id="summary"></a>
## Summary + Suggested Readings

You now have a one-page mental model for every metric the rest of the curriculum will throw at you. Whenever a notebook prints `R-squared:`, `F-statistic:`, `precision`, or `AUC`, return to the Rosetta Stone table at the bottom of the notebook for the one-line definition.

### Suggested Readings
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, ch. 7 (model assessment, bias–variance).
- Wooldridge, *Introductory Econometrics*, ch. 6 (R² and goodness of fit).
- scikit-learn user guide: ["Metrics and scoring"](https://scikit-learn.org/stable/modules/model_evaluation.html).

### Coming Up Next
The regression section (`02_regression/`) will print every one of these regression metrics in real OLS output on FRED macro data. The classification section (`03_classification/`) will print the classification metrics. The new ML-for-regression section (`02b_ml_regression/`) will compare RMSE / MAE / R² across OLS, Random Forest, and XGBoost on the same FRED dataset.
