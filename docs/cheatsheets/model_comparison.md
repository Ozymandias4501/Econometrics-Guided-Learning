# Cheatsheet: Model Comparison

## Regression Models

| Model | When to use | Pros | Cons | Health econ example |
|---|---|---|---|---|
| **OLS** | Linear relationship, few predictors, need interpretable coefficients and hypothesis tests | Unbiased estimates (if assumptions hold), closed-form solution, full inference toolkit (t-tests, F-tests, CIs) | Sensitive to outliers, no built-in variable selection, overfits with many predictors relative to observations | Estimating the effect of insurance coverage rates on county-level ER visits |
| **Ridge ($L_2$)** | Many correlated predictors, want to keep all variables in the model | Stabilizes estimates when predictors are correlated (shrinks but doesn't zero out), closed-form solution, always has a unique answer | No variable selection — all predictors stay in, coefficients are biased (intentionally, toward zero), harder to interpret magnitudes | Predicting hospital readmissions from 50+ correlated patient characteristics |
| **Lasso ($L_1$)** | Many predictors, want the model to automatically select a sparse subset | Sets some coefficients exactly to zero (built-in variable selection), interpretable sparse model | Unstable when predictors are highly correlated (may arbitrarily pick one of two similar variables), can select at most $n$ variables | Identifying which of 100 community health indicators best predict mortality |
| **Elastic Net** | Many correlated predictors + want some sparsity | Combines Ridge's stability with Lasso's sparsity — handles groups of correlated variables better than Lasso alone | Two tuning parameters ($\alpha$ for L1/L2 mix, $\lambda$ for penalty strength), more complex to tune | Predicting healthcare costs from a mix of demographic, clinical, and geographic features |
| **GLS / WLS** | Known structure in the errors (heteroskedasticity or autocorrelation) | More efficient (tighter CIs) than OLS when the error structure is correctly specified | Must correctly specify the error covariance — misspecification can make things worse than OLS | Modeling health expenditure where variance increases with income |

### Why regularization helps: geometric intuition

**Ridge** constrains coefficients to lie within a sphere (circle in 2D). The OLS solution is an ellipse of equal-cost contours. The sphere has no corners, so the constrained solution typically has all coefficients nonzero but shrunk toward zero.

**Lasso** constrains coefficients to lie within a diamond. The diamond has corners on the axes, and the OLS ellipse is most likely to first touch the diamond at a corner — meaning one or more coefficients are exactly zero. This is why Lasso produces sparse solutions.

## Classification Models

| Model | When to use | Pros | Cons | Health econ example |
|---|---|---|---|---|
| **Logistic Regression** | Binary outcome, interpretable model needed, want predicted probabilities | Coefficients are log-odds ratios (directly interpretable), outputs calibrated probabilities, works well with moderate number of predictors | Assumes a linear decision boundary in log-odds space, limited flexibility for complex nonlinear patterns | Predicting 30-day hospital readmission from patient demographics and prior utilization |
| **Decision Tree** | Exploratory analysis, want to visualize decision rules, interactions between variables | Highly interpretable (can draw the tree), naturally captures interactions and nonlinearities, no scaling needed | Overfits aggressively without pruning, unstable (small data changes can produce a completely different tree), high variance | Triage rules: which combination of symptoms and vitals predicts ICU admission? |
| **Random Forest** | Want strong predictive accuracy with moderate interpretability | Averages many trees to reduce variance, handles interactions and nonlinearities, robust to outliers and missing data | Less interpretable than a single tree (no single set of rules), slower to train, many hyperparameters to tune | Predicting which Medicare beneficiaries will have high annual spending |
| **Gradient Boosting (XGBoost, LightGBM)** | Maximum predictive accuracy is the priority | Often the best-performing model on tabular data, handles missing values natively, provides feature importance scores | Least interpretable of the tree methods, prone to overfitting without careful regularization (learning rate, tree depth, early stopping), expensive to tune | Kaggle-style prediction of hospital quality scores from facility characteristics |
| **Naive Bayes** | Text classification, very fast baseline, extremely small datasets | Very fast training and prediction, works well even with small samples, simple to implement | Assumes features are conditionally independent given the class (almost never true in practice), poor probability calibration | Quick baseline for classifying clinical notes into diagnostic categories |

## Model Selection Criteria

| Criterion | Formula / Idea | Strengths | Limitations |
|---|---|---|---|
| **Adjusted $R^2$** | $1 - \frac{SSR/(n-k)}{SST/(n-1)}$ — penalizes each additional predictor | Simple, built into every regression output | Only for linear regression; penalizes weakly, can still overfit |
| **AIC** | $-2\ln(L) + 2k$ — balances log-likelihood against number of parameters | Estimates out-of-sample prediction error; applicable to any likelihood-based model | Favors more complex models than BIC; not valid for comparing models on different datasets |
| **BIC** | $-2\ln(L) + k\ln(n)$ — like AIC but with a stronger penalty that grows with sample size | Consistent model selection (picks the true model as $n \to \infty$); penalizes complexity more heavily | Can underfit in small samples by penalizing too harshly |
| **Cross-validation** | Directly estimate out-of-sample error by repeatedly holding out portions of the data | Model-agnostic, directly measures what you care about (prediction accuracy), works for any model | Computationally expensive; random splits are invalid for time series |
| **Walk-forward CV** | Time-series version: always train on the past, test on the future, expand the training window | Respects temporal ordering, no look-ahead bias, realistic simulation of live forecasting | Requires enough data for multiple train/test splits; early splits have small training sets |

## Bias-Variance Trade-off

```
High bias (underfitting)          ←  Model complexity  →          High variance (overfitting)

Simple models                                                    Complex models
(OLS, logistic, shallow tree)                                    (deep trees, boosting, many features)
Training error ≈ test error                                      Training error << test error
(both high — model too rigid)                                    (model memorizes noise)
```

**Regularization** (Ridge, Lasso, tree pruning, early stopping) reduces variance at the cost of a small increase in bias. The goal is to find the complexity level that minimizes *total* error (bias$^2$ + variance).

**Practical sign**: If your model performs much better on training data than on test/validation data, you're overfitting (high variance). If it performs poorly on both, you're underfitting (high bias).

## Decision Flowchart

```
What is your primary goal?
│
├── Inference (understand and quantify relationships)
│   ├── Cross-section data → OLS + HC3 robust SE
│   ├── Time-series data → OLS + HAC/Newey-West SE
│   ├── Panel data (entity × time) → Fixed Effects + clustered SE
│   └── Causal effect → IV/2SLS, DiD, or RDD (see causal_inference cheatsheet)
│
├── Prediction (forecast an outcome as accurately as possible)
│   ├── Linear relationship, few predictors → OLS or Logistic Regression
│   ├── Many predictors, want sparsity → Lasso or Elastic Net
│   ├── Nonlinear patterns, interactions → Random Forest or Gradient Boosting
│   ├── Time-series target → Walk-forward validation required for any model
│   └── Need calibrated probabilities → Logistic Regression or any model + Platt scaling
│
└── Variable selection (identify which features matter)
    ├── Linear setting → Lasso (coefficients shrunk to zero = excluded)
    ├── Nonlinear setting → Tree-based feature importance (Gini, gain, or SHAP)
    └── Model-agnostic → Permutation importance (works with any fitted model)
```

## Key Trade-offs

| Dimension | One end | Other end | How to navigate |
|---|---|---|---|
| **Interpretability vs. accuracy** | OLS, logistic, single decision tree | Gradient boosting, random forest, ensembles | Start simple; only add complexity if prediction gains justify the interpretability loss |
| **Bias vs. variance** | Simple, high-bias models (underfit) | Flexible, high-variance models (overfit) | Use cross-validation to find the sweet spot; regularization helps |
| **In-sample fit vs. generalization** | Unregularized model with many variables (great training fit) | Regularized or simple model (worse training fit, better test fit) | Always evaluate on held-out data; never trust training-set metrics alone |
| **Speed vs. performance** | Logistic regression, Naive Bayes (fast) | XGBoost, random forest (slower) | For real-time scoring, speed matters; for offline analysis, prefer accuracy |
| **Inference vs. prediction** | OLS with robust SE (valid coefficient interpretation) | Black-box models (best predictions, no causal interpretation) | Define your goal before choosing a model — the best predictor is often useless for causal questions |
