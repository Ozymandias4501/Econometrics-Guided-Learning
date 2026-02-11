# Cheatsheet: Model Comparison

## Regression Models

| Model | When to use | Pros | Cons |
|---|---|---|---|
| **OLS** | Linear relationship, few predictors, inference needed | Unbiased (if assumptions hold), interpretable coefficients, hypothesis tests | Sensitive to outliers, no variable selection, can overfit with many predictors |
| **Ridge ($L_2$)** | Many correlated predictors, want to keep all variables | Handles multicollinearity, stable estimates, closed-form solution | Does not perform variable selection, less interpretable than OLS |
| **Lasso ($L_1$)** | Many predictors, want automatic variable selection | Sparse solutions (sets coefficients to zero), interpretable selected set | Can be unstable when predictors are correlated, selects at most $n$ variables |
| **Elastic Net** | Many correlated predictors + want sparsity | Combines Ridge stability with Lasso sparsity | Two tuning parameters ($\alpha$, $\lambda$) |
| **GLS / WLS** | Heteroskedastic or autocorrelated errors | More efficient estimates when error structure is known | Must correctly specify the error covariance |

## Classification Models

| Model | When to use | Pros | Cons |
|---|---|---|---|
| **Logistic Regression** | Binary outcome, interpretable model needed | Probabilistic output, coefficients are log-odds, works well with few predictors | Assumes linear decision boundary in log-odds, limited flexibility |
| **Decision Tree** | Exploring nonlinear splits, feature importance | Interpretable (visualize the tree), handles interactions naturally | Overfits easily, unstable (small data changes alter tree), high variance |
| **Random Forest** | Prediction accuracy, nonlinear relationships | Low variance (averages many trees), handles interactions, robust to outliers | Less interpretable than single tree, slower training, many hyperparameters |
| **Gradient Boosting (XGBoost, LightGBM)** | Maximum prediction accuracy | Often best predictive performance, handles missing data, feature importance | Least interpretable, risk of overfitting, requires careful tuning |
| **Naive Bayes** | Text classification, fast baseline | Very fast, works with small data, simple to implement | Strong independence assumption rarely holds |

## Model Selection Criteria

| Criterion | Formula / Idea | Use case |
|---|---|---|
| **Adjusted $R^2$** | Penalizes adding predictors that don't improve fit | Comparing nested regression models |
| **AIC** | $-2\ln(L) + 2k$ | Balances fit vs. complexity; favors prediction |
| **BIC** | $-2\ln(L) + k\ln(n)$ | Stronger penalty than AIC; favors parsimony |
| **Cross-validation** | Estimate out-of-sample error directly | Any model; gold standard for prediction |
| **Walk-forward CV** | Time-series version: train on past, test on future | Time-series and macro forecasting |

## Bias-Variance Trade-off

```
High bias (underfitting)          ←  Model complexity  →          High variance (overfitting)
Simple models (OLS, logistic)                                    Complex models (deep trees, boosting)
Training error ≈ test error (both high)                          Training error << test error
```

**Regularization** (Ridge, Lasso, tree pruning, early stopping) moves you left along this spectrum — trading a little bias for a large reduction in variance.

## Decision Flowchart

```
What is your goal?
├── Inference (understand relationships)
│   ├── Cross-section → OLS + HC3 SE
│   ├── Time series → OLS + HAC SE
│   ├── Panel data → Fixed Effects + Clustered SE
│   └── Causal effect → IV/2SLS or DiD
│
├── Prediction (forecast outcomes)
│   ├── Few predictors, linear → OLS or Logistic
│   ├── Many predictors → Ridge / Lasso / Elastic Net
│   ├── Nonlinear patterns → Random Forest or Gradient Boosting
│   └── Time series → Walk-forward validation required
│
└── Variable selection (which features matter?)
    ├── Linear setting → Lasso
    ├── Nonlinear setting → Tree-based feature importance
    └── Either → Permutation importance (model-agnostic)
```

## Key Trade-offs

| Trade-off | Left side | Right side |
|---|---|---|
| **Interpretability vs. accuracy** | OLS, logistic, single tree | Boosting, ensemble methods |
| **Bias vs. variance** | Simple, high-bias models | Flexible, high-variance models |
| **In-sample fit vs. out-of-sample** | Unregularized, many variables | Cross-validated, regularized |
| **Speed vs. performance** | Logistic, Naive Bayes | XGBoost, neural networks |
