### Deep Dive: Ridge vs lasso — stabilizing models when features are many/correlated

Regularization is a core tool when you have many predictors, multicollinearity, or limited sample sizes.

#### 1) Intuition (plain English)

OLS chooses coefficients to fit the training data as closely as possible.
With many correlated features, OLS can produce:
- large, unstable coefficients,
- high variance (overfitting),
- fragile inference.

Regularization trades a bit of bias for much lower variance.

#### 2) Notation + setup (define symbols)

OLS objective:

$$
\\min_{\\beta} \\; \\|y - X\\beta\\|_2^2.
$$

Ridge (L2) objective:

$$
\\min_{\\beta} \\; \\|y - X\\beta\\|_2^2 + \\lambda \\|\\beta\\|_2^2.
$$

Lasso (L1) objective:

$$
\\min_{\\beta} \\; \\|y - X\\beta\\|_2^2 + \\lambda \\|\\beta\\|_1.
$$

**What each term means**
- $\\lambda \\ge 0$ controls penalty strength.
- L2 shrinks coefficients smoothly; L1 can set some coefficients exactly to 0 (feature selection).

#### 3) Assumptions and practical requirements

Regularization is sensitive to feature scale:
- you should standardize features before ridge/lasso so the penalty is comparable across variables.

Regularization changes the estimand:
- coefficients are no longer the OLS estimand,
- classical p-values/CI are not straightforward after model selection.

#### 4) Estimation mechanics

Ridge has a closed-form solution:

$$
\\hat\\beta_{ridge} = (X'X + \\lambda I)^{-1} X'y.
$$

Lasso does not have a simple closed form; software uses optimization (coordinate descent).

#### 5) Inference: focus on prediction + stability

In this project, regularization is used primarily for:
- improving out-of-sample prediction,
- stabilizing coefficients,
- reducing variance in macro settings.

Treat inference (p-values) after lasso with caution; selection changes distribution.

#### 6) Diagnostics + robustness (minimum set)

1) **Cross-validation for $\\lambda$**
- use time-aware CV for forecasting tasks.

2) **Coefficient paths**
- inspect how coefficients shrink as $\\lambda$ increases.

3) **Stability across folds**
- do selected features change dramatically across time folds? That suggests instability.

#### 7) Interpretation + reporting

Report:
- how $\\lambda$ was chosen,
- whether features were standardized,
- out-of-sample metrics and stability.

#### Exercises

- [ ] Fit ridge and lasso with standardized features; compare out-of-sample performance.
- [ ] Plot coefficient paths vs $\\lambda$ and interpret shrinkage.
- [ ] Compare OLS vs ridge coefficients when features are collinear; explain why ridge is more stable.
