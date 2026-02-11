# Guide: 05_regularization_ridge_lasso

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/05_regularization_ridge_lasso.ipynb`.

When you have many predictors -- especially correlated ones -- OLS coefficients become unstable and out-of-sample performance degrades. Regularization adds a penalty to the objective function that shrinks coefficients toward zero, trading a small amount of bias for a large reduction in variance. This guide covers the two most common penalized regression methods (ridge and lasso), how to choose the penalty strength, and when regularization is the right tool.

### Key Terms (defined)
- **Ridge regression (L2)**: penalizes the sum of squared coefficients; shrinks all coefficients but never sets them exactly to zero.
- **Lasso regression (L1)**: penalizes the sum of absolute coefficients; can set some coefficients exactly to zero (feature selection).
- **Elastic net**: combines L1 and L2 penalties; a compromise between ridge and lasso.
- **Lambda ($\lambda$)**: the penalty strength parameter; larger $\lambda$ means more shrinkage.
- **Shrinkage**: the process of pulling coefficients toward zero to reduce variance.
- **Feature selection**: identifying which predictors matter by setting irrelevant coefficients to zero.
- **Cross-validation (CV)**: a resampling method used to choose $\lambda$ by estimating out-of-sample performance.
- **Coefficient paths**: plots showing how each coefficient changes as $\lambda$ increases from 0 to large values.
- **Standardization**: rescaling features to have mean 0 and variance 1, required before regularization so the penalty treats all features equally.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math behind ridge and lasso penalties.
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Standardize all features before fitting regularized models (do NOT standardize the target).
- Fit both ridge and lasso regression across a grid of $\lambda$ values.
- Plot coefficient paths (coefficients vs $\log(\lambda)$) for both ridge and lasso; note which lasso coefficients hit zero first.
- Use cross-validation (e.g., `RidgeCV`, `LassoCV`, or manual time-aware CV) to select the optimal $\lambda$.
- Compare out-of-sample performance: plain OLS vs ridge vs lasso at the CV-optimal $\lambda$.
- Report which features lasso selects (nonzero coefficients) and check whether the selection is stable across CV folds.

### Alternative Example (Not the Notebook Solution)

This example fits ridge and lasso on a dataset with many correlated features, showing how coefficient paths evolve as $\lambda$ increases.

```python
# Ridge and lasso on correlated features with coefficient path plots
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n, p = 200, 15

# Correlated features: 3 groups of 5 correlated variables
Z = rng.normal(size=(n, 3))
X = np.hstack([
    Z[:, [0]] + rng.normal(0, 0.3, (n, 5)),   # group 1: correlated
    Z[:, [1]] + rng.normal(0, 0.3, (n, 5)),   # group 2: correlated
    Z[:, [2]] + rng.normal(0, 0.3, (n, 5)),   # group 3: correlated
])
# True model: only first variable in each group matters
beta_true = np.array([1, 0, 0, 0, 0,  -0.5, 0, 0, 0, 0,  0.8, 0, 0, 0, 0])
y = X @ beta_true + rng.normal(0, 1, n)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Coefficient paths for lasso
alphas_lasso = np.logspace(-3, 1, 100)
coefs_lasso = []
for a in alphas_lasso:
    model = Lasso(alpha=a, max_iter=10000)
    model.fit(X_std, y)
    coefs_lasso.append(model.coef_.copy())
coefs_lasso = np.array(coefs_lasso)

plt.figure(figsize=(8, 5))
for j in range(p):
    plt.plot(np.log10(alphas_lasso), coefs_lasso[:, j], label=f'x{j}')
plt.xlabel('log10(lambda)')
plt.ylabel('Coefficient')
plt.title('Lasso coefficient paths')
plt.axhline(0, color='grey', linewidth=0.5)
plt.show()

# Cross-validate lambda
lasso_cv = LassoCV(alphas=alphas_lasso, cv=5).fit(X_std, y)
print(f"Best lambda: {lasso_cv.alpha_:.4f}")
print(f"Nonzero coefficients: {np.sum(lasso_cv.coef_ != 0)} / {p}")
```

**What to notice:** The correlated features within each group get similar coefficients under ridge (all shrunk but nonzero), while lasso tends to pick one representative from each group and zero out the rest. The coefficient paths show this divergence clearly.

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Prerequisites: OLS Foundations (Guide 00)

This guide builds on the OLS foundations covered in [Guide 00](00_single_factor_regression_micro.md). That guide covers the core regression framework: the OLS objective and normal equations, assumptions for unbiasedness and inference, coefficient interpretation, robust standard errors, and multicollinearity diagnostics (VIF). Read it first if you have not already. Regularization modifies the OLS objective by adding a penalty term -- everything else (matrix notation, interpretation caveats, diagnostics mindset) carries over.

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
\min_{\beta} \; \|y - X\beta\|_2^2.
$$

Ridge (L2) objective:

$$
\min_{\beta} \; \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2.
$$

Lasso (L1) objective:

$$
\min_{\beta} \; \|y - X\beta\|_2^2 + \lambda \|\beta\|_1.
$$

**What each term means**
- $\lambda \ge 0$ controls penalty strength.
- L2 shrinks coefficients smoothly toward zero but never reaches it; L1 can set some coefficients exactly to 0 (feature selection).

**Why L1 gives exact zeros (geometric intuition):** Picture the constraint region in coefficient space. For L2 (ridge), it is a circle (or hypersphere); for L1 (lasso), it is a diamond with corners on the axes. The OLS solution's contours are ellipses. The regularized solution is where the ellipses first touch the constraint region. Because the L1 diamond has sharp corners sitting on the axes, the first contact point is likely at a corner — which means one or more coefficients are exactly zero. The L2 circle has no corners, so contact typically occurs at a point where all coefficients are nonzero but shrunken. This is why lasso performs feature selection and ridge does not.

#### 3) Assumptions and practical requirements

Regularization is sensitive to feature scale:
- you should standardize features before ridge/lasso so the penalty is comparable across variables.

Regularization changes the estimand:
- coefficients are no longer the OLS estimand,
- classical p-values/CI are not straightforward after model selection.

#### 4) Estimation mechanics

Ridge has a closed-form solution:

$$
\hat\beta_{ridge} = (X'X + \lambda I)^{-1} X'y.
$$

Lasso does not have a simple closed form; software uses optimization (coordinate descent).

#### 5) Inference: focus on prediction + stability

In this project, regularization is used primarily for:
- improving out-of-sample prediction,
- stabilizing coefficients,
- reducing variance in macro settings.

Treat inference (p-values) after lasso with caution; selection changes distribution.

#### 6) Diagnostics + robustness (minimum set)

1) **Cross-validation for $\lambda$**
- use time-aware CV for forecasting tasks.

2) **Coefficient paths**
- inspect how coefficients shrink as $\lambda$ increases.

3) **Stability across folds**
- do selected features change dramatically across time folds? That suggests instability.

#### 7) Interpretation + reporting

Report:
- how $\lambda$ was chosen,
- whether features were standardized,
- out-of-sample metrics and stability.

#### Exercises

- [ ] Fit ridge and lasso with standardized features; compare out-of-sample performance.
- [ ] Plot coefficient paths vs $\lambda$ and interpret shrinkage.
- [ ] Compare OLS vs ridge coefficients when features are collinear; explain why ridge is more stable.

### When to Use Regularization in Practice

Regularization is not always the right tool. This section clarifies when it helps, when it does not, and what tradeoffs you accept.

#### Many predictors relative to observations ($p$ approaching $n$)

When the number of features $p$ is large relative to the sample size $n$, OLS becomes unreliable:
- If $p > n$, OLS cannot be computed at all (the system is underdetermined).
- If $p$ is close to $n$, OLS overfits badly -- it can fit the training data perfectly while performing terribly out of sample.

Regularization constrains the solution, making estimation feasible even when $p > n$ (lasso and ridge both work in this regime). In health economics, this arises frequently: predicting patient outcomes from hundreds of diagnosis codes, lab values, and demographic variables when the study sample is only a few thousand patients.

#### High multicollinearity

When predictors are highly correlated, OLS coefficients are individually unstable (high VIF). You have two options:
1. **Drop variables** -- simple, but you lose information and must choose which to drop.
2. **Regularize** -- ridge automatically downweights redundant predictors without discarding any, preserving all available signal for prediction.

Ridge is especially useful here because it handles groups of correlated features gracefully by distributing weight among them. Lasso tends to pick one representative and zero out the rest, which can be unstable (which variable it picks may change with small data perturbations).

#### Prediction vs inference tradeoff

This is the most important conceptual point:

- **Regularized coefficients are biased by construction.** The penalty deliberately pushes coefficients away from the OLS estimate toward zero. This bias improves prediction (lower variance more than compensates), but it means the coefficients no longer have the "unbiased estimate of the true effect" property that OLS has under exogeneity.

- **Do not use regularized coefficients for causal inference or treatment effect estimation.** If your goal is "what is the effect of $x$ on $y$?", use OLS (with appropriate SE) or a causal identification strategy. If your goal is "predict $y$ as accurately as possible," regularization is often the right choice.

- In practice, you might use lasso for variable *screening* (which variables seem predictive?) and then refit OLS on the selected variables for inference. But be aware that post-selection inference requires specialized methods (e.g., the "post-lasso OLS" approach of Belloni and Chernozhukov, or debiased lasso).

#### Elastic net as a compromise

Elastic net combines L1 and L2 penalties:

$$
\min_{\beta} \; \|y - X\beta\|_2^2 + \lambda \left[\alpha \|\beta\|_1 + (1-\alpha) \|\beta\|_2^2\right],
$$

where $\alpha \in [0,1]$ controls the mix ($\alpha=1$ is lasso, $\alpha=0$ is ridge). Elastic net is useful when:
- you want feature selection (like lasso) but have groups of correlated features (where lasso is unstable),
- you want the "grouping effect" of ridge (correlated features get similar coefficients) combined with sparsity.

In `scikit-learn`, use `ElasticNetCV` with a grid over both $\lambda$ and $\alpha$.

#### Health economics example: predicting hospital readmissions

Suppose you want to predict 30-day hospital readmission from patient and hospital characteristics. You have:
- 50+ patient features: age, sex, BMI, 20+ comorbidity indicators, lab values, medication counts, prior utilization.
- 10+ hospital features: bed count, teaching status, nurse-to-patient ratio, region dummies.
- $n = 3{,}000$ patients.

With 60+ features and only 3,000 observations, OLS is likely to overfit. Many comorbidity indicators are correlated (e.g., diabetes and hypertension frequently co-occur). The right approach:

1. Standardize all features.
2. Fit lasso with CV to identify which features are most predictive (expect many zeros).
3. Fit ridge with CV as a benchmark (often similar or better prediction, but no feature selection).
4. Compare AUC or Brier score across OLS, ridge, and lasso on a held-out test set.
5. If interpretability matters (e.g., explaining to clinicians which risk factors to target), report the lasso-selected features, but caveat that the specific selection may be unstable.

**What you should NOT do:** Use the regularized coefficients to claim "diabetes increases readmission risk by X%." The coefficients are shrunken and biased. For causal claims about individual risk factors, you need a different study design.

#### Exercises

- [ ] Fit elastic net with a grid over $\alpha$ and $\lambda$; compare to pure ridge and lasso.
- [ ] Take a dataset with $p > n/2$ and show that OLS fails while ridge/lasso produce reasonable predictions.
- [ ] Discuss: why would a clinician prefer lasso over ridge for a readmission risk score? What would they lose?

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`) + multicollinearity (`vif_table`)
- `src/evaluation.py`: regression metrics helpers (`regression_metrics`, `classification_metrics`)
- `src/evaluation.py`: splits (`time_train_test_split_index`, `walk_forward_splits`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)

### Common Mistakes
- Fitting ridge or lasso without standardizing features first; the penalty unfairly penalizes large-scale features.
- Interpreting regularized coefficients as unbiased estimates of causal effects.
- Using lasso for feature selection and then reporting the lasso coefficients (instead of refitting OLS on selected features for inference).
- Choosing $\lambda$ by training-set performance instead of cross-validation (guarantees overfitting for small $\lambda$).
- Forgetting to use time-aware CV splits for time-series or panel data (standard k-fold leaks future information).
- Ignoring coefficient path stability: if lasso selects different features in different CV folds, the selection is fragile.

<a id="summary"></a>
## Summary + Suggested Readings

After working through this guide you should be able to:
- explain the bias-variance tradeoff that motivates regularization,
- fit ridge and lasso, plot coefficient paths, and cross-validate $\lambda$,
- use the geometric intuition to explain why lasso gives exact zeros and ridge does not,
- compare OLS vs regularized predictions on held-out data, and
- recognize when regularization helps (prediction with many features) vs when it hurts (causal inference).

Suggested readings:
- Hastie, Tibshirani & Friedman: *Elements of Statistical Learning*, Ch. 3 (ridge, lasso, elastic net)
- James, Witten, Hastie & Tibshirani: *Introduction to Statistical Learning*, Ch. 6 (regularization with R/Python labs)
- Belloni & Chernozhukov: "High-Dimensional Methods and Inference on Structural and Treatment Effects" (2014) -- post-lasso inference
- scikit-learn docs: `Ridge`, `Lasso`, `ElasticNet`, `RidgeCV`, `LassoCV`
