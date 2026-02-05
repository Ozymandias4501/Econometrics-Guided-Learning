### Deep Dive: Multicollinearity and VIF — why coefficients become unstable

Multicollinearity is common in economic data and is one of the main reasons coefficient interpretation becomes fragile.

#### 1) Intuition (plain English)

If two predictors contain almost the same information, the regression struggles to decide “which variable deserves the credit.”

**Story example (macro):**
- many indicators co-move with the business cycle,
- a multifactor regression may assign unstable signs to “similar” indicators depending on sample period.

Prediction may still be fine, but coefficient stories become unreliable.

#### 2) Notation + setup (define symbols)

Regression in matrix form:

$$
\\mathbf{y} = \\mathbf{X}\\beta + \\varepsilon.
$$

OLS estimator:
$$
\\hat\\beta = (X'X)^{-1}X'y.
$$

Under classical assumptions:
$$
\\mathrm{Var}(\\hat\\beta \\mid X) = \\sigma^2 (X'X)^{-1}.
$$

**What each term means**
- When columns of $X$ are highly correlated, $X'X$ is close to singular.
- Then $(X'X)^{-1}$ has large entries → coefficient variance inflates.

#### 3) What multicollinearity does (and does not do)

- Often does **not** hurt prediction much.
- **Does** inflate standard errors (coefficients become noisy).
- **Does** make coefficients sensitive to small data changes (unstable signs/magnitudes).
- **Does not** automatically bias coefficients if exogeneity holds; it mostly increases variance.

#### 4) VIF (Variance Inflation Factor): what it measures

To compute VIF for feature $j$:
- regress $x_j$ on all other predictors,
- record the $R_j^2$ from that auxiliary regression.

Then:

$$
\\mathrm{VIF}_j = \\frac{1}{1 - R_j^2}.
$$

**Interpretation**
- If $R_j^2$ is near 1, $x_j$ is almost perfectly explained by other predictors.
- Then $\\mathrm{VIF}_j$ is large → coefficient uncertainty for $\\beta_j$ is inflated.

Rules of thumb (not laws):
- VIF > 5 suggests notable collinearity.
- VIF > 10 suggests serious collinearity.

#### 5) Estimation mechanics: “holding others fixed” becomes unrealistic

Coefficient interpretation relies on the counterfactual:
- “Increase $x_j$ by 1 while holding other predictors fixed.”

If predictors are tightly linked economically, that counterfactual can be meaningless (you cannot change one indicator while freezing another).

So multicollinearity is both:
- a statistical issue (variance inflation),
- and an economic interpretation issue (counterfactuals).

#### 6) Diagnostics + robustness (minimum set)

1) **Correlation matrix**
- identify groups of highly correlated features.

2) **VIF table**
- quantify redundancy; large VIF → unstable coefficient.

3) **Coefficient stability**
- fit the regression on different subperiods or bootstrap samples; do signs flip?

4) **Condition number**
- large condition number of $X$ suggests numerical instability.

#### 7) What to do about multicollinearity

Options (choose based on goals):

- **If you care about interpretation**
  - drop redundant variables,
  - combine variables into an index,
  - use domain-driven composites.

- **If you care about prediction**
  - use regularization (ridge/lasso),
  - use dimension reduction (PCA/factors),
  - use nonlinear models (trees) with care about leakage and evaluation.

#### 8) Interpretation + reporting

When multicollinearity is present, report:
- VIF or correlation evidence,
- coefficient instability (if observed),
- and avoid strong stories about individual coefficients.

#### Exercises

- [ ] Construct two highly correlated predictors and show VIF > 10.
- [ ] Fit OLS with both predictors; observe coefficient instability vs the true DGP.
- [ ] Drop one predictor and compare interpretability and fit.
- [ ] Fit ridge regression and compare coefficient stability to OLS.
