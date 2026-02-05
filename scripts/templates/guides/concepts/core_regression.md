### Core Regression: mechanics, assumptions, and interpretation (OLS as the baseline)

Linear regression is the baseline model for both econometrics and ML. Even when you use nonlinear models, the regression mindset (assumptions → estimation → inference → diagnostics) remains essential.

#### 1) Intuition (plain English)

Regression answers questions like:
- “How does $Y$ vary with $X$ on average?”
- “Holding other observed controls fixed, what is the association between one feature and the outcome?”

In economics we care about two different uses:
- **prediction:** does a model forecast well out-of-sample?
- **inference:** what is the estimated relationship and its uncertainty?

#### 2) Notation + setup (define symbols)

Scalar form (observation $i=1,\\dots,n$):

$$
y_i = \\beta_0 + \\beta_1 x_{i1} + \\cdots + \\beta_K x_{iK} + \\varepsilon_i.
$$

Matrix form:

$$
\\mathbf{y} = \\mathbf{X}\\beta + \\varepsilon.
$$

**What each term means**
- $\\mathbf{y}$: $n\\times 1$ vector of outcomes.
- $\\mathbf{X}$: $n\\times (K+1)$ design matrix (includes an intercept column).
- $\\beta$: $(K+1)\\times 1$ vector of coefficients.
- $\\varepsilon$: $n\\times 1$ vector of errors (unobserved determinants).

#### 3) Assumptions (what you need for unbiasedness and for inference)

For interpretation and inference, it helps to separate:

**(A) Assumptions for unbiased coefficients**

1) **Linearity in parameters**
- $y$ is linear in $\\beta$ (you can still include nonlinear transformations of $x$).

2) **No perfect multicollinearity**
- columns of $X$ are not perfectly linearly dependent.

3) **Exogeneity (key!)**
$$
\\mathbb{E}[\\varepsilon \\mid X] = 0.
$$

This rules out:
- omitted variable bias,
- reverse causality,
- many forms of measurement error problems.

**(B) Assumptions for classical standard errors**

4) **Homoskedasticity**
$$
\\mathrm{Var}(\\varepsilon \\mid X) = \\sigma^2 I.
$$

5) **No autocorrelation (time series)**
$$
\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_{t-k}) = 0 \\text{ for } k \\neq 0.
$$

When (4)–(5) fail, OLS coefficients can remain valid under (A), but naive SE are wrong → robust/HAC/clustered SE.

#### 4) Estimation mechanics: deriving OLS

OLS chooses coefficients to minimize the sum of squared residuals:

$$
\\hat\\beta = \\arg\\min_{\\beta} \\sum_{i=1}^{n} (y_i - x_i'\\beta)^2
= \\arg\\min_{\\beta} (\\mathbf{y} - \\mathbf{X}\\beta)'(\\mathbf{y} - \\mathbf{X}\\beta).
$$

Take derivatives (the “normal equations”):

$$
\\frac{\\partial}{\\partial \\beta} (\\mathbf{y}-\\mathbf{X}\\beta)'(\\mathbf{y}-\\mathbf{X}\\beta)
= -2\\mathbf{X}'(\\mathbf{y}-\\mathbf{X}\\beta) = 0.
$$

Solve:
$$
\\mathbf{X}'\\mathbf{X}\\hat\\beta = \\mathbf{X}'\\mathbf{y}
\\quad \\Rightarrow \\quad
\\hat\\beta = (\\mathbf{X}'\\mathbf{X})^{-1}\\mathbf{X}'\\mathbf{y}.
$$

**What each term means**
- $(X'X)^{-1}$ exists only if there is no perfect multicollinearity.
- OLS is a projection of $y$ onto the column space of $X$.

#### 5) Coefficient interpretation (and why “holding fixed” is tricky)

In the model, $\\beta_j$ means:

> the expected change in $y$ when $x_j$ increases by one unit, holding other regressors fixed (within the model).

In economics, “holding fixed” can be unrealistic if regressors move together (multicollinearity).
That is why:
- coefficient signs can flip,
- SE can inflate,
- interpretation must be cautious.

#### 6) Inference: standard errors, t-stats, confidence intervals

Under classical assumptions:

$$
\\mathrm{Var}(\\hat\\beta \\mid X) = \\sigma^2 (X'X)^{-1}.
$$

In practice we estimate $\\sigma^2$ and compute standard errors:
- $\\widehat{SE}(\\hat\\beta_j)$
- t-stat: $t_j = \\hat\\beta_j / \\widehat{SE}(\\hat\\beta_j)$
- 95% CI: $\\hat\\beta_j \\pm 1.96\\,\\widehat{SE}(\\hat\\beta_j)$ (approx.)

When assumptions fail, use robust SE:
- **HC3** for cross-section heteroskedasticity,
- **HAC/Newey–West** for time-series autocorrelation + heteroskedasticity,
- **clustered SE** for grouped dependence (panels/DiD).

#### 7) Diagnostics + robustness (minimum set)

1) **Residual checks**
- plot residuals vs fitted values; look for heteroskedasticity/nonlinearity.

2) **Multicollinearity**
- compute VIF; large VIF → unstable coefficients.

3) **Time-series dependence**
- check residual autocorrelation; use HAC when needed.

4) **Stability**
- rolling regressions or sub-sample splits; do coefficients drift?

#### 8) Interpretation + reporting

Always report:
- coefficient in units (or standardized units),
- robust SE appropriate to data structure,
- a short causal warning unless you have a causal design.

**What this does NOT mean**
- Regression does not “control away” all confounding automatically.
- A small p-value does not imply economic importance.
- A high $R^2$ does not imply good forecasting out-of-sample.

#### Exercises

- [ ] Derive the normal equations and explain each step in words.
- [ ] Fit OLS and HC3 (or HAC) and compare SE; explain why they differ.
- [ ] Create two correlated regressors and show how multicollinearity affects coefficient stability.
- [ ] Write a 6-sentence interpretation of one regression output, including what you can and cannot claim.
