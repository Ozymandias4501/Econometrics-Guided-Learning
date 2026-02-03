### Core Regression: Mechanics, Interpretation, and Uncertainty

Regression is used for both prediction and inference.

#### The model
We write a linear regression as:

$$
\mathbf{y} = \mathbf{X}\beta + \varepsilon
$$

- $\mathbf{y}$: outcomes
- $\mathbf{X}$: predictors (features)
- $\beta$: coefficients
- $\varepsilon$: error term (everything not modeled)

OLS estimates:

$$
\hat\beta = (X'X)^{-1}X'y
$$

#### Coefficient interpretation
> **Definition:** A **coefficient** $\beta_j$ is the expected change in $y$ for a one-unit change in $x_j$, holding other features fixed (within the model).

Key interpretation cautions:
- "Holding others fixed" can be unrealistic when predictors are correlated.
- A coefficient is not automatically causal.

#### Standard errors and confidence intervals
> **Definition:** A **standard error** measures uncertainty in an estimated coefficient.

A 95% confidence interval is roughly:

$$
\hat\beta_j \pm 1.96 \cdot \widehat{SE}(\hat\beta_j)
$$

(Exact multipliers depend on the t distribution and sample size.)

#### Robust standard errors
Robust SE do not change coefficients, but they change uncertainty estimates.
- HC3: common for cross-section (heteroskedasticity)
- HAC/Newey-West: common for time series (autocorrelation + heteroskedasticity)

#### Prediction vs inference
- For prediction, use time-aware evaluation and report out-of-sample metrics.
- For inference, report uncertainty and diagnose assumptions.
