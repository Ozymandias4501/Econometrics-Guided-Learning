### Deep Dive: Logistic Regression as a Probability Model (Odds and Log-Odds)

Logistic regression is a linear model for probabilities.

#### Key terms (defined)
> **Definition:** **Odds** are $p/(1-p)$.

> **Definition:** **Log-odds** are $\log\left(\frac{p}{1-p}\right)$.

> **Definition:** The **sigmoid** function maps real numbers to (0,1):

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

#### Model form
Logistic regression assumes log-odds are linear:

$$
\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k
$$

Equivalently:

$$
p = \sigma(\beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k)
$$

#### Coefficient interpretation
If you increase $x_j$ by 1 unit (holding other features fixed), log-odds change by $\beta_j$.
That means odds are multiplied by $e^{\beta_j}$.

Example:
- if $\beta_j = 0.7$, then odds multiply by $e^{0.7} \approx 2.0$.

#### Why scaling matters
If predictors are on different scales, coefficients are not directly comparable.
Standardizing features helps interpret relative influence.

#### Python demo: odds ratio from a fitted model (commented)
```python
import numpy as np

# Suppose coef is a learned coefficient (from sklearn or statsmodels)
coef = 0.7
odds_multiplier = np.exp(coef)
print('odds multiplier for +1 unit:', odds_multiplier)
```

#### Interpretation cautions in macro
- Coefficients are conditional associations, not causal effects.
- If features are collinear, coefficients can be unstable.
- Always evaluate out-of-sample with time-aware splits.
