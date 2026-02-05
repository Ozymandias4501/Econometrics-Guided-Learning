### Deep Dive: Log–log regression (elasticity-style interpretation)

Log transforms are common in micro data because many variables (income, rent, population) are heavy-tailed and relationships are often multiplicative.

#### 1) Intuition (plain English)

In a log–log model, coefficients are approximately **elasticities**:
- “A 1% increase in $x$ is associated with a $\\beta$% change in $y$.”

This is often a more interpretable economic statement than “one dollar changes rent by …”

#### 2) Notation + setup (define symbols)

Log–log regression:

$$
\\log(y_i) = \\alpha + \\beta \\log(x_i) + \\varepsilon_i.
$$

**What each term means**
- $\\log(\\cdot)$ compresses scale and turns multiplicative relationships into additive ones.
- $\\beta$ is the elasticity-like coefficient.

Why elasticity? For small changes:

$$
\\Delta \\log(x) \\approx \\frac{\\Delta x}{x}
\\quad \\Rightarrow \\quad
\\Delta \\log(y) \\approx \\beta \\Delta \\log(x).
$$

So a 1% change in $x$ corresponds to about a $\\beta$% change in $y$.

#### 3) Assumptions (and practical caveats)

Log transforms require:
- $x_i > 0$ and $y_i > 0$.

Common issues:
- zeros (log undefined),
- negative values,
- heavy measurement error at small values.

Workarounds (must be justified):
- filter to positive values,
- use `log1p` (changes interpretation),
- use alternative functional forms.

#### 4) Estimation mechanics

Once transformed, you fit OLS on $\\log(y)$ and $\\log(x)$ as usual.
Interpretation should be in percent changes, not raw units.

#### 5) Inference

If heteroskedasticity is present (common in micro), use robust SE (HC3).

#### 6) Diagnostics + robustness (minimum set)

1) **Check positivity**
- count how many observations would be dropped by logging.

2) **Residual diagnostics**
- plot residuals vs fitted; heteroskedasticity is common.

3) **Functional form sensitivity**
- compare log–log to level-level or log-level if meaningful.

#### 7) Interpretation + reporting

Report:
- how you handled zeros,
- whether coefficients are interpreted as elasticities,
- robust SE choice.

#### Exercises

- [ ] Fit a log–log regression and interpret $\\beta$ as an elasticity in words.
- [ ] Compare to a level-level regression; explain how interpretation changes.
- [ ] Demonstrate the “small change” approximation numerically for one observation.
