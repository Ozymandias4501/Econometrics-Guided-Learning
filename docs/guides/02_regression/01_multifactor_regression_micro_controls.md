# Guide: 01_multifactor_regression_micro_controls

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/01_multifactor_regression_micro_controls.ipynb`.

This regression module covers both prediction and inference, with a strong emphasis on interpretation.

### Key Terms (defined)
- **OLS (Ordinary Least Squares)**: chooses coefficients that minimize squared prediction errors.
- **Coefficient**: expected change in the target per unit change in a feature (holding others fixed).
- **Standard error (SE)**: uncertainty estimate for a coefficient.
- **p-value**: probability of observing an effect at least as extreme if the true effect were zero (under assumptions).
- **Confidence interval (CI)**: a range of plausible coefficient values under assumptions.
- **Heteroskedasticity**: non-constant error variance; common in cross-section.
- **Autocorrelation**: errors correlated over time; common in time series.
- **HAC/Newey-West**: robust SE for time-series autocorrelation/heteroskedasticity.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Choose controls
- Complete notebook section: Fit model
- Complete notebook section: Compare coefficients
- Fit at least one plain OLS model and one robust-SE variant (HC3 or HAC).
- Interpret coefficients in units (or standardized units) and explain what they do *not* mean.
- Run at least one diagnostic: residual plot, VIF table, or rolling coefficient stability plot.

### Alternative Example (Not the Notebook Solution)
```python
# Toy OLS with robust SE (not the notebook data):
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)
x = rng.normal(size=200)
y = 2.0 + 0.5*x + rng.normal(scale=1 + 0.5*np.abs(x), size=200)  # heteroskedastic errors
X = sm.add_constant(pd.DataFrame({'x': x}))
res = sm.OLS(y, X).fit()
res_hc3 = res.get_robustcov_results(cov_type='HC3')
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: Omitted Variable Bias (Why Adding Controls Changes Coefficients)

**Omitted variable bias (OVB)** happens when:
1) you omit a variable Z that affects Y, and
2) Z is correlated with an included regressor X.

Then the coefficient on X partly absorbs Z's effect.

#### Key Terms (defined)
- **Confounder**: a variable related to both X and Y.
- **Control variable**: a variable included to reduce confounding.
- **Specification**: the set of variables you include in a regression.

#### Python demo: a confounder makes X look important
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)
n = 2000

# Z affects Y, and Z also affects X.
z = rng.normal(size=n)
x = 0.8*z + rng.normal(scale=1.0, size=n)
y = 2.0*z + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({'y': y, 'x': x, 'z': z})

# Omitted Z: biased coefficient on x
res_omit = sm.OLS(df['y'], sm.add_constant(df[['x']])).fit()

# Include Z: coefficient on x shrinks toward 0
res_full = sm.OLS(df['y'], sm.add_constant(df[['x', 'z']])).fit()

print('omit z:', res_omit.params)
print('full  :', res_full.params)
```

#### Practical rule
- If your coefficient flips sign or changes drastically when adding plausible controls,
  your original interpretation was likely fragile.


### Deep Dive: Robust Standard Errors (HC3) for Cross-Sectional Data

In cross-sectional economics, heteroskedasticity is common: richer counties often have different variance in outcomes.
Naive OLS standard errors assume constant variance; HC3 relaxes that.

#### Key Terms (defined)
- **Heteroskedasticity**: error variance changes with x.
- **Robust SE**: covariance estimates that remain valid under certain violations.
- **HC3**: a popular heteroskedasticity-robust SE variant (often conservative).

#### What changes when you use HC3?
- Coefficients (beta) do not change.
- Standard errors / p-values / confidence intervals change.

#### Python demo: heteroskedastic errors and robust SE
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)
n = 400
x = rng.normal(size=n)

# Error variance increases with |x|
eps = rng.normal(scale=1 + 2*np.abs(x), size=n)
y = 1.0 + 0.5*x + eps

X = sm.add_constant(pd.DataFrame({'x': x}))
res = sm.OLS(y, X).fit()
res_hc3 = res.get_robustcov_results(cov_type='HC3')

print('naive SE:', res.bse)
print('HC3 SE  :', res_hc3.bse)
```

#### Interpretation warning
- A small p-value is not a causal certificate.
- Robust SE helps with *uncertainty* under heteroskedasticity, not with confounding.


### OLS Objective
- Model: `y = Xβ + ε`
- OLS chooses `β` to minimize `Σ (y_i - ŷ_i)^2`.

### Interpreting Coefficients
- In a simple regression with one feature, the slope is the expected change in `y` for a +1 change in `x`.
- In a multi-factor regression, the slope is the expected change in `y` for a +1 change in `x_j` *holding other X fixed*.
- If features are correlated (multicollinearity), "holding others fixed" can be a fragile, unrealistic counterfactual.

### Inference vs Prediction
- Inference: emphasize coefficient uncertainty and assumptions.
- Prediction: emphasize out-of-sample performance.
- You can have strong prediction with weak/unstable coefficients.

### Robust Standard Errors
- **HC3** addresses heteroskedasticity (common for cross-sectional county data).
- **HAC/Newey-West** addresses autocorrelation + heteroskedasticity (common for quarterly macro time series).


### Common Mistakes
- Interpreting a coefficient as causal without a causal design.
- Ignoring multicollinearity (high VIF) and over-trusting coefficient signs.
- Using naive SE on time series and over-trusting p-values.

<a id="summary"></a>
## Summary + Suggested Readings

Regression is the core bridge between statistics and ML. You should now be able to:
- fit interpretable linear models,
- quantify uncertainty (robust SE), and
- diagnose when coefficients are unstable.


Suggested readings:
- Wooldridge: Introductory Econometrics (OLS, robust SE, interpretation)
- Angrist & Pischke: Mostly Harmless Econometrics (causal thinking)
- statsmodels docs: robust covariance (HCx, HAC)
