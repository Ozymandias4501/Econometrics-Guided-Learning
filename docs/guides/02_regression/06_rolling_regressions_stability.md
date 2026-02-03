# Guide: 06_rolling_regressions_stability

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/06_rolling_regressions_stability.ipynb`.

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
- Complete notebook section: Rolling regression
- Complete notebook section: Coefficient drift
- Complete notebook section: Regime interpretation
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

### Deep Dive: Rolling Regressions (Stability and Structural Breaks)

A rolling regression repeatedly re-fits a model on a moving window of past data.
This is a practical way to detect coefficient drift and regime sensitivity.

#### Key Terms (defined)
- **Rolling window**: a fixed-size window that moves forward through time.
- **Expanding window**: a window that grows over time (always includes all past).
- **Structural break**: the relationship between X and Y changes.
- **Regime**: an era where relationships are relatively stable.

#### Why this matters in macro
- Policy regimes change.
- Financial structure changes.
- Data definitions change.
A single "global" coefficient can hide these shifts.

#### Python demo: relationship changes mid-sample
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)
n = 200
x = rng.normal(size=n)

# Coefficient changes halfway
beta = np.r_[np.repeat(0.2, n//2), np.repeat(-0.2, n - n//2)]
y = 1.0 + beta*x + rng.normal(scale=1.0, size=n)

idx = pd.date_range('1970-03-31', periods=n, freq='QE')
df = pd.DataFrame({'y': y, 'x': x}, index=idx)

window = 60
betas = []
dates = []
for end in range(window, len(df)+1):
    sub = df.iloc[end-window:end]
    res = sm.OLS(sub['y'], sm.add_constant(sub[['x']])).fit()
    betas.append(res.params['x'])
    dates.append(sub.index[-1])

out = pd.Series(betas, index=dates)
out.head()
```

#### Interpretation
- If coefficients drift, your model is not describing a single stable mechanism.
- For prediction, you may prefer recent windows.
- For inference, you must be careful about claiming a single "effect" across eras.


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
