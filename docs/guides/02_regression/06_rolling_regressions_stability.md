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

### Deep Dive: Rolling Regressions (Stability and Structural Breaks)

A rolling regression re-fits a model on a moving window of past data.

#### Key terms (defined)
> **Definition:** A **rolling window** is a fixed-size window that moves forward through time.

> **Definition:** A **structural break** is when the relationship between X and Y changes.

#### Why rolling regressions matter in macro
Relationships can change across eras.
A single coefficient can hide that instability.

#### Python demo: coefficient changes over time (commented)
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)

n = 200
x = rng.normal(size=n)

# Coefficient changes halfway
beta = np.r_[np.repeat(0.2, n//2), np.repeat(-0.2, n - n//2)]
y = 1.0 + beta * x + rng.normal(scale=1.0, size=n)

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

beta_series = pd.Series(betas, index=dates)
print(beta_series.head())
```

#### Interpretation
If coefficients drift:
- the model is not describing a single stable mechanism
- for prediction, you may want to weight recent history more
- for inference, be cautious about a single "effect" claim

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`) + multicollinearity (`vif_table`)
- `src/macro.py`: GDP + labels (`gdp_growth_*`, `technical_recession_label`)
- `src/evaluation.py`: regression metrics helpers
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

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
