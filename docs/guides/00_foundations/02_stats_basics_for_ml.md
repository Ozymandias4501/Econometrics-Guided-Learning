# Guide: 02_stats_basics_for_ml

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_foundations/02_stats_basics_for_ml.ipynb`.

This foundations module builds core intuition you will reuse in every later notebook.

### Key Terms (defined)
- **Time series**: data indexed by time; ordering is meaningful and must be respected.
- **Leakage**: using future information in features/labels, producing unrealistically good results.
- **Train/test split**: separating data for model fitting vs evaluation.
- **Multicollinearity**: predictors are highly correlated; coefficients can become unstable.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Correlation vs causation
- Complete notebook section: Multicollinearity (VIF)
- Complete notebook section: Bias/variance
- Run the bootstrap cell and confirm `PROJECT_ROOT` points to the repo root.
- Complete all TODOs (no `...` left).
- Write a short paragraph explaining a leakage example you created.

### Alternative Example (Not the Notebook Solution)
```python
# Toy leakage example (not the notebook data):
import numpy as np
import pandas as pd

idx = pd.date_range('2020-01-01', periods=200, freq='D')
y = pd.Series(np.sin(np.linspace(0, 12, len(idx))) + 0.1*np.random.randn(len(idx)), index=idx)

# Correct feature: yesterday
x_lag1 = y.shift(1)

# Leakage feature: tomorrow (do NOT do this)
x_leak = y.shift(-1)
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

This notebook is about the statistical failure modes that make models look smart when they are not.

### Deep Dive: Correlation vs Causation (Practical)
- **Correlation** answers: do X and Y move together?
- **Causation** answers: if we intervene on X, does Y change?
- Most observational economic datasets support correlation claims by default, not causal claims.

### Deep Dive: Overfitting (What It Looks Like)
- Overfitting is when a model learns noise specific to the training sample.
- Symptom: training performance improves while test performance stagnates or worsens.
- Fixes: simpler models, more data, regularization, better features, better splits.

### Deep Dive: Multicollinearity (Why Coefficients Become Unstable)

**Multicollinearity** means two or more predictors contain overlapping information (they are highly correlated).

**What multicollinearity does (and does not do)**
- It often does **not** hurt prediction much.
- It **does** make individual coefficients unstable and their standard errors large.
- It makes p-values fragile: you can flip signs or lose significance by adding/removing correlated features.

**Variance Inflation Factor (VIF)**
- VIF for feature j: how much the variance of β_j is inflated because x_j is correlated with other features.
- Rule of thumb: VIF > 5 (or 10) suggests serious multicollinearity.

**Python demo: correlated predictors -> unstable coefficients**
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.econometrics import vif_table

rng = np.random.default_rng(0)
n = 500
x1 = rng.normal(size=n)
x2 = x1 * 0.95 + rng.normal(scale=0.2, size=n)  # highly correlated with x1
y = 1.0 + 2.0*x1 + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
print(vif_table(df, ['x1', 'x2']))

X = sm.add_constant(df[['x1', 'x2']])
res = sm.OLS(df['y'], X).fit()
print(res.params)
print(res.bse)
```

**Mitigations (practical)**
- Drop one of the correlated variables (choose based on interpretability).
- Combine them (PCA/factors, or domain-driven composite indices).
- Use regularization (ridge tends to stabilize coefficients).

**Economic interpretation warning**
- Macro indicators often move together.
- If two indicators are highly correlated, "holding one fixed" is not an economically realistic counterfactual.
- Treat coefficients as conditional correlations, not causal effects.


### Common Mistakes
- Using `train_test_split(shuffle=True)` on time-indexed data.
- Looking at the test set repeatedly while tuning ("test leakage").
- Assuming a significant p-value implies causation.

<a id="summary"></a>
## Summary + Suggested Readings

You now have the tooling to avoid the two most common beginner mistakes in economic ML:
1) leaking future information, and
2) over-interpreting correlated coefficients.


Suggested readings:
- Hyndman & Athanasopoulos: Forecasting: Principles and Practice (time series basics)
- Wooldridge: Introductory Econometrics (interpretation + pitfalls)
- scikit-learn docs: model evaluation and cross-validation
