# Guide: 00_setup

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_foundations/00_setup.ipynb`.

This foundations module builds core intuition you will reuse in every later notebook.

### Key Terms (defined)
- **Time series**: data indexed by time; ordering is meaningful and must be respected.
- **Leakage**: using future information in features/labels, producing unrealistically good results.
- **Train/test split**: separating data for model fitting vs evaluation.
- **Multicollinearity**: predictors are highly correlated; coefficients can become unstable.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Environment bootstrap
- Complete notebook section: Verify API keys
- Complete notebook section: Load sample data
- Complete notebook section: Checkpoints
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

### Why Time Ordering Changes Everything
- Random splits destroy chronology. In forecasting tasks, your model must only use information available at prediction time.

### Correlation vs Causation
- **Correlation**: variables move together.
- **Causation**: changing X changes Y (a stronger claim that needs design/assumptions).
- In macro/micro data, you will see many correlated variables. Treat coefficients as *associational* unless you have a causal identification strategy.

### Multicollinearity and VIF
- **VIF (Variance Inflation Factor)**: how much a coefficient's variance is inflated due to collinearity.
- High VIF implies unstable coefficients and wide confidence intervals, even if predictions are okay.


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
