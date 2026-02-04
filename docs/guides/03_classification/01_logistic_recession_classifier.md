# Guide: 01_logistic_recession_classifier

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/01_logistic_recession_classifier.ipynb`.

This classification module predicts **next-quarter technical recession** from macro indicators.

### Key Terms (defined)
- **Logistic regression**: a linear model that outputs probabilities via a sigmoid function.
- **Log-odds**: `log(p/(1-p))`; logistic regression is linear in log-odds.
- **Threshold**: rule converting probability into class (e.g., 1 if p>=0.5).
- **Precision/Recall**: trade off false positives vs false negatives.
- **ROC-AUC / PR-AUC**: threshold-free ranking metrics.
- **Calibration**: whether predicted probabilities match observed frequencies.
- **Brier score**: mean squared error of probabilities (lower is better).


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Train/test split
- Complete notebook section: Fit logistic
- Complete notebook section: ROC/PR
- Complete notebook section: Threshold tuning
- Establish baselines before fitting any model.
- Fit at least one probabilistic classifier and evaluate ROC-AUC, PR-AUC, and Brier score.
- Pick a threshold intentionally (cost-based or metric-based) and justify it.

### Alternative Example (Not the Notebook Solution)
```python
# Toy logistic regression (not the notebook data):
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 3))
p = 1 / (1 + np.exp(-(0.2 + 1.0*X[:,0] - 0.8*X[:,1])))
y = rng.binomial(1, p)

clf = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])
clf.fit(X, y)
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Core Classification: Probabilities, Metrics, and Thresholds

In this project, classification is about predicting recession risk as a probability.

#### Logistic regression mechanics
Logistic regression models probabilities via log-odds:

$$
\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k
$$

Then:

$$
 p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots)}}
$$

#### Metrics you should treat as standard
- ROC-AUC: ranking quality across thresholds
- PR-AUC: often more informative when positives are rare
- Brier score (or log loss): probability quality

#### Thresholding is a decision rule
> **Definition:** A **threshold** converts probabilities into labels.

Default 0.5 is rarely optimal for imbalanced, cost-sensitive problems.
Pick thresholds based on:
- decision costs
- desired recall/precision tradeoff
- calibration quality

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

### Project Code Map
- `src/evaluation.py`: classification metrics (ROC-AUC, PR-AUC, Brier, precision/recall)
- `scripts/train_recession.py`: training script that writes artifacts
- `scripts/predict_recession.py`: prediction script that loads artifacts
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Reporting only accuracy (can be misleading if recessions are rare).
- Picking threshold=0.5 by default without considering costs.
- Evaluating with random splits (time leakage).

<a id="summary"></a>
## Summary + Suggested Readings

You should now be able to build a recession probability model and explain:
- what the probability means,
- how you evaluated it, and
- why your chosen threshold makes sense.


Suggested readings:
- scikit-learn docs: classification metrics, calibration
- Murphy: Machine Learning (probabilistic interpretation)
- Applied time-series evaluation articles (walk-forward validation)
