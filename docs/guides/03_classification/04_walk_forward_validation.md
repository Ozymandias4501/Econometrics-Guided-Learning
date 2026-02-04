# Guide: 04_walk_forward_validation

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/04_walk_forward_validation.ipynb`.

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
- Complete notebook section: Walk-forward splits
- Complete notebook section: Metric stability
- Complete notebook section: Failure analysis
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

### Deep Dive: Walk-Forward Validation (Stability Over Time)

> **Definition:** **Walk-forward validation** repeatedly trains on the past and tests on the next time block.

It answers: "Does my model work across multiple eras, or only in one?"

#### Why walk-forward matters in economics
Economic relationships shift:
- policy changes
- technology shifts
- measurement changes
- financial crises

A single split can hide fragility.

#### Procedure (expanding window)
Typical expanding-window walk-forward:
- fold 1: train [0:t1], test [t1:t2]
- fold 2: train [0:t2], test [t2:t3]
- ...

> **Definition:** An **expanding window** keeps all past data in training.

> **Definition:** A **rolling window** uses only the most recent fixed-size window for training.

#### Pseudo-code
```python
# for each fold:
#   train = data[:train_end]
#   test  = data[train_end:train_end+test_size]
#   fit model on train
#   evaluate on test
#   advance train_end
```

#### Project touchpoints (where walk-forward is implemented)
- `src/evaluation.py` implements `walk_forward_splits` for fold generation.
- The walk-forward notebook uses this helper and asks you to plot metrics by era.

```python
from src.evaluation import walk_forward_splits

# Example: quarterly data with ~120 points
n = 120
splits = list(walk_forward_splits(n, initial_train_size=40, test_size=8))
splits[:3]
```

#### What to interpret
- If metrics vary widely across folds, the model is regime-sensitive.
- If performance collapses in certain periods, analyze what changed:
  - indicator behavior
  - label definition
  - missing data

#### Debug checklist
1. Ensure each fold trains strictly on the past.
2. Avoid reusing test periods for tuning.
3. Plot metrics over time, not just averages.
4. Keep the feature engineering fixed when comparing across folds.

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
