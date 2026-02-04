# Guide: 02_calibration_and_costs

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/02_calibration_and_costs.ipynb`.

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
- Complete notebook section: Calibration
- Complete notebook section: Brier score
- Complete notebook section: Decision costs
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

### Deep Dive: Calibration, Brier Score, and Decision Thresholds

In classification, you often want probabilities, not just labels.

> **Definition:** A model is **calibrated** if events predicted with probability 0.3 happen about 30% of the time.

#### Brier score (math)
> **Definition:** The **Brier score** is a proper scoring rule for probability forecasts.

For binary outcomes $y_i \in \{0,1\}$ and predicted probabilities $p_i$:

$$
\mathrm{Brier} = \frac{1}{n} \sum_{i=1}^n (p_i - y_i)^2
$$

Lower is better.

#### Why calibration matters for recession risk
A recession probability model is only useful if you can make decisions from its probabilities:
- allocate risk
- run stress tests
- change thresholds based on costs

If probabilities are not calibrated, "30%" and "70%" are not meaningful signals.

#### Calibration curve (reliability diagram)
A calibration curve groups predictions into bins and compares:
- average predicted probability in the bin
- actual fraction of positives in the bin

If the curve follows the diagonal, calibration is good.

#### Python demo: calibration and Brier (commented)
```python
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# y_true: 0/1 outcomes
# y_prob: predicted probabilities

# Example placeholders:
# y_true = np.array([...])
# y_prob = np.array([...])

# Brier score
# print('brier:', brier_score_loss(y_true, y_prob))

# Calibration curve
# prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
# print(prob_pred)
# print(prob_true)
```

#### Thresholds and decision costs
> **Definition:** A **decision threshold** converts probabilities to class labels (e.g., predict recession if p >= 0.4).

A good threshold depends on costs:
- false positives (crying wolf)
- false negatives (missing a recession)

A common pattern:
1. define a cost ratio (how bad is FN vs FP?)
2. choose threshold to minimize expected cost

#### Debug checklist
1. Always compute base rate (how rare is the positive class?).
2. Report PR-AUC and Brier score (not just accuracy).
3. Compare calibrated vs uncalibrated models.
4. Re-check calibration across eras (walk-forward).

#### Project touchpoints (where you will use these)
- `src/evaluation.py` computes classification metrics including ROC-AUC, PR-AUC, and Brier score.
- The calibration notebook asks you to plot a reliability diagram and pick a threshold based on explicit costs.

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
