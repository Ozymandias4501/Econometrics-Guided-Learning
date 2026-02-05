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


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

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

### Core Classification: probabilities, losses, and decision thresholds

In this repo, classification means: predict **recession risk** as a probability and make decisions with explicit trade-offs.

#### 1) Intuition (plain English)

Binary labels (recession vs not) hide uncertainty.
The useful object is the probability:
- “Given data today, how likely is a recession next quarter?”

Probabilities let you:
- compare risk over time,
- set thresholds based on costs,
- evaluate calibration (whether 30% means ~30% in reality).

#### 2) Notation + setup (define symbols)

Let:
- $y_i \\in \\{0,1\\}$ be the true label (1 = recession),
- $x_i$ be features,
- $p_i = \\Pr(y_i=1 \\mid x_i)$ be the model probability.

Logistic regression uses the log-odds (“logit”) link:

$$
\\log\\left(\\frac{p_i}{1-p_i}\\right) = x_i'\\beta.
$$

Equivalently:

$$
p_i = \\sigma(x_i'\\beta) = \\frac{1}{1 + e^{-x_i'\\beta}}.
$$

**What each term means**
- $\\sigma(\\cdot)$ maps real numbers to (0,1).
- coefficients move probabilities through the log-odds scale.

#### 3) Assumptions (and what “probability model” means)

Logistic regression assumes:
- a linear relationship in log-odds,
- observations are conditionally independent given $x$ (often violated in time series),
- no perfect multicollinearity in features.

Even if the model is misspecified, it can still be useful for ranking risk.
But calibration can suffer, so we measure it.

#### 4) Estimation mechanics (how the model is fit)

Logistic regression is typically fit by maximum likelihood:
- choose $\\beta$ to maximize the probability of the observed labels.

The negative log-likelihood corresponds to **log loss** (cross-entropy):

$$
\\ell(\\beta) = -\\sum_i \\left[y_i \\log(p_i) + (1-y_i)\\log(1-p_i)\\right].
$$

In practice you use libraries (`sklearn` or `statsmodels`) rather than coding this by hand.

#### 5) Inference vs prediction

- `statsmodels` gives standard errors and p-values (inference framing).
- `sklearn` focuses on predictive performance (pipelines, CV, regularization).

In this project:
- prioritize time-aware out-of-sample evaluation,
- treat inference outputs as descriptive unless you have identification.

#### 6) Metrics (what to measure and why)

At minimum, treat these as standard:

- **ROC-AUC:** ranking performance (threshold-free).
- **PR-AUC:** often more informative when positives are rare.
- **Brier score:** mean squared error of probabilities:
$$
\\text{Brier} = \\frac{1}{n} \\sum_i (p_i - y_i)^2.
$$
- **Calibration plots:** do predicted probabilities match observed frequencies?

#### 7) Thresholding is a decision rule (not a model property)

A threshold $\\tau$ converts probability to a hard label:
$$
\\hat y_i = 1[p_i \\ge \\tau].
$$

Choosing $\\tau$ should reflect costs:
- false positives (crying wolf),
- false negatives (missing recessions).

#### 8) Diagnostics + robustness (minimum set)

1) **Time-aware evaluation**
- use a time split or walk-forward; avoid random splits for forecasting tasks.

2) **Calibration**
- plot predicted vs observed probabilities; compute Brier score.

3) **Threshold sensitivity**
- show how precision/recall changes with threshold.

4) **Feature stability**
- check whether model performance is stable over subperiods (structural change).

#### 9) Interpretation + reporting

Report:
- horizon (what “next quarter” means),
- split method and dates,
- probability calibration (not just accuracy),
- threshold choice rationale.

**What this does NOT mean**
- AUC does not tell you if probabilities are calibrated.
- A good backtest does not guarantee future performance in a new regime.

#### Exercises

- [ ] Fit a classifier and report ROC-AUC, PR-AUC, and Brier; explain what each measures.
- [ ] Produce a calibration plot and interpret whether probabilities are over/under-confident.
- [ ] Choose a threshold based on a cost story (false negative vs false positive) and justify it.
- [ ] Compare random-split vs time-split AUC and explain the difference.

### Deep Dive: Calibration and Brier score (probabilities you can trust)

Calibration is the difference between “good ranking” and “usable probabilities.”

#### 1) Intuition (plain English)

If a model says “30% recession probability” many times, then about 30% of those cases should actually be recessions.
If not, the model is miscalibrated (over- or under-confident).

#### 2) Notation + setup (define symbols)

Let:
- $p_i$ be predicted probability,
- $y_i \\in \\{0,1\\}$ be the realized label.

Brier score:

$$
\\text{Brier} = \\frac{1}{n}\\sum_{i=1}^{n} (p_i - y_i)^2.
$$

**What it measures**
- probability mean squared error (lower is better).

Calibration curve:
- bin predictions into groups (e.g., 0.0–0.1, 0.1–0.2, …),
- compare average predicted probability vs empirical frequency in each bin.

#### 3) Assumptions

Calibration assessment assumes:
- evaluation is out-of-sample (time split / walk-forward),
- enough data in bins (small samples produce noisy curves),
- label definition is stable.

#### 4) Estimation mechanics: why calibration can fail

Even if a classifier ranks well, probabilities can be miscalibrated due to:
- regularization strength,
- class imbalance,
- dataset shift (new regime),
- model misspecification.

Calibration methods:
- Platt scaling (logistic calibration),
- isotonic regression.

These should be fit on validation data, not on the test set.

#### 5) Inference: decisions require calibrated probabilities

If you use probabilities for decisions (alerts, risk management), calibration matters more than AUC.
AUC only checks ranking, not absolute probability accuracy.

#### 6) Diagnostics + robustness (minimum set)

1) **Calibration curve + Brier**
- always pair a curve with a scalar metric.

2) **Subperiod calibration**
- check calibration separately in different eras (structural change).

3) **Threshold sensitivity**
- miscalibration can shift optimal thresholds across time.

#### 7) Interpretation + reporting

Report:
- Brier score,
- calibration plot (with bin counts),
- whether calibration was improved with post-processing (and how).

#### Exercises

- [ ] Produce a calibration curve and identify whether the model is over- or under-confident.
- [ ] Compute Brier score and compare to a baseline (constant probability = prevalence).
- [ ] Fit a calibration method on validation data and re-check calibration on test data.

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
