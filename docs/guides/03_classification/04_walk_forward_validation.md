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


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

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

### Deep Dive: Walk-forward validation — repeated “train on past, test on next”

Walk-forward (rolling-origin) validation is the most common evaluation scheme for forecasting.

#### 1) Intuition (plain English)

A single time split answers: “How did I do in one future period?”

Walk-forward answers: “How do I do across many future periods as time moves forward?”

This reduces sensitivity to one arbitrary split boundary and better matches how models are used:
you re-train as new data arrives.

#### 2) Notation + setup (define symbols)

Let:
- $t$ be time,
- $h$ be forecast horizon,
- $W$ be a training window length (optional).

A walk-forward scheme produces folds indexed by $m=1,\\dots,M$:
- Train on $\\{1,\\dots,t_m\\}$ (or last $W$ periods),
- Test on $t_m+h$ (or a short future window).

#### 3) Assumptions

Walk-forward assumes:
- you would realistically update the model over time,
- you respect feature/label timing (no leakage),
- the evaluation horizon $h$ is fixed and meaningful.

#### 4) Estimation mechanics: what it estimates

Walk-forward estimates expected future error by averaging over many future evaluation points:

$$
\\widehat{\\mathrm{Err}} = \\frac{1}{M} \\sum_{m=1}^{M} \\ell(\\hat y_{t_m+h}, y_{t_m+h}).
$$

This is closer to the real “live” error than a random split.

#### 5) Inference and uncertainty

Because folds are time-ordered, fold errors can be correlated.
Treat fold-to-fold variability as descriptive, not as independent draws.

#### 6) Diagnostics + robustness (minimum set)

1) **Plot fold errors over time**
- do errors spike in recessions or structural breaks?

2) **Compare expanding vs rolling windows**
- expanding: more data, potentially stale regimes.
- rolling: focuses on recent regime, higher variance.

3) **Check stability across horizons**
- repeat for $h=1$ vs $h=2$ if relevant.

#### 7) Interpretation + reporting

Report:
- how you constructed folds (expanding/rolling),
- window length (if rolling),
- metrics averaged across folds and variation across time.

#### Exercises

- [ ] Implement walk-forward splits and compute fold-by-fold metrics; plot them over time.
- [ ] Compare expanding vs rolling training windows and interpret the trade-off.
- [ ] Identify a period where the model fails and propose a hypothesis (regime change? missing feature?).

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
