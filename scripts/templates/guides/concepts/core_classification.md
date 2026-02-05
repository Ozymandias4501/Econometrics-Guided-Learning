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
