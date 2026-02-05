### Deep Dive: Logistic regression as a probability model (odds and log-odds)

Logistic regression is the baseline probabilistic classifier in econometrics and ML.

#### 1) Intuition (plain English)

We want a model that outputs a probability in (0,1).
Logistic regression does this by modeling **log-odds** as a linear function of features.

**Story example:** Predict next-quarter recession risk.
Probabilities let you compare risk over time and choose thresholds based on costs.

#### 2) Notation + setup (define symbols)

Let:
- $y_i \\in \\{0,1\\}$ be the label,
- $p_i = \\Pr(y_i=1 \\mid x_i)$ be the model probability.

Define odds:
$$
\\text{odds}_i = \\frac{p_i}{1-p_i}.
$$

Define log-odds:
$$
\\log\\left(\\frac{p_i}{1-p_i}\\right).
$$

Logistic regression assumes:

$$
\\log\\left(\\frac{p_i}{1-p_i}\\right) = x_i'\\beta.
$$

Equivalently:

$$
p_i = \\sigma(x_i'\\beta) = \\frac{1}{1 + e^{-x_i'\\beta}}.
$$

#### 3) Assumptions (what this model assumes)

Key assumptions:
- log-odds is linear in features,
- observations are conditionally independent given features (often violated in time series),
- features are not perfectly collinear.

Misspecification is common; the model can still be useful for ranking risk, but calibration must be checked.

#### 4) Estimation mechanics (maximum likelihood in one line)

Logistic regression is typically fit by maximizing the log-likelihood:

$$
\\sum_i \\left[y_i \\log(p_i) + (1-y_i)\\log(1-p_i)\\right].
$$

In practice you use libraries; the key is to interpret outputs correctly.

#### 5) Inference: interpreting coefficients

Coefficient meaning:
- a 1-unit increase in $x_j$ changes log-odds by $\\beta_j$ (holding other features fixed).
- odds multiply by $e^{\\beta_j}$.

Example:
- if $\\beta_j = 0.7$, odds multiply by $e^{0.7} \\approx 2.0$.

Probability marginal effects depend on baseline probability:
- the same $\\beta_j$ can correspond to different probability changes at different risk levels.

#### 6) Diagnostics + robustness (minimum set)

1) **Calibration**
- check whether predicted probabilities match observed frequencies.

2) **Threshold sensitivity**
- precision/recall trade-off depends on threshold; do not default to 0.5.

3) **Feature scaling**
- standardize features when comparing coefficients; otherwise scale drives magnitude.

#### 7) Interpretation + reporting

Report:
- horizon (what event/time),
- calibration (Brier / reliability plot),
- threshold decision rationale if you present class labels.

**What this does NOT mean**
- logistic coefficients are not causal effects without identification,
- a good AUC does not imply calibrated probabilities.

#### Exercises

- [ ] Convert a coefficient to an odds ratio ($e^{\\beta}$) and interpret it in words.
- [ ] Show how a fixed log-odds change maps to different probability changes at p=0.1 vs p=0.9.
- [ ] Fit logistic regression and produce a calibration plot; interpret over/under-confidence.
