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
