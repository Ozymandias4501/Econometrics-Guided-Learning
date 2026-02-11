# Guide: 02_calibration_and_costs

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/02_calibration_and_costs.ipynb`.

This guide focuses on **calibration, Brier score, and cost-based decision-making** -- the bridge between a model that ranks risk well and one whose probabilities you can actually trust. A classifier with excellent AUC can still produce misleading probabilities. This guide explains how to diagnose that problem, fix it, and embed explicit cost trade-offs into threshold selection.

### Key Terms (defined)
- **Calibration**: whether predicted probabilities match observed frequencies (e.g., among patients given 30% risk, roughly 30% actually experience the event).
- **Reliability diagram (calibration curve)**: plot of mean predicted probability vs observed frequency in each bin.
- **Brier score**: mean squared error of predicted probabilities; $\frac{1}{n}\sum(p_i - y_i)^2$ (lower is better).
- **Expected calibration error (ECE)**: weighted average absolute gap between predicted and observed frequency across bins.
- **Platt scaling**: post-hoc calibration by fitting a logistic regression on model outputs.
- **Isotonic regression**: non-parametric post-hoc calibration using a monotone step function.
- **Cost-optimal threshold**: $\tau^* = c_{FP}/(c_{FP} + c_{FN})$; the threshold that minimizes expected misclassification cost.
- **Decision curve**: plot of net benefit as a function of threshold probability.

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
- Produce a reliability diagram and interpret it.
- Pick a threshold intentionally (cost-based or metric-based) and justify it.

### Alternative Example: Calibrating a Readmission Risk Model
```python
# Calibration assessment for a hospital readmission model.
# NOT the notebook solution -- illustrates calibration mechanics.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

rng = np.random.default_rng(42)
n = 1000

# Simulate patient features and outcomes
age = rng.normal(68, 10, n)
comorbidities = rng.poisson(2, n)
X = np.column_stack([age, comorbidities])

logit_p = -3.5 + 0.02 * age + 0.45 * comorbidities
p_true = 1 / (1 + np.exp(-logit_p))
y = rng.binomial(1, p_true)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Fit model
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]

# Brier score
bs = brier_score_loss(y_test, probs)
print(f"Brier score: {bs:.4f}")

# Calibration curve
fraction_pos, mean_predicted = calibration_curve(y_test, probs, n_bins=10)
for frac, pred in zip(fraction_pos, mean_predicted):
    print(f"  predicted ~{pred:.2f}, observed {frac:.2f}")

# Post-hoc calibration with Platt scaling
cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
cal_clf.fit(X_train, y_train)
cal_probs = cal_clf.predict_proba(X_test)[:, 1]
print(f"Brier after Platt scaling: {brier_score_loss(y_test, cal_probs):.4f}")
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

> **Prerequisites:** [Classification foundations](00_recession_classifier_baselines.md) -- core classification concepts (probabilities, losses, thresholds, metrics, calibration overview, class imbalance).

### Deep Dive: Calibration and Brier Score (Probabilities You Can Trust)

Calibration is the difference between "good ranking" and "usable probabilities." A model can have perfect AUC (it ranks every positive above every negative) yet be terribly calibrated (it says 90% when the true rate is 30%). When decisions depend on the magnitude of risk -- not just the ordering -- calibration is essential.

#### 1) Intuition: what calibration means in practice

If a model says "30% recession probability" many times, then about 30% of those cases should actually be recessions. If the model says 30% but only 10% materialize, the model is **over-confident**. If 50% materialize, it is **under-confident**.

**Health econ motivation.** A hospital readmission risk model predicts each discharged patient's 30-day readmission probability. A care coordinator uses these probabilities to allocate follow-up resources: patients above 25% risk get a home visit; those between 15-25% get a phone call; those below 15% get standard discharge. If the probabilities are poorly calibrated -- say the model reports 25% for patients who actually have 10% risk -- the coordinator wastes home visits on low-risk patients and misses genuinely high-risk ones. The ranking might be fine (sicker patients still score higher), but the absolute cutoffs are wrong. This is why calibration matters whenever probabilities drive resource allocation.

#### 2) Reliability diagrams, step by step

A reliability diagram (calibration curve) is the primary visual diagnostic.

**How to construct one:**

1. **Obtain predicted probabilities** $\hat{p}_1, \dots, \hat{p}_n$ from the model on held-out data.
2. **Sort and bin** the predictions into $B$ groups (e.g., $B = 10$ decile bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]).
3. **For each bin $b$**, compute:
   - the mean predicted probability: $\bar{p}_b = \frac{1}{|B_b|}\sum_{i \in B_b} \hat{p}_i$
   - the observed frequency: $\bar{y}_b = \frac{1}{|B_b|}\sum_{i \in B_b} y_i$
4. **Plot** $\bar{p}_b$ (x-axis) vs $\bar{y}_b$ (y-axis). A perfectly calibrated model lies on the 45-degree diagonal.

**Reading the plot:**
- Points **above** the diagonal: model is under-confident (says 20%, reality is higher).
- Points **below** the diagonal: model is over-confident (says 40%, reality is lower).
- Bins with very few observations produce noisy points -- always report bin counts.

**Practical choice of bins:** 10 equal-width bins is the default. With small test sets (< 200), use fewer bins (5-8) or use isotonic-style adaptive binning.

#### 3) Brier score and its decomposition

The **Brier score** is the mean squared error of predicted probabilities:

$$
\text{Brier} = \frac{1}{n}\sum_{i=1}^{n} (\hat{p}_i - y_i)^2.
$$

It ranges from 0 (perfect) to 1 (worst). A baseline model that always predicts the prevalence $\pi = \bar{y}$ achieves:

$$
\text{Brier}_{\text{baseline}} = \pi(1-\pi).
$$

**Decomposition (Murphy, 1973).** The Brier score can be decomposed into three components:

$$
\text{Brier} = \underbrace{\text{Reliability}}_{\text{calibration error}} - \underbrace{\text{Resolution}}_{\text{ability to separate}} + \underbrace{\text{Uncertainty}}_{\text{inherent difficulty}}.
$$

- **Reliability** (lower is better): measures how far the calibration curve deviates from the diagonal. This is the part you can fix with post-hoc calibration.
- **Resolution** (higher is better): measures how much predicted probabilities vary across bins. A model that gives the same probability to everyone has zero resolution.
- **Uncertainty** = $\pi(1-\pi)$: depends only on the base rate and cannot be changed by the model.

The decomposition helps diagnose whether a high Brier score comes from poor calibration (fixable) or lack of discrimination (model needs better features).

#### 4) Expected calibration error (ECE)

ECE summarizes the reliability diagram as a single number:

$$
\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left|\bar{y}_b - \bar{p}_b\right|.
$$

Each bin's calibration gap is weighted by the fraction of observations in that bin. Lower ECE means better calibration.

**Limitations:**
- ECE depends on the binning scheme (number and type of bins).
- With equal-width bins, bins near 0 and 1 may be nearly empty.
- Report ECE alongside the reliability diagram, not as a standalone metric.

#### 5) Cost-based threshold selection

A threshold $\tau$ converts probability to a hard label: $\hat{y}_i = 1[\hat{p}_i \ge \tau]$. The optimal threshold depends on the relative costs of false positives and false negatives.

**Worked example.** Suppose missing a recession ($c_{FN}$) is 5 times worse than a false alarm ($c_{FP} = 1$). The cost-optimal threshold is:

$$
\tau^* = \frac{c_{FP}}{c_{FP} + c_{FN}} = \frac{1}{1 + 5} \approx 0.167.
$$

You flag a recession whenever $\hat{p}_i > 0.167$, not the default 0.50. This dramatically increases recall at the cost of more false alarms -- which is the right trade-off when misses are expensive.

**Why this formula works.** The expected cost of a decision at threshold $\tau$ for observation $i$ is:

$$
\text{EC}_i(\tau) = c_{FP} \cdot (1-p_i) \cdot 1[\hat{p}_i \ge \tau] + c_{FN} \cdot p_i \cdot 1[\hat{p}_i < \tau].
$$

Setting the two terms equal and solving for $p_i$ gives the breakeven probability $\tau^*$. Below this probability, the expected cost of acting exceeds the expected cost of not acting.

**Health econ example.** For a readmission model, suppose a preventive home visit costs \$200 ($c_{FP}$, the cost of intervening on a patient who would not have been readmitted) and a readmission costs \$15,000 ($c_{FN}$). Then:

$$
\tau^* = \frac{200}{200 + 15000} \approx 0.013.
$$

You would intervene for nearly everyone -- because readmissions are so expensive relative to the intervention that even a 1.3% risk justifies action. This illustrates why cost analysis sometimes produces surprising thresholds.

**In practice:** plot precision and recall as a function of $\tau$ alongside the cost curve. The cost-optimal threshold assumes calibrated probabilities, so calibrate first.

#### 6) Platt scaling vs isotonic regression

When raw model probabilities are miscalibrated, post-hoc calibration can fix them without retraining the model.

**Platt scaling:**
- Fits a logistic regression $\sigma(a \cdot f(x) + b)$ on the model's raw scores $f(x)$.
- Two parameters only ($a$ and $b$), so it works well with limited validation data.
- Assumes the calibration function is sigmoid-shaped (S-curve mapping from raw scores to probabilities).
- Best for: models whose miscalibration is roughly monotone and smooth (e.g., SVMs, neural nets).

**Isotonic regression:**
- Fits a non-parametric, monotone step function mapping raw scores to calibrated probabilities.
- Very flexible -- can correct any monotone miscalibration pattern.
- Requires more validation data (it can overfit with fewer than ~1,000 calibration samples).
- Best for: large datasets where the calibration function has a complex shape.

**Key trade-off:** Platt scaling has high bias but low variance (only 2 parameters). Isotonic regression has low bias but higher variance (many parameters). With small validation sets (common in health econ with rare outcomes), prefer Platt scaling.

**Important:** Always fit calibration methods on a held-out validation set, never on the test set. Use `CalibratedClassifierCV` in sklearn, which handles the internal cross-validation.

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling (logistic calibration)
cal_platt = CalibratedClassifierCV(base_estimator, method='sigmoid', cv=5)

# Isotonic regression
cal_iso = CalibratedClassifierCV(base_estimator, method='isotonic', cv=5)
```

#### 7) Decision curves and net benefit (brief introduction)

Decision curve analysis extends cost-based thresholding by plotting **net benefit** as a function of the threshold probability $p_t$:

$$
\text{Net Benefit}(p_t) = \frac{TP}{n} - \frac{FP}{n} \cdot \frac{p_t}{1 - p_t}.
$$

The term $p_t / (1-p_t)$ is the odds at the threshold, which represents the exchange rate between false positives and true positives.

**How to read a decision curve:**
- Compare the model's net benefit to two baselines: "treat all" and "treat none."
- The model is useful at threshold $p_t$ if its net benefit exceeds both baselines.
- The range of thresholds where the model adds value defines its clinical utility.

Decision curves are increasingly standard in clinical prediction model research (Vickers & Elkin, 2006). For a full treatment, see the `dcurves` Python package.

#### 8) Alternative example: calibration assessment for a cost-sensitive readmission model

```python
# Full calibration + cost workflow for readmission prediction.
# NOT the notebook solution -- illustrates calibration and cost mechanics.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

rng = np.random.default_rng(7)
n = 800

# Simulate: age, comorbidities, prior_admissions
age = rng.normal(67, 11, n)
comorbidities = rng.poisson(2, n)
prior_admissions = rng.poisson(1, n)
X = np.column_stack([age, comorbidities, prior_admissions])

logit_p = -4.0 + 0.025 * age + 0.4 * comorbidities + 0.3 * prior_admissions
p_true = 1 / (1 + np.exp(-logit_p))
y = rng.binomial(1, p_true)

# Time-like split (first 600 train, last 200 test)
X_tr, X_te = X[:600], X[600:]
y_tr, y_te = y[:600], y[600:]

clf = LogisticRegression(max_iter=5000)
clf.fit(X_tr, y_tr)
probs = clf.predict_proba(X_te)[:, 1]

# --- Brier score ---
brier = brier_score_loss(y_te, probs)
baseline_brier = y_te.mean() * (1 - y_te.mean())
print(f"Brier: {brier:.4f}  (baseline: {baseline_brier:.4f})")

# --- Reliability diagram data ---
frac_pos, mean_pred = calibration_curve(y_te, probs, n_bins=8)
print("\nReliability diagram:")
for fp, mp in zip(frac_pos, mean_pred):
    gap = "over-confident" if mp > fp else "under-confident"
    print(f"  predicted {mp:.2f}, observed {fp:.2f}  ({gap})")

# --- ECE ---
bins = np.linspace(0, 1, 9)  # 8 bins
bin_idx = np.digitize(probs, bins) - 1
ece = 0.0
for b in range(8):
    mask = bin_idx == b
    if mask.sum() == 0:
        continue
    ece += mask.sum() / len(probs) * abs(y_te[mask].mean() - probs[mask].mean())
print(f"\nECE: {ece:.4f}")

# --- Cost-optimal threshold ---
c_fp = 200      # cost of unnecessary home visit
c_fn = 15000    # cost of missed readmission
tau_star = c_fp / (c_fp + c_fn)
print(f"\nCost-optimal threshold: {tau_star:.4f}")
y_hat = (probs >= tau_star).astype(int)
print(f"Flagged {y_hat.sum()} of {len(y_hat)} patients for intervention")
```

#### 9) Diagnostics and robustness

1. **Time-aware evaluation** -- calibration assessed on random splits can be misleadingly good. Use a time split or walk-forward design.
2. **Subperiod calibration** -- check calibration separately in different time windows. Structural change (new treatment protocols, policy shifts) can degrade calibration even if the model was well-calibrated historically.
3. **Bin sensitivity** -- re-run the reliability diagram with different numbers of bins (5, 10, 15). If conclusions change substantially, your test set may be too small for reliable calibration assessment.
4. **Calibration after recalibration** -- after applying Platt or isotonic, re-check the calibration curve on a held-out set. Confirm that the fix actually worked.

### Key Terms (Calibration-Specific)

| Term | Definition |
|------|-----------|
| Reliability | Component of Brier decomposition measuring calibration error |
| Resolution | Component of Brier decomposition measuring the model's ability to separate risk groups |
| Uncertainty | $\pi(1-\pi)$; inherent difficulty determined by the base rate |
| Platt scaling | Sigmoid calibration; 2 parameters; good for small validation sets |
| Isotonic regression | Non-parametric monotone calibration; flexible but needs more data |
| ECE | Expected calibration error; weighted mean absolute calibration gap across bins |
| Net benefit | Decision-theoretic metric balancing true positives against weighted false positives |
| Cost-optimal threshold | $\tau^* = c_{FP}/(c_{FP} + c_{FN})$; minimizes expected misclassification cost |

### Common Mistakes (Calibration-Specific)
- **Evaluating calibration on training data.** Calibration must be assessed out-of-sample. In-sample calibration is nearly meaningless.
- **Calibrating on the test set.** Platt scaling and isotonic regression must be fit on a validation set, not the final test set. Use cross-validated calibration (`CalibratedClassifierCV`).
- **Ignoring bin counts in reliability diagrams.** A bin with 3 observations is noise, not evidence. Always report how many observations fall in each bin.
- **Treating ECE as the only calibration metric.** ECE depends on binning choices. Always pair it with a visual reliability diagram.
- **Assuming calibration is preserved after model changes.** Retraining, adding features, or changing hyperparameters invalidates previous calibration. Re-assess after any model change.
- **Using cost-optimal thresholds without calibrated probabilities.** The formula $\tau^* = c_{FP}/(c_{FP} + c_{FN})$ assumes the model's probabilities are well-calibrated. If they are not, calibrate first, then apply the threshold.

### Exercises

- [ ] Produce a reliability diagram with 10 bins and annotate which bins are over-confident vs under-confident.
- [ ] Compute the Brier score for your model and for a baseline that always predicts the prevalence. How much better is your model?
- [ ] Apply Platt scaling and isotonic regression to the same model. Compare calibration curves and Brier scores. Which method works better on your data, and why?
- [ ] Compute the ECE before and after calibration. By how much did calibration improve?
- [ ] Write a cost story for your application (what are $c_{FP}$ and $c_{FN}$?). Derive $\tau^*$ and report precision/recall at that threshold.
- [ ] Construct a simple decision curve comparing your model to the "treat all" and "treat none" strategies. At which thresholds does the model add value?

<a id="summary"></a>
## Summary + Suggested Readings

You should now be able to:
- construct and interpret a reliability diagram,
- compute and decompose the Brier score,
- apply Platt scaling or isotonic regression and assess whether calibration improved,
- derive a cost-optimal threshold from explicit cost assumptions,
- explain why calibration matters for health-economic decision-making beyond simple ranking.

Suggested readings:
- Niculescu-Mizil & Caruana (2005), "Predicting good probabilities with supervised learning" -- ICML (Platt vs isotonic comparison)
- Murphy (1973), "A new vector partition of the probability score" -- *J. Applied Meteorology* (Brier decomposition)
- Vickers & Elkin (2006), "Decision curve analysis" -- *Medical Decision Making* (net benefit framework)
- Van Calster et al. (2019), "Calibration: the Achilles heel of predictive analytics" -- *BMC Medicine* (modern overview)
- scikit-learn docs: `calibration_curve`, `CalibratedClassifierCV`, `brier_score_loss`
- Steyerberg, *Clinical Prediction Models* -- Ch. 15 (calibration in clinical settings)
