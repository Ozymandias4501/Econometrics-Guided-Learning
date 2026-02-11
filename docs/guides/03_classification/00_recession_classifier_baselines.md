# Guide: 00_recession_classifier_baselines

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/00_recession_classifier_baselines.ipynb`.

This classification module establishes **baseline classifiers** before fitting any real model, and covers the evaluation pitfalls that arise when the positive class is rare (class imbalance).

### Key Terms (defined)
- **Baseline classifier**: a simple rule (e.g., always predict the majority class) that sets a performance floor; any real model must beat this.
- **Prevalence**: the proportion of positive cases in the data, $\pi = \Pr(y=1)$.
- **Class imbalance**: when one class dominates the data (e.g., 90% negative, 10% positive).
- **Stratification**: splitting data so that train and test sets preserve the original class ratio.
- **SMOTE (Synthetic Minority Oversampling Technique)**: generates synthetic minority-class examples by interpolating between existing ones.
- **Precision/Recall**: trade off false positives vs false negatives.
- **ROC-AUC / PR-AUC**: threshold-free ranking metrics.
- **Brier score**: mean squared error of probabilities (lower is better).

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Load data
- Complete notebook section: Define baselines
- Complete notebook section: Evaluate metrics
- Establish baselines (majority-class, prevalence-based, stratified dummy) before fitting any model.
- Evaluate each baseline on ROC-AUC, PR-AUC, Brier score, and accuracy to see which metrics are fooled by naive strategies.
- Use stratified splits so that train and test sets reflect the true class ratio.

### Alternative Example (Not the Notebook Solution)
```python
# Baseline classifiers for imbalanced binary data (not the notebook data):
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, brier_score_loss, roc_auc_score)

rng = np.random.default_rng(42)

# Simulate imbalanced data: ~12% positive (e.g., hospital readmissions)
n = 1000
X = rng.normal(size=(n, 4))
y = rng.binomial(1, 0.12, size=n)

# Stratified train/test split preserves class ratio
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
train_idx, test_idx = next(splitter.split(X, y))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Baseline 1: always predict the majority class
majority = DummyClassifier(strategy='most_frequent')
majority.fit(X_train, y_train)
y_pred_maj = majority.predict(X_test)

# Baseline 2: predict class proportions (prevalence-based)
prevalence = DummyClassifier(strategy='stratified', random_state=0)
prevalence.fit(X_train, y_train)
y_pred_prev = prevalence.predict(X_test)

# Compare: accuracy looks great for majority-class, but recall is 0
print(f"Majority-class  — Acc: {accuracy_score(y_test, y_pred_maj):.2f}, "
      f"Recall: {recall_score(y_test, y_pred_maj):.2f}")
print(f"Prevalence-based — Acc: {accuracy_score(y_test, y_pred_prev):.2f}, "
      f"Recall: {recall_score(y_test, y_pred_prev):.2f}")
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Core Classification: probabilities, losses, and decision thresholds

> **Canonical reference.** This section is the single authoritative classification primer for this project. Guides 01 (logistic regression), 02 (calibration and costs), 03 (tree models), and 04 (walk-forward validation) cover specific topics and cross-reference this section for foundational concepts. If you need the basics of probabilistic classification, metrics, or thresholding, start here.

In this repo, classification means: predict **event risk** as a probability and make decisions with explicit trade-offs.

#### 1) Intuition (plain English)

Binary labels (event vs non-event) hide uncertainty.
The useful object is the probability:
- "Given data today, how likely is a recession next quarter?"
- "Given patient characteristics, how likely is a 30-day hospital readmission?"

Probabilities let you:
- compare risk over time or across patients,
- set thresholds based on costs,
- evaluate calibration (whether 30% means ~30% in reality).

#### 2) Notation + setup (define symbols)

Let:
- $y_i \in \{0,1\}$ be the true label (1 = positive event),
- $x_i$ be features,
- $p_i = \Pr(y_i=1 \mid x_i)$ be the model probability.

Logistic regression uses the log-odds ("logit") link:

$$
\log\left(\frac{p_i}{1-p_i}\right) = x_i'\beta.
$$

Equivalently:

$$
p_i = \sigma(x_i'\beta) = \frac{1}{1 + e^{-x_i'\beta}}.
$$

**What each term means**
- $\sigma(\cdot)$ maps real numbers to (0,1).
- coefficients move probabilities through the log-odds scale.

#### 3) Assumptions (and what "probability model" means)

Logistic regression assumes:
- a linear relationship in log-odds,
- observations are conditionally independent given $x$ (often violated in time series),
- no perfect multicollinearity in features.

Even if the model is misspecified, it can still be useful for ranking risk.
But calibration can suffer, so we measure it.

#### 4) Estimation mechanics (how the model is fit)

Logistic regression is typically fit by maximum likelihood:
- choose $\beta$ to maximize the probability of the observed labels.

The negative log-likelihood corresponds to **log loss** (cross-entropy):

$$
\ell(\beta) = -\sum_i \left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right].
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
\text{Brier} = \frac{1}{n} \sum_i (p_i - y_i)^2.
$$
- **Calibration plots:** do predicted probabilities match observed frequencies?

#### 7) Thresholding is a decision rule (not a model property)

A threshold $\tau$ converts probability to a hard label:
$$
\hat y_i = 1[p_i \ge \tau].
$$

Choosing $\tau$ should reflect costs:
- false positives (crying wolf),
- false negatives (missing the event).

See Guide 02 (calibration and costs) for a full treatment of cost-sensitive threshold selection.

#### 8) Diagnostics + robustness (minimum set)

1) **Time-aware evaluation** — use a time split or walk-forward; avoid random splits for forecasting tasks.
2) **Calibration** — plot predicted vs observed probabilities; compute Brier score.
3) **Threshold sensitivity** — show how precision/recall changes with threshold.
4) **Feature stability** — check whether model performance is stable over subperiods (structural change).

#### 9) Interpretation + reporting

Report: prediction horizon, split method and dates, probability calibration (not just accuracy), and threshold choice rationale.

**What this does NOT mean**
- AUC does not tell you if probabilities are calibrated.
- A good backtest does not guarantee future performance in a new regime.

#### Exercises
- [ ] Fit a classifier and report ROC-AUC, PR-AUC, and Brier; explain what each measures.
- [ ] Produce a calibration plot and interpret whether probabilities are over/under-confident.
- [ ] Choose a threshold based on a cost story (false negative vs false positive) and justify it.
- [ ] Compare random-split vs time-split AUC and explain the difference.

### Deep Dive: Class imbalance, baselines, and resampling (why accuracy is misleading)

Rare events are the norm in health economics and macro forecasting. Hospital readmissions occur in roughly 10-15% of discharges; recessions cover roughly 5-10% of quarters. Rare events require different evaluation habits and deliberate baseline-setting before you fit any real model.

#### 1) Intuition: the majority-class trap

If 90% of patients are NOT readmitted, a model that always predicts "not readmitted" achieves 90% accuracy. That accuracy is useless — the model catches zero actual readmissions.

This is the **majority-class trap**: accuracy rewards a classifier for simply echoing the dominant class. The more imbalanced the data, the more misleading accuracy becomes.

**Why this happens mathematically.** Accuracy is:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}.
$$

When negatives dominate, $TN$ is large regardless of what the model does with positives. A model with $TP = 0$ and $FN = \text{all positives}$ still has high accuracy because $TN$ overwhelms the numerator.

**Health econ example: hospital readmissions.** Suppose a hospital has 10,000 discharges per year and a 12% readmission rate (1,200 readmissions). A model that predicts "no readmission" for every patient achieves 88% accuracy but identifies zero patients for intervention. Meanwhile, each preventable readmission costs the hospital $15,000-$25,000 in penalties and care under CMS penalty programs. The "accurate" model is operationally worthless.

The real question is not "what fraction of predictions are correct?" but rather "of the patients we flagged, how many truly needed follow-up?" (precision) and "of the patients who were readmitted, how many did we catch?" (recall). These are the metrics that map directly to resource allocation decisions: how many care coordinators to hire, which patients to call after discharge, and how to prioritize limited transitional care budgets.

So we use metrics that focus on:
- ranking risk (AUC),
- detecting positives (recall),
- avoiding false alarms (precision),
- probability quality (Brier/log loss).

#### 2) Baselines you should always establish

Before fitting any real model, compute these naive benchmarks so you know the floor:

- **Majority-class classifier.** Always predict the dominant class. Accuracy floor; recall for the minority class is always 0.
- **Prevalence-based (stratified dummy) classifier.** Predict each class with probability equal to its prevalence. ROC-AUC = 0.50 by construction (random ranking). Any real model must beat this.
- **Constant-probability classifier.** Predict $\hat{p}_i = \pi$ for every observation. Brier score floor = $\pi(1-\pi)$. A model using features should achieve strictly lower Brier.

Why baselines matter: if your logistic regression achieves ROC-AUC = 0.52 and the dummy achieves 0.50, you have barely learned anything. Without baselines, you cannot tell whether a metric value represents genuine signal or just reflects class proportions.

#### 3) Notation + setup

Confusion matrix: TP (true positives), FP (false positives), TN (true negatives), FN (false negatives). Key derived metrics:

$$
\text{Precision} = \frac{TP}{TP + FP}
\qquad
\text{Recall} = \frac{TP}{TP + FN}
\qquad
\text{F1} = 2 \cdot \frac{\text{Precision}\cdot \text{Recall}}{\text{Precision}+\text{Recall}}
$$

Prevalence: $\pi = \Pr(y=1)$.

#### 4) Assumptions (what metrics assume)

Metrics assume:
- you evaluate on future-like data (time-aware or stratified splits),
- labels are correctly aligned to the prediction horizon,
- the positive class definition is stable across train and test,
- class proportions in the test set reflect the deployment environment (do NOT resample the test set).

#### 5) Ranking vs thresholding

Two distinct evaluation tasks:
- **Ranking:** can the model rank high-risk cases above low-risk? (ROC-AUC, PR-AUC)
- **Decisions:** choose a threshold $\tau$ and act (precision/recall at $\tau$).

PR-AUC is often more informative than ROC-AUC when positives are rare, because ROC-AUC can look good even when the model generates many false positives among the large negative pool. Intuitively, ROC-AUC measures how well the model separates classes overall, but the false positive rate denominator ($FP / (FP + TN)$) is dominated by the large $TN$ count, masking the absolute number of false alarms. PR-AUC uses precision ($TP / (TP + FP)$), which is not diluted by $TN$ and directly answers: "Of the patients I flagged as high-risk, how many actually were?"

#### 6) Stratification: preserving class proportions in splits

When the positive class is rare, naive random splits can produce test sets with very few (or zero) positive examples, making metric estimates unreliable. **Stratified splitting** ensures each fold or split preserves the original class ratio.

In sklearn, use `StratifiedShuffleSplit` or `StratifiedKFold`. For time-series data where chronological ordering matters, stratification is less straightforward — use time-based splits with a check that both train and test contain sufficient positive examples.

Health econ note: in hospital readmission studies, stratification also matters when outcomes vary by subgroup (e.g., surgical vs medical patients). If your test set happens to oversample a low-readmission subgroup, your metrics will look artificially good.

#### 7) Resampling strategies for imbalanced classes

When the minority class is small, the model may not see enough positive examples to learn useful patterns. Three common strategies address this — each with trade-offs.

| Strategy | How it works | Pros | Cons | Best when |
|---|---|---|---|---|
| **SMOTE** | Creates synthetic minority examples by interpolating between nearest minority neighbors in feature space | Increases minority representation without discarding data | Can generate noisy points if minority examples are scattered; does not add genuinely new information; can blur decision boundaries | Moderate imbalance (5-30% minority), 50+ minority examples |
| **Random undersampling** | Randomly discards majority examples until classes are closer to balanced | Simple; reduces training time; works well when majority class has redundant examples | Discards potentially useful data; results vary depending on which examples are kept | Abundant majority data; homogeneous majority class |
| **Class weights** | Changes the loss function to penalize minority-class errors more heavily (`class_weight='balanced'` in sklearn) | No data manipulation; uses all data; directly controls precision/recall trade-off | Can produce poorly calibrated probabilities; does not help if the issue is insufficient minority examples | Simple first approach; plan to recalibrate afterward (see Guide 02) |

**Practical recommendation.** Start with class weights — they are simplest and require no additional library. If performance is still poor, try SMOTE on the training set. Use undersampling only when you have abundant majority data. In all cases, recalibrate probabilities after resampling (e.g., Platt scaling or isotonic regression) because resampling changes the effective prevalence the model sees during training.

**Rule: NEVER resample the test set.** Resampling (SMOTE, undersampling, oversampling) is a training-time strategy only. The test set must reflect the real-world class distribution, otherwise metrics are meaningless. If you SMOTE the test set, you evaluate on synthetic data that does not exist in deployment. Precision, recall, and calibration all become uninterpretable. This is one of the most common and damaging mistakes in applied classification.

#### 8) Diagnostics + robustness (minimum set)

1) **Report prevalence** — always report the positive rate in the test set; without this, no metric is interpretable.
2) **Use PR curves** — PR curves are sensitive to imbalance and directly reflect precision/recall trade-offs.
3) **Threshold sweep** — show metrics across thresholds; do not report a single arbitrary threshold. The optimal threshold depends on the cost ratio of false negatives (missed readmissions) vs false positives (unnecessary interventions). See Guide 02 for cost-sensitive threshold selection.
4) **Error analysis** — inspect false positives and false negatives; are they clustered in certain subgroups or regimes?
5) **Check baselines** — compare every model to the majority-class and constant-probability baselines; report the improvement.

#### 9) Interpretation + reporting

Report: ROC-AUC + PR-AUC (and baseline values for comparison), at least one thresholded operating point (precision/recall), how the threshold was chosen, and whether any resampling was applied (training data only).

**What this does NOT mean**
- A high accuracy can be meaningless under imbalance.
- AUC does not tell you calibration.
- SMOTE does not create real data — it fills in feature space and can introduce noise.
- Resampling does not fix fundamentally insufficient data — if you have only 15 positive cases, no resampling method will reliably learn the positive class.

**Connection to Guide 02.** Once you have established baselines and chosen a resampling strategy, the next step is threshold selection. Guide 02 covers how to derive the optimal threshold from a cost matrix, calibrate probabilities, and evaluate whether your model is well-calibrated after resampling.

#### Exercises

- [ ] Compute accuracy, precision, recall for a majority-class baseline; explain why accuracy is misleading.
- [ ] Compute the Brier score of a constant-probability baseline ($\hat{p} = \pi$) and verify it equals $\pi(1-\pi)$.
- [ ] Plot ROC and PR curves for a dummy classifier and a logistic regression; explain why PR is more informative here.
- [ ] Apply SMOTE to training data, fit a model, and compare calibration before and after. Does SMOTE distort predicted probabilities?
- [ ] Choose a threshold based on a cost story and report the resulting confusion matrix.

### Project Code Map
- `src/evaluation.py`: classification metrics (ROC-AUC, PR-AUC, Brier, precision/recall)
- `scripts/train_recession.py`: training script that writes artifacts
- `scripts/predict_recession.py`: prediction script that loads artifacts
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Not establishing baselines before fitting models — you cannot tell if a model learned anything without a floor to compare against.
- Using accuracy as the primary metric with imbalanced classes — majority-class accuracy is high by construction.
- Applying SMOTE or any resampling to the test set — the test set must reflect real-world prevalence.
- Ignoring stratification in train/test splits — naive splits can leave few or zero positives in the test set.
- Picking threshold = 0.5 by default — the optimal threshold depends on costs, not convention.

<a id="summary"></a>
## Summary + Suggested Readings

You should now be able to:
- establish baseline classifiers (majority-class, prevalence-based, stratified dummy) and explain why they are necessary,
- evaluate classifiers using metrics that are robust to class imbalance (PR-AUC, Brier score, recall),
- choose among resampling strategies (SMOTE, undersampling, class weights) and know when each is appropriate,
- explain why accuracy is misleading for rare events and why you must never resample the test set.

Suggested readings:
- He & Garcia (2009), "Learning from Imbalanced Data" — the standard survey on class imbalance
- Chawla et al. (2002), "SMOTE: Synthetic Minority Over-sampling Technique" — the original SMOTE paper
- Saito & Rehmsmeier (2015), "The Precision-Recall Plot Is More Informative than the ROC Plot" — why PR-AUC matters under imbalance
- scikit-learn docs: DummyClassifier, classification metrics, calibration
- Guide 02 in this series: calibration and cost-sensitive thresholds
