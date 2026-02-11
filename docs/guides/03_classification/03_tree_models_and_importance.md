# Guide: 03_tree_models_and_importance

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/03_tree_models_and_importance.ipynb`.

This classification module predicts **next-quarter technical recession** from macro indicators using tree-based models and interprets their feature-importance outputs.

### Key Terms (defined)
- **Decision tree**: a model that recursively partitions feature space via binary splits.
- **Gini impurity**: a measure of node "mixedness"; lower means purer.
- **Entropy / information gain**: an alternative split criterion based on information theory.
- **Bagging**: training many models on bootstrap samples and averaging predictions to reduce variance.
- **Random Forest**: bagged decision trees where each split considers a random feature subset.
- **Boosting**: sequential training where each new tree corrects the previous ensemble's errors.
- **Gradient Boosted Trees (GBT)**: boosting that fits each tree to the negative gradient of the loss.
- **Feature importance (impurity-based)**: total reduction in impurity across all splits on a feature.
- **Feature importance (permutation)**: drop in performance when a single feature is shuffled.
- **SHAP values**: Shapley-based attributions that decompose each prediction into per-feature contributions.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Fit tree model
- Complete notebook section: Compare metrics
- Complete notebook section: Interpret importance
- Fit a Random Forest or Gradient Boosted classifier and compare its ROC-AUC to the logistic baseline.
- Compute both impurity-based and permutation importance and note where they disagree.
- Identify at least one correlated-feature group and explain how it affects importance rankings.

### Alternative Example (Not the Notebook Solution)
```python
# Predicting 30-day hospital readmission with a Random Forest (toy data):
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

rng = np.random.default_rng(42)
n = 500

# Simulated patient features
age = rng.normal(65, 12, n)
length_of_stay = rng.poisson(5, n)
num_comorbidities = rng.poisson(2, n)
discharge_score = rng.normal(0, 1, n)

X = np.column_stack([age, length_of_stay, num_comorbidities, discharge_score])
feature_names = ["age", "length_of_stay", "num_comorbidities", "discharge_score"]

# True readmission probability depends nonlinearly on age and comorbidities
logit = -3 + 0.03 * age + 0.4 * num_comorbidities + 0.1 * length_of_stay
prob = 1 / (1 + np.exp(-logit))
y = rng.binomial(1, prob)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=10,
    random_state=0,
)
rf.fit(X_train, y_train)

y_prob = rf.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

# Compare importance methods
imp_gini = rf.feature_importances_
perm = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=0)

for name, gi, pi in zip(feature_names, imp_gini, perm.importances_mean):
    print(f"  {name:>20s}  Gini={gi:.3f}  Perm={pi:.3f}")
```

**Interpreting the output:** `age` is a continuous variable with many possible split points, so Gini importance tends to inflate its ranking relative to the discrete `num_comorbidities`. Permutation importance, which measures the actual performance drop when a feature is shuffled, often gives a ranking more aligned with the true data-generating process. This illustrates why you should always compute both.


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

**Prerequisites:** [Classification foundations](00_recession_classifier_baselines.md) -- core classification concepts (probabilities, losses, thresholds, metrics, calibration overview, class imbalance).

### Deep Dive: Tree Models and Feature Importance

Tree-based models can capture nonlinearities and interactions that linear classifiers miss. However, interpreting "which features matter" requires understanding the mechanics beneath the importance scores.

#### 1) How a decision tree splits: Gini impurity

A decision tree greedily partitions the data at each node by choosing the feature and threshold that produce the purest child nodes. The most common criterion for classification is **Gini impurity**.

For a node with $K$ classes and class proportions $p_k$:

$$
G = 1 - \sum_{k=1}^{K} p_k^2.
$$

For binary classification ($K=2$):

$$
G = 2\,p\,(1-p),
$$

where $p$ is the proportion of the positive class. When the node is perfectly pure ($p=0$ or $p=1$), $G=0$. When it is maximally mixed ($p=0.5$), $G=0.5$.

**Worked example -- splitting on age > 65 for readmission prediction:**

Suppose a node has 200 patients: 60 readmitted ($p = 0.30$, $G = 2 \times 0.30 \times 0.70 = 0.42$).

Consider splitting on `age > 65`:
- **Left child** (age $\le$ 65): 120 patients, 24 readmitted. $p_L = 0.20$, $G_L = 2 \times 0.20 \times 0.80 = 0.32$.
- **Right child** (age $> 65$): 80 patients, 36 readmitted. $p_R = 0.45$, $G_R = 2 \times 0.45 \times 0.55 = 0.495$.

Weighted Gini after split:

$$
G_{\text{split}} = \frac{120}{200} \times 0.32 + \frac{80}{200} \times 0.495 = 0.192 + 0.198 = 0.390.
$$

Impurity reduction: $\Delta G = 0.42 - 0.39 = 0.03$. The tree evaluates every feature and every threshold and picks the split with the largest $\Delta G$.

An alternative criterion is **entropy** (information gain):

$$
H = -\sum_{k=1}^{K} p_k \log_2 p_k.
$$

In practice, Gini and entropy usually produce similar trees. Gini is slightly faster to compute and is the default in scikit-learn.

#### 2) Why trees overfit and how ensembles help

A single decision tree grown to full depth will memorize the training data (zero training error, poor generalization). Two ensemble strategies address this:

**Bagging (Bootstrap Aggregating) $\to$ Random Forest.**
- Draw $B$ bootstrap samples from the training set.
- Fit one tree per sample. At each split, consider only a random subset of $\sqrt{p}$ features (for classification) or $p/3$ (for regression).
- Average predicted probabilities across all $B$ trees.
- Averaging reduces variance without increasing bias. The random feature subset decorrelates trees, which is critical: averaging highly correlated trees gives little variance reduction.

**Boosting $\to$ Gradient Boosted Trees (GBT).**
- Start with a simple initial prediction (e.g., the base rate).
- Fit a small (shallow) tree to the negative gradient of the loss (for log loss, this is the residual on the probability scale).
- Add that tree's predictions to the ensemble, scaled by a learning rate $\eta$.
- Repeat for $B$ rounds.
- Each tree is intentionally weak (shallow), and the ensemble gradually reduces bias. Overfitting is controlled by the learning rate and early stopping.

#### 3) Random Forest vs Gradient Boosting: when to use each

| Aspect | Random Forest | Gradient Boosted Trees |
|---|---|---|
| **Bias vs variance** | Low bias, reduces variance via averaging | Reduces bias sequentially; variance controlled by $\eta$ |
| **Tuning difficulty** | Relatively forgiving; few critical hyperparameters | More sensitive; learning rate, depth, and n_estimators interact |
| **Training speed** | Embarrassingly parallel (trees are independent) | Sequential (each tree depends on the previous) |
| **Performance ceiling** | Often competitive, sometimes slightly below GBT | Often achieves the best tabular performance |
| **Overfitting risk** | Low (more trees rarely hurts) | Higher (too many rounds or too high $\eta$ overfits) |
| **Recommended when** | You want a strong, low-effort baseline | You are willing to tune carefully and want maximum performance |

In health economics applications with moderate sample sizes and many correlated features (e.g., claims data), Random Forest is often the safer starting point. Gradient Boosted Trees (via `XGBoost`, `LightGBM`, or `sklearn.ensemble.HistGradientBoostingClassifier`) are preferred when performance is paramount and you can afford a proper tuning budget.

#### 4) Hyperparameter tuning: the key parameters

**Random Forest:**
| Parameter | What it controls | Rule of thumb |
|---|---|---|
| `n_estimators` | Number of trees | 200--1000; more is rarely harmful, just slower |
| `max_depth` | Maximum tree depth | 6--15; `None` (full growth) is the default but risks overfitting on small data |
| `min_samples_leaf` | Minimum observations in a leaf | 5--50; acts as regularization |
| `max_features` | Features considered per split | `"sqrt"` (classification) or `"log2"` |

**Gradient Boosted Trees:**
| Parameter | What it controls | Rule of thumb |
|---|---|---|
| `n_estimators` | Number of boosting rounds | 100--2000; use early stopping on a validation set |
| `learning_rate` ($\eta$) | Step size per tree | 0.01--0.1; lower requires more trees but generalizes better |
| `max_depth` | Depth of each tree | 3--6; shallow trees are standard for boosting |
| `min_samples_leaf` | Minimum observations in a leaf | 10--50 |
| `subsample` | Fraction of data per tree | 0.5--0.8; adds stochasticity, reduces overfitting |

**Tuning strategy:** Start with a low learning rate (0.05) and use early stopping to find `n_estimators`. Then tune `max_depth` and `min_samples_leaf` via cross-validation (time-aware for forecasting tasks). Avoid grid-searching all parameters simultaneously -- it is computationally wasteful and rarely better than sequential tuning.

#### 5) Feature importance: impurity-based vs permutation

**Impurity-based importance** (also called "Gini importance" or "mean decrease in impurity"):
- For each feature, sum the impurity reductions ($\Delta G$) across every split that uses that feature, across all trees.
- Normalize so they sum to 1.
- **Pros:** Free (computed during training), fast.
- **Cons:**
  - Biased toward high-cardinality features (continuous variables and features with many unique values get more candidate split points).
  - Biased toward correlated features: when two features carry similar information, the tree randomly picks one at each split, so importance is **split between them**. Neither appears as important as it truly is.
  - Computed on training data, so it can reflect overfitting.

**Permutation importance:**
- Fit the model. Record baseline performance (e.g., AUC) on the test set.
- For each feature: randomly shuffle that feature's values in the test set, re-score, and record the performance drop.
- Repeat many times (e.g., 30) to get a distribution of importance.
- **Pros:**
  - Model-agnostic (works with any classifier).
  - Evaluated on held-out data (reflects generalization, not training fit).
  - Less biased by cardinality.
- **Cons:**
  - Slower (requires repeated re-scoring).
  - Still affected by correlated features, but in a different way: shuffling one feature in a correlated pair can be "compensated" by the other, so both may appear less important than the pair truly is. Grouping correlated features and shuffling them together addresses this.

**Why they disagree -- a concrete scenario:**
Consider a recession prediction model with two highly correlated features: `yield_spread_10y2y` and `yield_spread_10y3m`. Both carry similar information. In a Random Forest:
- Impurity-based importance splits the credit roughly 50/50, so each appears only moderately important.
- Permutation importance for either feature alone may be low (because the other feature compensates), but permuting both simultaneously produces a large drop.

**Takeaway:** Always compute both, compare them, and when they disagree, investigate feature correlations.

#### 6) SHAP values: the gold standard for tree model interpretation

SHAP (SHapley Additive exPlanations) values come from cooperative game theory. For each prediction, SHAP decomposes the output into additive contributions from each feature:

$$
f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j,
$$

where $\phi_0$ is the base value (average prediction) and $\phi_j$ is the contribution of feature $j$ for that specific observation.

**Why SHAP is preferred for tree models:**
- **Local explanations:** SHAP tells you why *this specific patient* was flagged as high-risk, not just which features matter on average.
- **Consistent:** if a feature contributes more to the prediction in model A than model B, its SHAP value will be larger. Impurity importance lacks this property.
- **Efficient for trees:** the `TreeSHAP` algorithm computes exact SHAP values in polynomial time for tree ensembles (as opposed to the exponential cost of exact Shapley values for arbitrary models).
- **Rich visualizations:** summary plots (feature importance as distributions), dependence plots (feature value vs SHAP value), and force plots (per-prediction explanations).

```python
# Quick SHAP usage with a fitted Random Forest:
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Global importance (mean |SHAP|)
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)
```

In health economics, SHAP values are especially valuable because stakeholders (clinicians, policymakers) need to understand *why* a model predicts high readmission risk for a specific patient, not just that it does. Regulatory contexts may require this level of interpretability.

#### 7) Health econ example: predicting 30-day hospital readmissions

Consider a Random Forest predicting 30-day readmission after heart failure hospitalization. Features include patient demographics (age, sex), clinical measures (ejection fraction, BNP level, number of comorbidities), prior utilization (ED visits in last 6 months, prior admissions), and discharge characteristics (length of stay, discharge disposition).

**Which features matter most?**

Typical findings from the literature and from SHAP analysis:
- **Number of prior admissions** and **ED visits** dominate: these are strong proxies for underlying severity and social determinants.
- **Ejection fraction** and **BNP** rank high clinically, but their permutation importance may be lower if they are correlated with each other and with age.
- **Length of stay** has a nonlinear effect: very short stays (possible premature discharge) and very long stays (indicating severity) both increase readmission risk. SHAP dependence plots reveal this U-shape, which impurity importance and permutation importance rankings alone would miss.
- **Discharge disposition** (home vs skilled nursing facility) interacts with age: being discharged home is protective for younger patients but risky for older patients with limited support. Trees capture this interaction naturally; feature importance tells you the feature matters but not *how*.

**Key lesson:** Importance rankings tell you *what* to look at. SHAP values and partial dependence plots tell you *how* features drive predictions. Always follow importance with deeper investigation.

#### 8) Diagnostics and robustness

1) **Out-of-sample importance.** Always compute permutation importance on held-out data, not training data. Training-set permutation importance reflects overfitting.

2) **Stability across time periods.** Compute importance separately on pre-2008, 2008--2015, and post-2015 data. If the ranking changes dramatically, the model is sensitive to regime and importance claims are fragile.

3) **Correlation groups.** Before interpreting individual feature importance, compute a correlation matrix. Group features with $|\rho| > 0.7$ and interpret the group rather than any single member.

4) **Calibration check.** Tree ensembles (especially Random Forests) tend to produce probabilities clustered away from 0 and 1. Check calibration plots and consider Platt scaling or isotonic regression if probabilities need to be well-calibrated (e.g., for clinical risk scores).

5) **OOB (Out-of-Bag) error.** Random Forests provide a free estimate of generalization error: each tree was not trained on roughly 37% of the data, so those "out-of-bag" observations can be used for evaluation without a separate validation set. This is useful for quick diagnostics but does not replace proper time-aware evaluation for forecasting.

#### Exercises

- [ ] Fit a Random Forest and a Gradient Boosted classifier to the same data. Compare ROC-AUC and Brier score. Which model performs better, and why might that be?
- [ ] Compute impurity-based and permutation importance for the same model. Where do the rankings disagree? Identify a correlated feature pair that explains the disagreement.
- [ ] Evaluate importance stability across two non-overlapping time periods. Which features remain consistently important?
- [ ] Produce a SHAP summary plot for the top 8 features. Identify one feature with a nonlinear effect (visible as a non-monotone color pattern).
- [ ] Tune `max_depth` and `min_samples_leaf` for a Random Forest using time-aware cross-validation. Report the best configuration and the improvement over the default.

### Key Terms (Tree Models)

| Term | Definition |
|---|---|
| **Gini impurity** | $G = 1 - \sum p_k^2$; measures node impurity. Lower is purer. |
| **Entropy** | $H = -\sum p_k \log_2 p_k$; alternative split criterion. |
| **Bagging** | Bootstrap aggregating: train on bootstrap samples, average predictions. |
| **Boosting** | Sequential training: each model corrects the ensemble's residuals. |
| **Random Forest** | Bagged trees with random feature subsets at each split. |
| **Gradient Boosted Trees** | Boosting where each tree fits the negative gradient of the loss. |
| **Impurity importance** | Total impurity reduction across all splits on a feature; biased by cardinality. |
| **Permutation importance** | Performance drop when a feature is shuffled on test data; model-agnostic. |
| **SHAP values** | Shapley-based per-feature, per-prediction attributions; gold standard for tree interpretation. |
| **Out-of-bag (OOB) error** | Free generalization estimate from bootstrap samples not used to train each tree. |
| **Concept drift** | Change in the data-generating process over time, degrading model performance. |

### Common Mistakes (Tree Models)

- **Treating impurity importance as ground truth.** It is biased toward high-cardinality and correlated features. Always cross-check with permutation importance.
- **Interpreting importance as causation.** A feature being "important" for prediction does not mean changing it would change the outcome. This is a prediction model, not a causal model.
- **Growing fully deep trees without regularization.** Unconstrained trees memorize noise. Set `max_depth` and `min_samples_leaf` explicitly.
- **Ignoring calibration.** Random Forests and GBTs can rank risk well (high AUC) while producing poorly calibrated probabilities. If the probability value itself matters (as it often does in health econ applications), calibrate post-hoc.
- **Using training-set permutation importance.** This reflects what the model memorized, not what generalizes. Always permute on held-out data.
- **Over-tuning on a single validation fold.** Use walk-forward or repeated time splits for hyperparameter tuning.

### Project Code Map
- `src/evaluation.py`: classification metrics (ROC-AUC, PR-AUC, Brier, precision/recall) and splits (`time_train_test_split_index`, `walk_forward_splits`)
- `scripts/train_recession.py`: training script that writes artifacts
- `scripts/predict_recession.py`: prediction script that loads artifacts
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)

<a id="summary"></a>
## Summary + Suggested Readings

You should now be able to:
- Explain how a decision tree selects splits using Gini impurity (or entropy).
- Describe why single trees overfit and how bagging (Random Forest) and boosting (GBT) address this.
- Tune the most critical hyperparameters for both Random Forest and GBT.
- Compute and compare impurity-based and permutation importance, explaining why they can disagree.
- Use SHAP values to produce local (per-prediction) explanations for tree ensemble models.
- Apply these tools to health economics contexts like hospital readmission prediction.

Suggested readings:
- Breiman (2001): "Random Forests" -- the foundational paper.
- Friedman (2001): "Greedy Function Approximation: A Gradient Boosting Machine."
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions" (SHAP).
- scikit-learn docs: `RandomForestClassifier`, `GradientBoostingClassifier`, `permutation_importance`.
- Hastie, Tibshirani, & Friedman: *Elements of Statistical Learning*, Ch. 10 (Boosting) and Ch. 15 (Random Forests).
- Christodoulou et al. (2019): "A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models" -- a sobering health econ perspective.
