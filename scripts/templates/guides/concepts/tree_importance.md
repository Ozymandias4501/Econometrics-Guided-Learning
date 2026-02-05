### Deep Dive: Tree models and feature importance (interpretation pitfalls)

Tree-based models can capture nonlinearities and interactions, but interpretation requires care.

#### 1) Intuition (plain English)

Trees can outperform linear models in prediction, especially when relationships are nonlinear.
But tree “feature importance” is often misunderstood.

**Story example:** A random forest says a variable is “important.”
That does not mean changing the variable causes the outcome; it means the variable helps prediction in the fitted model.

#### 2) Notation + setup (define terms)

Tree models partition feature space into regions and predict with averages (regression) or probabilities (classification).

Feature importance measures (common types):
- **impurity-based importance:** how much splits reduce impurity across the forest,
- **permutation importance:** how much performance drops when a feature is shuffled.

#### 3) Assumptions

Interpretation assumes:
- evaluation is leakage-free,
- features are aligned correctly in time,
- importance is stable across folds/periods.

Correlated features complicate importance:
- the model can “spread” importance across correlated predictors.

#### 4) Estimation mechanics (high level)

Impurity-based importance is fast but can be biased toward:
- variables with many possible split points,
- noisy continuous features.

Permutation importance is often more reliable:
- measure baseline performance,
- shuffle one feature in the test set,
- measure performance drop.

#### 5) Inference: treat importance as descriptive

Importance does not come with simple p-values.
Uncertainty can be assessed via:
- cross-validation variability,
- bootstrap resampling,
- permutation distributions.

#### 6) Diagnostics + robustness (minimum set)

1) **Out-of-sample importance**
- compute importance on test/validation data, not training.

2) **Stability across folds/time**
- if importance changes drastically across periods, interpretation is fragile.

3) **Correlation groups**
- check whether important variables are part of a correlated cluster; interpret the group, not a single variable.

#### 7) Interpretation + reporting

Report:
- model type and evaluation scheme,
- the importance method (impurity vs permutation),
- stability checks.

**What this does NOT mean**
- importance is not a causal effect,
- importance is not the same as “economic significance.”

#### Exercises

- [ ] Compute impurity and permutation importance for the same model; compare and explain differences.
- [ ] Evaluate importance stability across two time periods.
- [ ] Identify a correlated feature group and explain why “the most important feature” can be unstable.
