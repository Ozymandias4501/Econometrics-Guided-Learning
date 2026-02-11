# Guide: 01_logistic_recession_classifier

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/03_classification/01_logistic_recession_classifier.ipynb`.

This guide focuses on **logistic regression as a probability model** -- the workhorse classifier in econometrics and health economics. Where guide 00 covers the shared classification toolkit (metrics, thresholds, calibration overview), this guide goes deep on the logistic model itself: odds, log-odds, coefficient interpretation, maximum likelihood estimation, regularization, and common pitfalls.

### Key Terms (defined)
- **Odds**: ratio $p/(1-p)$; odds of 3 mean the event is 3 times as likely as not.
- **Log-odds (logit)**: $\log(p/(1-p))$; the scale on which logistic regression is linear.
- **Sigmoid function**: $\sigma(z) = 1/(1+e^{-z})$; maps any real number to (0, 1).
- **Odds ratio**: $e^{\beta_j}$; the multiplicative change in odds for a 1-unit increase in $x_j$.
- **Marginal effect**: the change in predicted probability for a small change in a feature.
- **Maximum likelihood estimation (MLE)**: choosing parameters that make the observed data most probable.
- **Regularization**: penalizing large coefficients to prevent overfitting (L1 = Lasso, L2 = Ridge).
- **Separation**: when a feature perfectly predicts the outcome, causing MLE to diverge.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Train/test split
- Complete notebook section: Fit logistic
- Complete notebook section: ROC/PR
- Complete notebook section: Threshold tuning
- Establish baselines before fitting any model.
- Fit at least one probabilistic classifier and evaluate ROC-AUC, PR-AUC, and Brier score.
- Pick a threshold intentionally (cost-based or metric-based) and justify it.

### Alternative Example: Logistic Regression for Hospital Readmission
```python
# Predict 30-day hospital readmission from patient features.
# This is NOT the notebook solution -- it illustrates logistic mechanics.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(42)
n = 500

# Simulate patient features
age = rng.normal(65, 12, n)            # age in years
comorbidity_index = rng.poisson(2, n)  # Charlson-like index (0-10)
length_of_stay = rng.exponential(5, n) # days

X = np.column_stack([age, comorbidity_index, length_of_stay])

# True log-odds: intercept -3, plus contributions from each feature
logit_p = -3.0 + 0.02 * age + 0.4 * comorbidity_index + 0.05 * length_of_stay
p_true = 1 / (1 + np.exp(-logit_p))
y = rng.binomial(1, p_true)

print(f"Readmission rate: {y.mean():.1%}")  # roughly 20-25%

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty='l2', C=1.0, max_iter=5000))
])
clf.fit(X, y)

# Inspect coefficients (on standardized scale)
coefs = clf.named_steps['lr'].coef_[0]
feature_names = ['age', 'comorbidity_index', 'length_of_stay']
for name, b in zip(feature_names, coefs):
    print(f"  {name}: coef = {b:.3f}, odds ratio = {np.exp(b):.3f}")
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

> **Prerequisites:** [Classification foundations](00_recession_classifier_baselines.md) -- core classification concepts (probabilities, losses, thresholds, metrics, calibration overview, class imbalance).

### Deep Dive: Logistic Regression as a Probability Model

Logistic regression is the baseline probabilistic classifier in both econometrics and health economics. It is interpretable, fast, and -- when properly regularized -- surprisingly competitive.

#### 1) Odds and log-odds intuition

**Probability, odds, and log-odds are three ways to express the same quantity.** Understanding all three is essential because logistic regression is linear on the log-odds scale, not on the probability scale.

| Probability $p$ | Odds $p/(1-p)$ | Log-odds $\log(p/(1-p))$ |
|:---:|:---:|:---:|
| 0.10 | 0.111 | -2.20 |
| 0.25 | 0.333 | -1.10 |
| 0.50 | 1.000 | 0.00 |
| 0.75 | 3.000 | 1.10 |
| 0.90 | 9.000 | 2.20 |

**Worked numerical example.** Suppose the baseline 30-day readmission probability is $p = 0.20$. Then:

$$
\text{odds} = \frac{0.20}{1 - 0.20} = \frac{0.20}{0.80} = 0.25
$$

$$
\text{log-odds} = \log(0.25) \approx -1.386.
$$

If a patient has one additional comorbidity and the coefficient is $\beta = 0.4$, the new log-odds are $-1.386 + 0.4 = -0.986$, corresponding to:

$$
p_{\text{new}} = \frac{1}{1 + e^{0.986}} \approx 0.272.
$$

The probability rose from 20.0% to 27.2%. Notice the same $\beta = 0.4$ would produce a different absolute probability change at a different baseline (e.g., from 50% to 60%). This non-linearity is a key feature of the logistic model.

#### 2) Notation and the logistic model

Let:
- $y_i \in \{0,1\}$ be the label (1 = event of interest),
- $x_i \in \mathbb{R}^k$ be the feature vector,
- $p_i = \Pr(y_i=1 \mid x_i)$ be the model probability.

Logistic regression assumes:

$$
\log\left(\frac{p_i}{1-p_i}\right) = x_i'\beta = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik}.
$$

Inverting:

$$
p_i = \sigma(x_i'\beta) = \frac{1}{1 + e^{-x_i'\beta}}.
$$

Key assumptions:
- log-odds is linear in features (no quadratic terms unless you add them),
- observations are conditionally independent given $x$ (often violated in time series -- see guide 00),
- features are not perfectly collinear.

#### 3) Coefficient interpretation as odds ratios

This is the single most important interpretation skill for health economics applications.

**Rule:** A 1-unit increase in $x_j$ multiplies the odds of the outcome by $e^{\beta_j}$, holding all other features fixed.

**Health econ example -- hospital readmission:**

Suppose you fit a logistic model of 30-day readmission and find $\beta_{\text{comorbidity}} = 0.40$. Then:

$$
e^{0.40} \approx 1.49
$$

*Interpretation:* Each additional point on the comorbidity index multiplies the odds of readmission by 1.49 (a 49% increase in odds), holding age and length of stay constant.

For a binary feature like "discharged on a weekend" with $\beta = 0.30$:

$$
e^{0.30} \approx 1.35
$$

*Interpretation:* Weekend discharge is associated with 35% higher odds of readmission compared to weekday discharge.

**Confidence intervals for odds ratios.** In `statsmodels`, exponentiate the coefficient confidence interval:

$$
\text{95\% CI for OR} = \left(e^{\beta_j - 1.96 \cdot \text{SE}(\beta_j)},\; e^{\beta_j + 1.96 \cdot \text{SE}(\beta_j)}\right).
$$

If the interval includes 1, the association is not statistically significant at the 5% level.

#### 4) Marginal effects: at the mean vs average marginal effects

Because the logistic function is non-linear, the effect of $x_j$ on the probability depends on where you are on the curve.

**Marginal effect at the mean (MEM):**
Evaluate the derivative at $x = \bar{x}$ (the sample mean):

$$
\text{MEM}_j = \beta_j \cdot p(\bar{x}) \cdot (1 - p(\bar{x}))
$$

where $p(\bar{x}) = \sigma(\bar{x}'\beta)$. This gives the probability change for one unit of $x_j$ at the "average" patient.

**Average marginal effect (AME):**
Compute the marginal effect for every observation, then average:

$$
\text{AME}_j = \frac{1}{n}\sum_{i=1}^{n} \beta_j \cdot p_i \cdot (1 - p_i).
$$

**When to use which:**
- **AME** is generally preferred in health economics because it reflects the population distribution, not a hypothetical "average" individual who may not exist.
- **MEM** is simpler and is common in older econometrics textbooks.
- In `statsmodels`, use `model.get_margeff(at='overall')` for AME and `model.get_margeff(at='mean')` for MEM.

**Example:** If $\beta_j = 0.4$ and the average predicted probability is $\bar{p} = 0.20$:

$$
\text{AME}_j \approx 0.4 \times 0.20 \times 0.80 = 0.064
$$

So an additional comorbidity is associated with roughly a 6.4 percentage point increase in readmission probability, on average across patients.

#### 5) Maximum likelihood estimation: why likelihood, not least squares?

**Intuition.** In linear regression we minimize squared error because the errors are (assumed) Gaussian, and squared error corresponds to the Gaussian log-likelihood. For binary outcomes, the "error" is not Gaussian -- the outcome is 0 or 1. The natural loss is the **Bernoulli log-likelihood**:

$$
\ell(\beta) = \sum_{i=1}^{n} \left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right].
$$

Maximizing this is equivalent to minimizing **log loss** (cross-entropy):

$$
\mathcal{L}(\beta) = -\frac{1}{n}\sum_{i=1}^{n} \left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right].
$$

**Why not least squares?** You could minimize $\sum (y_i - p_i)^2$, but:
1. The objective is non-convex in $\beta$ when $p_i = \sigma(x_i'\beta)$, so gradient descent may find local minima.
2. The log-likelihood is globally concave, guaranteeing a unique optimum.
3. MLE gives asymptotically efficient estimates (lowest variance among consistent estimators).

In practice, `sklearn` uses iterative solvers (L-BFGS, Newton-CG, or coordinate descent for L1) to find the MLE.

#### 6) Regularization in logistic regression

**Why regularize?** With many features or correlated features, unregularized MLE can produce unstable, large coefficients -- especially in small samples common in health econ.

**L2 regularization (Ridge, default in sklearn):**

$$
\min_\beta \; \mathcal{L}(\beta) + \frac{1}{2C}\|\beta\|_2^2
$$

- Shrinks all coefficients toward zero.
- Does not set any to exactly zero.
- `C` in sklearn is the inverse regularization strength: smaller `C` = stronger penalty.
- Default `C=1.0` is rarely optimal; tune it with cross-validation.

**L1 regularization (Lasso):**

$$
\min_\beta \; \mathcal{L}(\beta) + \frac{1}{C}\|\beta\|_1
$$

- Can set coefficients to exactly zero, performing feature selection.
- Useful when you suspect many features are irrelevant.
- Use `LogisticRegression(penalty='l1', solver='saga')` in sklearn.

**Elastic net** (`penalty='elasticnet'`, `l1_ratio` in [0,1]) combines both penalties. This is useful when you have groups of correlated features and want some sparsity.

**Practical advice:**
- Always scale features before regularization (use `StandardScaler` in a pipeline).
- Tune `C` using time-aware cross-validation, not random CV, for forecasting tasks.
- Report which penalty you used and the chosen `C` value.

#### 7) Separation and quasi-separation

**What is separation?** If a feature (or combination of features) perfectly predicts the outcome -- all patients with comorbidity index > 8 are readmitted, and none with index <= 8 are -- then the MLE does not exist. The optimizer will push $\beta \to \pm\infty$ trying to achieve a perfect step function.

**Quasi-separation** is a milder form: the data is almost but not perfectly separable. The MLE may technically exist but the estimates are extremely large with enormous standard errors.

**Symptoms:**
- Coefficients that are suspiciously large (e.g., $|\beta| > 10$).
- Very large standard errors or confidence intervals spanning several orders of magnitude.
- Warnings from `statsmodels` about "perfect separation detected" or failure to converge.

**Solutions:**
- **Regularization** is the most practical fix. L2 (Ridge) keeps all coefficients finite. This is why sklearn uses `penalty='l2'` by default.
- **Firth's penalized likelihood** (available in the `firthlogist` package) adds a small penalty that eliminates the separation problem while introducing minimal bias.
- **Inspect your data.** Sometimes separation is a data problem (e.g., a leaked feature or a variable that encodes the outcome). Always check before applying a statistical fix.

**Health econ example:** In a small-sample study of ICU mortality, if all patients with a particular rare complication died, the coefficient for that complication will blow up. Regularization or Firth's method will give you a finite, interpretable estimate.

#### 8) Alternative example: logistic regression with odds-ratio interpretation

```python
# Logistic regression for 30-day readmission with odds-ratio reporting.
# NOT the notebook solution -- illustrates coefficient interpretation.
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(99)
n = 400

# Simulate patient data
age = rng.normal(70, 10, n)
female = rng.binomial(1, 0.55, n)
comorbidities = rng.poisson(3, n)

X = np.column_stack([age, female, comorbidities])
X_with_const = sm.add_constant(X)

logit_p = -4.0 + 0.03 * age - 0.2 * female + 0.35 * comorbidities
p_true = 1 / (1 + np.exp(-logit_p))
y = rng.binomial(1, p_true)

# Fit with statsmodels for inference
model = sm.Logit(y, X_with_const)
result = model.fit(disp=0)
print(result.summary2())

# Odds ratios with 95% CI
params = result.params[1:]      # skip intercept
conf = result.conf_int().iloc[1:]
names = ['age', 'female', 'comorbidities']
for i, name in enumerate(names):
    or_val = np.exp(params.iloc[i])
    or_lo  = np.exp(conf.iloc[i, 0])
    or_hi  = np.exp(conf.iloc[i, 1])
    print(f"  {name}: OR = {or_val:.2f} (95% CI: {or_lo:.2f} - {or_hi:.2f})")

# Average marginal effects
mfx = result.get_margeff(at='overall')
print(mfx.summary())
```

#### 9) Diagnostics and robustness

1. **Calibration** -- check whether predicted probabilities match observed frequencies. Use the calibration tools from guide 02.
2. **Feature scaling** -- standardize features when comparing coefficient magnitudes; otherwise the scale of each variable drives the size of $\beta$.
3. **Multicollinearity** -- compute variance inflation factors (VIF). High VIF inflates standard errors and makes individual coefficients unreliable, even if predictions are fine.
4. **Specification** -- consider adding interaction terms or polynomial features if the log-odds linearity assumption is suspect. A Hosmer-Lemeshow test can flag gross misspecification.

### Key Terms (Logistic-Specific)

| Term | Definition |
|------|-----------|
| Odds ratio | $e^{\beta_j}$; multiplicative change in odds per 1-unit increase in $x_j$ |
| Marginal effect at the mean (MEM) | Probability change evaluated at $x = \bar{x}$ |
| Average marginal effect (AME) | Mean of individual marginal effects across all observations |
| Log loss (cross-entropy) | The negative log-likelihood; the objective logistic regression minimizes |
| L1 / Lasso | Regularization that can zero out coefficients (feature selection) |
| L2 / Ridge | Regularization that shrinks coefficients (default in sklearn) |
| Separation | Perfect prediction by a feature; causes MLE to diverge |
| Firth's penalty | Bias-reduction method that handles separation in small samples |

### Common Mistakes (Logistic-Specific)
- **Interpreting coefficients as probability changes.** Coefficients are log-odds changes. Convert to odds ratios ($e^{\beta}$) or compute marginal effects for probability-scale interpretation.
- **Forgetting to scale features before regularization.** Unscaled features make regularization penalize large-scale variables disproportionately.
- **Ignoring separation warnings.** If `statsmodels` warns about perfect separation, do not trust the point estimates. Apply regularization or Firth's method.
- **Using default `C=1.0` without tuning.** The default regularization strength is arbitrary. Always tune `C` with cross-validation.
- **Claiming causal effects from logistic coefficients.** Without proper identification (randomization, IV, RDD, etc.), logistic coefficients reflect associations, not causal effects.
- **Reporting only p-values without odds ratios or marginal effects.** In health economics, the magnitude of the association (and its confidence interval) matters more than statistical significance alone.

### Exercises

- [ ] Convert the coefficient $\beta = 0.50$ to an odds ratio and write a one-sentence interpretation in a health context.
- [ ] Compute the marginal effect of that coefficient at $p = 0.10$ and at $p = 0.50$. Explain why they differ.
- [ ] Fit a logistic regression with L2 regularization. Vary `C` over [0.01, 0.1, 1, 10, 100] and plot test-set log loss. Which `C` wins?
- [ ] Use `statsmodels` to produce odds ratios with 95% confidence intervals. Identify which features are statistically significant.
- [ ] Simulate a dataset with perfect separation (one feature perfectly predicts the outcome). Fit logistic regression with and without regularization. Compare the coefficients.
- [ ] Compute both the MEM and AME for a continuous feature. When do they disagree substantially?

<a id="summary"></a>
## Summary + Suggested Readings

You should now be able to:
- explain why logistic regression models log-odds (not probabilities) as linear,
- convert coefficients to odds ratios and interpret them in health econ language,
- distinguish marginal effects at the mean from average marginal effects,
- articulate why MLE is the right estimator for binary outcomes,
- choose between L1 and L2 regularization and tune the penalty strength,
- diagnose separation and know how to handle it.

Suggested readings:
- Wooldridge, *Introductory Econometrics* -- Ch. 17 (logit/probit models, marginal effects)
- Hosmer, Lemeshow & Sturdivant, *Applied Logistic Regression* -- the standard health-sciences reference
- scikit-learn docs: `LogisticRegression`, regularization parameters, solvers
- Norton, Wang & Ai (2004), "Computing interaction effects and standard errors in logit and probit models" -- *Economics Letters*
- King & Zeng (2001), "Logistic regression in rare events data" -- *Political Analysis* (relevant when recession quarters are rare)
