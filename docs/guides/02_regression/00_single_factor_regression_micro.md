# Guide: 00_single_factor_regression_micro

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/02_regression/00_single_factor_regression_micro.ipynb`.

This regression module covers both prediction and inference, with a strong emphasis on interpretation.

### Key Terms (defined)
- **OLS (Ordinary Least Squares)**: chooses coefficients that minimize squared prediction errors.
- **Coefficient**: expected change in the target per unit change in a feature (holding others fixed).
- **Standard error (SE)**: uncertainty estimate for a coefficient.
- **p-value**: probability of observing an effect at least as extreme if the true effect were zero (under assumptions).
- **Confidence interval (CI)**: a range of plausible coefficient values under assumptions.
- **Heteroskedasticity**: non-constant error variance; common in cross-section.
- **Autocorrelation**: errors correlated over time; common in time series.
- **HAC/Newey-West**: robust SE for time-series autocorrelation/heteroskedasticity.


### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Load census data
- Complete notebook section: Build log variables
- Complete notebook section: Fit OLS + HC3
- Complete notebook section: Interpretation
- Fit at least one plain OLS model and one robust-SE variant (HC3 or HAC).
- Interpret coefficients in units (or standardized units) and explain what they do *not* mean.
- Run at least one diagnostic: residual plot, VIF table, or rolling coefficient stability plot.

### Alternative Example (Not the Notebook Solution)
```python
# Toy OLS with robust SE (not the notebook data):
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)
x = rng.normal(size=200)
y = 2.0 + 0.5*x + rng.normal(scale=1 + 0.5*np.abs(x), size=200)  # heteroskedastic errors
X = sm.add_constant(pd.DataFrame({'x': x}))
res = sm.OLS(y, X).fit()
res_hc3 = res.get_robustcov_results(cov_type='HC3')
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Core Regression: mechanics, assumptions, and interpretation (OLS as the baseline)

Linear regression is the baseline model for both econometrics and ML. Even when you use nonlinear models, the regression mindset (assumptions → estimation → inference → diagnostics) remains essential.

#### 1) Intuition (plain English)

Regression answers questions like:
- “How does $Y$ vary with $X$ on average?”
- “Holding other observed controls fixed, what is the association between one feature and the outcome?”

In economics we care about two different uses:
- **prediction:** does a model forecast well out-of-sample?
- **inference:** what is the estimated relationship and its uncertainty?

#### 2) Notation + setup (define symbols)

Scalar form (observation $i=1,\dots,n$):

$$
y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_K x_{iK} + \varepsilon_i.
$$

Matrix form:

$$
\mathbf{y} = \mathbf{X}\beta + \varepsilon.
$$

**What each term means**
- $\mathbf{y}$: $n\times 1$ vector of outcomes.
- $\mathbf{X}$: $n\times (K+1)$ design matrix (includes an intercept column).
- $\beta$: $(K+1)\times 1$ vector of coefficients.
- $\varepsilon$: $n\times 1$ vector of errors (unobserved determinants).

#### 3) Assumptions (what you need for unbiasedness and for inference)

For interpretation and inference, it helps to separate:

**(A) Assumptions for unbiased coefficients**

1) **Linearity in parameters**
- $y$ is linear in $\beta$ (you can still include nonlinear transformations of $x$).

2) **No perfect multicollinearity**
- columns of $X$ are not perfectly linearly dependent.

3) **Exogeneity (key!)**
$$
\mathbb{E}[\varepsilon \mid X] = 0.
$$

This rules out:
- omitted variable bias,
- reverse causality,
- many forms of measurement error problems.

**Why this matters (concrete example):** Suppose you regress wages on years of education. The error $\varepsilon_i$ absorbs everything else that affects wages — including unobserved ability. If ability is positively correlated with education (high-ability people tend to get more schooling), then $\varepsilon_i$ is correlated with $x_i$ (education), violating $\mathbb{E}[\varepsilon \mid X] = 0$. The result: OLS attributes some of ability's effect to education, inflating $\hat\beta$. This is omitted variable bias in action — the coefficient captures a mix of the education effect *and* the ability effect. Robust standard errors do not fix this; only a better identification strategy (e.g., an instrument or a natural experiment) can.

**(B) Assumptions for classical standard errors**

4) **Homoskedasticity**
$$
\mathrm{Var}(\varepsilon \mid X) = \sigma^2 I.
$$

5) **No autocorrelation (time series)**
$$
\mathrm{Cov}(\varepsilon_t, \varepsilon_{t-k}) = 0 \text{ for } k \neq 0.
$$

When (4)–(5) fail, OLS coefficients can remain valid under (A), but naive SE are wrong → robust/HAC/clustered SE.

#### 4) Estimation mechanics: deriving OLS

OLS chooses coefficients to minimize the sum of squared residuals:

$$
\hat\beta = \arg\min_{\beta} \sum_{i=1}^{n} (y_i - x_i'\beta)^2
= \arg\min_{\beta} (\mathbf{y} - \mathbf{X}\beta)'(\mathbf{y} - \mathbf{X}\beta).
$$

Take derivatives (the “normal equations”):

$$
\frac{\partial}{\partial \beta} (\mathbf{y}-\mathbf{X}\beta)'(\mathbf{y}-\mathbf{X}\beta)
= -2\mathbf{X}'(\mathbf{y}-\mathbf{X}\beta) = 0.
$$

Solve:
$$
\mathbf{X}'\mathbf{X}\hat\beta = \mathbf{X}'\mathbf{y}
\quad \Rightarrow \quad
\hat\beta = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}.
$$

**What each term means**
- $(X'X)^{-1}$ exists only if there is no perfect multicollinearity.
- OLS is a projection of $y$ onto the column space of $X$.

#### 5) Coefficient interpretation (and why “holding fixed” is tricky)

In the model, $\beta_j$ means:

> the expected change in $y$ when $x_j$ increases by one unit, holding other regressors fixed (within the model).

In economics, “holding fixed” can be unrealistic if regressors move together (multicollinearity).
That is why:
- coefficient signs can flip,
- SE can inflate,
- interpretation must be cautious.

#### 6) Inference: standard errors, t-stats, confidence intervals

Under classical assumptions:

$$
\mathrm{Var}(\hat\beta \mid X) = \sigma^2 (X'X)^{-1}.
$$

In practice we estimate $\sigma^2$ and compute standard errors:
- $\widehat{SE}(\hat\beta_j)$
- t-stat: $t_j = \hat\beta_j / \widehat{SE}(\hat\beta_j)$
- 95% CI: $\hat\beta_j \pm 1.96\,\widehat{SE}(\hat\beta_j)$ (approx.)

When assumptions fail, use robust SE:
- **HC3** for cross-section heteroskedasticity,
- **HAC/Newey–West** for time-series autocorrelation + heteroskedasticity,
- **clustered SE** for grouped dependence (panels/DiD).

#### 7) Diagnostics + robustness (minimum set)

1) **Residual checks**
- plot residuals vs fitted values; look for heteroskedasticity/nonlinearity.

2) **Multicollinearity**
- compute VIF; large VIF → unstable coefficients.

3) **Time-series dependence**
- check residual autocorrelation; use HAC when needed.

4) **Stability**
- rolling regressions or sub-sample splits; do coefficients drift?

#### 8) Interpretation + reporting

Always report:
- coefficient in units (or standardized units),
- robust SE appropriate to data structure,
- a short causal warning unless you have a causal design.

**What this does NOT mean**
- Regression does not “control away” all confounding automatically.
- A small p-value does not imply economic importance.
- A high $R^2$ does not imply good forecasting out-of-sample.

#### Exercises

- [ ] Derive the normal equations and explain each step in words.
- [ ] Fit OLS and HC3 (or HAC) and compare SE; explain why they differ.
- [ ] Create two correlated regressors and show how multicollinearity affects coefficient stability.
- [ ] Write a 6-sentence interpretation of one regression output, including what you can and cannot claim.

### Deep Dive: Log–log regression (elasticity-style interpretation)

Log transforms are common in micro data because many variables (income, rent, population) are heavy-tailed and relationships are often multiplicative.

#### 1) Intuition (plain English)

In a log–log model, coefficients are approximately **elasticities**:
- “A 1% increase in $x$ is associated with a $\beta$% change in $y$.”

This is often a more interpretable economic statement than “one dollar changes rent by …”

#### 2) Notation + setup (define symbols)

Log–log regression:

$$
\log(y_i) = \alpha + \beta \log(x_i) + \varepsilon_i.
$$

**What each term means**
- $\log(\cdot)$ compresses scale and turns multiplicative relationships into additive ones.
- $\beta$ is the elasticity-like coefficient.

Why elasticity? For small changes:

$$
\Delta \log(x) \approx \frac{\Delta x}{x}
\quad \Rightarrow \quad
\Delta \log(y) \approx \beta \Delta \log(x).
$$

So a 1% change in $x$ corresponds to about a $\beta$% change in $y$.

#### 3) Assumptions (and practical caveats)

Log transforms require:
- $x_i > 0$ and $y_i > 0$.

Common issues:
- zeros (log undefined),
- negative values,
- heavy measurement error at small values.

Workarounds (must be justified):
- filter to positive values,
- use `log1p` (changes interpretation),
- use alternative functional forms.

#### 4) Estimation mechanics

Once transformed, you fit OLS on $\log(y)$ and $\log(x)$ as usual.
Interpretation should be in percent changes, not raw units.

#### 5) Inference

If heteroskedasticity is present (common in micro), use robust SE (HC3).

#### 6) Diagnostics + robustness (minimum set)

1) **Check positivity**
- count how many observations would be dropped by logging.

2) **Residual diagnostics**
- plot residuals vs fitted; heteroskedasticity is common.

3) **Functional form sensitivity**
- compare log–log to level-level or log-level if meaningful.

#### 7) Interpretation + reporting

Report:
- how you handled zeros,
- whether coefficients are interpreted as elasticities,
- robust SE choice.

#### Exercises

- [ ] Fit a log–log regression and interpret $\beta$ as an elasticity in words.
- [ ] Compare to a level-level regression; explain how interpretation changes.
- [ ] Demonstrate the “small change” approximation numerically for one observation.

### Deep Dive: Robust Standard Errors (HC3) for Cross-Sectional Data

Robust standard errors are about **honest uncertainty** when error variance differs across observations.

#### 1) Intuition (plain English)

In cross-sectional micro data, the variance of “unexplained” outcomes often differs systematically:
- income has higher variance at higher education levels,
- spending variability rises with income,
- measurement error differs across regions.

If you assume constant error variance when it is not true, your coefficient estimate may be fine, but your **standard errors** can be wrong—often too small.

**Why classical SE fail under heteroskedasticity:** Classical SE compute one pooled estimate of $\sigma^2$ and apply it to every observation. If residual variance is actually larger for some observations (e.g., high-income counties have more variable rents), the pooled estimate averages over the quiet and noisy parts of the data. For coefficients driven mainly by the noisy subgroup, the averaged $\sigma^2$ understates true uncertainty, producing SE that are too small and p-values that reject too often. You get "statistical significance" that does not hold up in reality.

**Why HC3 specifically:** HC0 (White's original) replaces the assumed $\sigma^2$ with each observation's own squared residual $\hat{u}_i^2$. But OLS residuals systematically underestimate the true errors at high-leverage points (observations that have outsized influence on the fitted line). HC3 corrects for this by dividing each $\hat{u}_i^2$ by $(1 - h_{ii})^2$, where $h_{ii}$ is the leverage. This inflates residuals at influential points, preventing them from hiding their own impact. In small-to-moderate samples, HC3 performs noticeably better than HC0 or HC1.

#### 2) Notation + setup (define symbols)

Regression model:

$$
\mathbf{y} = \mathbf{X}\beta + \mathbf{u}.
$$

**What each term means**
- $\mathbf{y}$: $n\times 1$ outcome vector.
- $\mathbf{X}$: $n\times p$ design matrix (includes intercept).
- $\beta$: $p\times 1$ coefficient vector.
- $\mathbf{u}$: $n\times 1$ error vector.

Classical OLS inference assumes homoskedasticity:
$$
\mathrm{Var}(\mathbf{u} \mid \mathbf{X}) = \sigma^2 I_n.
$$

Heteroskedasticity means:
$$
\mathrm{Var}(\mathbf{u} \mid \mathbf{X}) = \Omega,
\quad \text{where } \Omega \text{ is not } \sigma^2 I_n.
$$

Often $\Omega$ is diagonal with unequal variances (but we don’t need to specify the exact pattern to build robust SE).

#### 3) Estimation mechanics: what changes and what does not

OLS coefficients:
$$
\hat\beta = (X'X)^{-1}X'y.
$$

**Key fact**
- $\hat\beta$ is the same whether you use classical or robust SE.
- What changes is the estimated variance of $\hat\beta$ (and therefore t-stats, p-values, and CI).

#### 4) The robust “sandwich” covariance estimator

A general heteroskedasticity-robust variance estimator has the form:

$$
\widehat{\mathrm{Var}}(\hat\beta)
= (X'X)^{-1} \left(X'\hat\Omega X\right) (X'X)^{-1}.
$$

**What each term means**
- “Bread”: $(X'X)^{-1}$ is the usual OLS matrix.
- “Meat”: $X'\hat\Omega X$ estimates the error variance structure.

Different HC estimators choose different $\hat\Omega$.

#### 5) Why HC3 specifically? (leverage adjustment)

HC3 is designed to be more conservative in finite samples when some points have high leverage.

Define:
- residuals $\hat u_i$,
- leverage $h_{ii}$ (diagonal of the hat matrix $H = X(X'X)^{-1}X'$):
$$
h_{ii} = x_i'(X'X)^{-1}x_i.
$$

HC3 uses:
$$
\hat\Omega_{ii}^{HC3} = \frac{\hat u_i^2}{(1-h_{ii})^2}.
$$

**Interpretation**
- if a point has high leverage ($h_{ii}$ large), it gets more conservative variance contribution.

#### 6) Mapping to code (statsmodels)

In `statsmodels`, you can request HC3 in two common ways:
- `res = sm.OLS(y, X).fit(cov_type='HC3')`
- or `res_hc3 = res.get_robustcov_results(cov_type='HC3')`

#### 7) Diagnostics + robustness (minimum set)

1) **Residual vs fitted plot**
- look for “fan shapes” where variance increases with fitted values.

2) **Compare naive vs HC3 SE**
- report the ratio; big changes mean heteroskedasticity mattered.

3) **Leverage / influential points**
- if a few points dominate, inference is fragile; consider robust checks.

4) **Spec sensitivity**
- add/remove plausible controls; see if estimates are stable (robust SE does not fix omitted variables).

#### 8) Interpretation + reporting

HC3 improves uncertainty estimates under heteroskedasticity.
It does **not**:
- fix bias from confounding,
- make a coefficient causal,
- correct misspecification.

Report:
- coefficient + HC3 SE,
- sample size,
- a quick heteroskedasticity diagnostic (plot or comparison).

#### Exercises

- [ ] Simulate heteroskedastic data and compare naive vs HC3 SE; explain why the coefficient stays similar.
- [ ] Fit the same regression with HC0/HC1/HC3 (if available) and compare SE; which is most conservative?
- [ ] Identify a high-leverage point and explain how HC3 changes its influence on uncertainty.
- [ ] Write 5 sentences: “What robust SE fixes” vs “what it does not fix.”

### Deep Dive: Hypothesis Testing — how to read p-values without fooling yourself

Hypothesis tests show up everywhere in econometrics output. The goal of this section is not to worship p-values, but to understand what they *are* and what they *are not*.

#### 1) Intuition (plain English)

A hypothesis test is a structured way to ask:
- “If the true effect were zero, how surprising is my estimate?”

It is **not** a direct answer to:
- “What is the probability the effect is real?”
- “Is my model correct?”

**Story example:** You regress unemployment on an interest-rate spread and get a small p-value.
That might mean:
- the relationship is real in-sample,
- or your SE are wrong (autocorrelation),
- or you tried many specs (multiple testing),
- or the effect is tiny but precisely estimated.

#### 2) Notation + setup (define symbols)

We usually test a claim about a population parameter $\theta$ (mean, regression coefficient, difference in means, …).

Define:
- $H_0$: the **null hypothesis** (default claim),
- $H_1$: the **alternative hypothesis** (what you consider if evidence contradicts $H_0$),
- $T$: a **test statistic** computed from data,
- $\alpha$: a pre-chosen significance level (e.g., 0.05).

Example in regression:
- $H_0: \beta_j = 0$
- $H_1: \beta_j \neq 0$ (two-sided)

#### 3) Assumptions (why tests are conditional statements)

Every p-value is conditional on:
- the statistical model (e.g., OLS assumptions),
- the standard error estimator you use (naive vs robust vs HAC vs clustered),
- the sample and selection process.

If those assumptions fail, the p-value may be meaningless.

#### 4) Estimation mechanics in OLS: where t-stats come from

OLS estimates coefficients:

$$
\hat\beta = (X'X)^{-1}X'y.
$$

For coefficient $\beta_j$, you compute an estimated standard error $\widehat{SE}(\hat\beta_j)$.

The t-statistic for testing $H_0: \beta_j = 0$ is:

$$
t_j = \frac{\hat\beta_j - 0}{\widehat{SE}(\hat\beta_j)}.
$$

**What each term means**
- numerator: your estimated effect.
- denominator: your uncertainty estimate.
- large |t| means “many standard errors away from 0.”

Under suitable assumptions, $t_j$ is compared to a t distribution (or asymptotic normal), producing a p-value.

#### 5) What the p-value actually means

> **Definition:** The **p-value** is the probability (under the null and model assumptions) of observing a test statistic at least as extreme as what you observed.

So:
- p-value is about the *data under the null model*,
- not about the probability the null is true.

Also: p-values do not measure effect size.

#### 6) Confidence intervals (often more informative than p-values)

A 95% confidence interval is approximately:

$$
\hat\beta_j \pm t_{0.975} \cdot \widehat{SE}(\hat\beta_j).
$$

Interpretation:
- it is a range of values consistent with the data under assumptions,
- it shows both sign and magnitude uncertainty.

If the 95% CI excludes 0, the two-sided p-value is typically < 0.05.

#### 7) Robust SE change p-values (without changing coefficients)

Different SE estimators correspond to different assumptions about errors:
- **Naive OLS SE:** homoskedastic, uncorrelated errors.
- **HC3:** heteroskedasticity-robust (cross-section).
- **HAC/Newey–West:** autocorrelation + heteroskedasticity (time series).
- **Clustered SE:** within-cluster correlated errors (panels/DiD).

**Key idea:** changing SE changes $\widehat{SE}(\hat\beta_j)$ → changes t-stat and p-value, even when $\hat\beta_j$ is identical.

#### 8) Diagnostics: how hypothesis testing goes wrong (minimum set)

1) **Multiple testing**
- If you try many features/specs, some will “work” by chance.
- A few p-values < 0.05 are expected even if all true effects are 0.

2) **P-hacking / specification search**
- tweaking the model until p-values look good invalidates the usual interpretation.

3) **Wrong SE (dependence)**
- autocorrelation or clustering can make naive SE far too small.

4) **Confounding**
- a “significant” association is not a causal effect without identification.

Practical rule:
- interpret p-values as one piece of evidence, not a conclusion.

#### 9) Interpretation + reporting (how to write results responsibly)

Good reporting includes:
- effect size (coefficient) in meaningful units,
- uncertainty (CI preferred),
- correct SE choice for the data structure,
- a note about model limitations and identification.

**What this does NOT mean**
- “Significant” is not “important.”
- “Not significant” is not “no effect” (could be low power).

#### 10) Small Python demo (optional)

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

rng = np.random.default_rng(0)

# 1) One-sample t-test
x = rng.normal(loc=0.2, scale=1.0, size=200)
t_stat, p_val = stats.ttest_1samp(x, popmean=0.0)
print('t-test t:', t_stat, 'p:', p_val)

# 2) Regression t-test
n = 300
x2 = rng.normal(size=n)
y = 1.0 + 0.5 * x2 + rng.normal(scale=1.0, size=n)

df = pd.DataFrame({'y': y, 'x': x2})
X = sm.add_constant(df[['x']])
res = sm.OLS(df['y'], X).fit()
print(res.summary())
```

#### Exercises

- [ ] Take one regression output and rewrite it in words: coefficient, CI, and what assumptions the p-value relies on.
- [ ] Show how p-values change when you switch from naive SE to HC3 or HAC (same coefficient, different uncertainty).
- [ ] Create a multiple-testing demonstration: test 50 random predictors against random noise and count how many p-values < 0.05.
- [ ] Write 6 sentences explaining why “statistically significant” is not the same as “economically meaningful.”

### Deep Dive: Multicollinearity and VIF — why coefficients become unstable

Multicollinearity is common in economic data and is one of the main reasons coefficient interpretation becomes fragile.

#### 1) Intuition (plain English)

If two predictors contain almost the same information, the regression struggles to decide “which variable deserves the credit.”

**Story example (macro):**
- many indicators co-move with the business cycle,
- a multifactor regression may assign unstable signs to “similar” indicators depending on sample period.

Prediction may still be fine, but coefficient stories become unreliable.

**Core intuition:** Regression asks: "What is the effect of changing $x_j$ while holding everything else fixed?" When two predictors move together in the data, you rarely observe one changing without the other. The data provides little information about the separate effect of each. It is like trying to determine whether salt or pepper makes a dish tasty when they are always added together — any split of credit is fragile.

#### 2) Notation + setup (define symbols)

Regression in matrix form:

$$
\mathbf{y} = \mathbf{X}\beta + \varepsilon.
$$

OLS estimator:
$$
\hat\beta = (X'X)^{-1}X'y.
$$

Under classical assumptions:
$$
\mathrm{Var}(\hat\beta \mid X) = \sigma^2 (X'X)^{-1}.
$$

**What each term means**
- When columns of $X$ are highly correlated, $X'X$ is close to singular.
- Then $(X'X)^{-1}$ has large entries → coefficient variance inflates.

#### 3) What multicollinearity does (and does not do)

- Often does **not** hurt prediction much.
- **Does** inflate standard errors (coefficients become noisy).
- **Does** make coefficients sensitive to small data changes (unstable signs/magnitudes).
- **Does not** automatically bias coefficients if exogeneity holds; it mostly increases variance.

#### 4) VIF (Variance Inflation Factor): what it measures

To compute VIF for feature $j$:
- regress $x_j$ on all other predictors,
- record the $R_j^2$ from that auxiliary regression.

Then:

$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^2}.
$$

**Interpretation**
- If $R_j^2$ is near 1, $x_j$ is almost perfectly explained by other predictors.
- Then $\mathrm{VIF}_j$ is large → coefficient uncertainty for $\beta_j$ is inflated.

Rules of thumb (not laws):
- VIF > 5 suggests notable collinearity.
- VIF > 10 suggests serious collinearity (it means $R_j^2 > 0.90$, i.e., more than 90% of $x_j$'s variance is explained by the other predictors — there is very little "unique" information left for the regression to use).

#### 5) Estimation mechanics: “holding others fixed” becomes unrealistic

Coefficient interpretation relies on the counterfactual:
- “Increase $x_j$ by 1 while holding other predictors fixed.”

If predictors are tightly linked economically, that counterfactual can be meaningless (you cannot change one indicator while freezing another).

So multicollinearity is both:
- a statistical issue (variance inflation),
- and an economic interpretation issue (counterfactuals).

#### 6) Diagnostics + robustness (minimum set)

1) **Correlation matrix**
- identify groups of highly correlated features.

2) **VIF table**
- quantify redundancy; large VIF → unstable coefficient.

3) **Coefficient stability**
- fit the regression on different subperiods or bootstrap samples; do signs flip?

4) **Condition number**
- large condition number of $X$ suggests numerical instability.

#### 7) What to do about multicollinearity

Options (choose based on goals):

- **If you care about interpretation**
  - drop redundant variables,
  - combine variables into an index,
  - use domain-driven composites.

- **If you care about prediction**
  - use regularization (ridge/lasso),
  - use dimension reduction (PCA/factors),
  - use nonlinear models (trees) with care about leakage and evaluation.

#### 8) Interpretation + reporting

When multicollinearity is present, report:
- VIF or correlation evidence,
- coefficient instability (if observed),
- and avoid strong stories about individual coefficients.

#### Exercises

- [ ] Construct two highly correlated predictors and show VIF > 10.
- [ ] Fit OLS with both predictors; observe coefficient instability vs the true DGP.
- [ ] Drop one predictor and compare interpretability and fit.
- [ ] Fit ridge regression and compare coefficient stability to OLS.

### Project Code Map
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`) + multicollinearity (`vif_table`)
- `src/macro.py`: GDP + labels (`gdp_growth_*`, `technical_recession_label`)
- `src/evaluation.py`: regression metrics helpers
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Common Mistakes
- Interpreting a coefficient as causal without a causal design.
- Ignoring multicollinearity (high VIF) and over-trusting coefficient signs.
- Using naive SE on time series and over-trusting p-values.

<a id="summary"></a>
## Summary + Suggested Readings

Regression is the core bridge between statistics and ML. You should now be able to:
- fit interpretable linear models,
- quantify uncertainty (robust SE), and
- diagnose when coefficients are unstable.


Suggested readings:
- Wooldridge: Introductory Econometrics (OLS, robust SE, interpretation)
- Angrist & Pischke: Mostly Harmless Econometrics (causal thinking)
- statsmodels docs: robust covariance (HCx, HAC)
