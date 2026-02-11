# Guide: 01a_random_effects_hausman

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/07_causal/01a_random_effects_hausman.ipynb`.

Fixed effects (FE) eliminates time-invariant unobserved heterogeneity by demeaning within each unit, but it also eliminates any time-invariant regressors you care about (e.g., gender, geography, race). Random effects (RE) offers a more efficient alternative by assuming the unobserved unit effect is uncorrelated with the regressors. The Hausman test arbitrates: if RE and FE estimates disagree significantly, the RE assumption is violated and FE is preferred.

### Key Terms (defined)
- **Random effects (RE)**: a GLS estimator that partially demeans the data, assuming $\mathrm{Cov}(\alpha_i, x_{it}) = 0$.
- **Fixed effects (FE)**: the within estimator, which eliminates $\alpha_i$ entirely by full demeaning.
- **Hausman test**: tests whether the RE and FE estimates are systematically different; rejection implies RE is inconsistent.
- **Partial demeaning**: RE subtracts $\hat\theta \bar{x}_i$ rather than $\bar{x}_i$; $\hat\theta$ depends on the signal-to-noise ratio of the unit effect.
- **Mundlak (1978) approach**: augments the RE model with group means of time-varying regressors; nests both FE and RE.
- **Between estimator**: regresses group means $\bar{y}_i$ on $\bar{x}_i$; captures cross-sectional variation only.
- **Composite error**: $v_{it} = \alpha_i + \varepsilon_{it}$, where $\alpha_i$ is the unit effect and $\varepsilon_{it}$ is the idiosyncratic error.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Estimate pooled OLS, FE, and RE models
- Complete notebook section: Compare coefficient estimates across estimators
- Complete notebook section: Run the Hausman test
- Complete notebook section: Implement the Mundlak (1978) approach
- Interpret the Hausman test result and state which estimator you prefer and why.
- Report clustered SE for all panel estimators.

### Alternative Example (Not the Notebook Solution)
```python
# Toy panel RE vs FE comparison (not the notebook data):
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects

rng = np.random.default_rng(0)
N, T = 50, 10
entity = np.repeat(np.arange(N), T)
time = np.tile(np.arange(T), N)
alpha_i = rng.normal(0, 2, N)  # unit effects
x = rng.normal(size=N*T) + np.repeat(alpha_i, T) * 0.5  # correlated with alpha
y = 1 + 2 * x + np.repeat(alpha_i, T) + rng.normal(size=N*T)

df = pd.DataFrame({'y': y, 'x': x, 'entity': entity, 'time': time})
df = df.set_index(['entity', 'time'])

res_fe = PanelOLS(df['y'], df[['x']], entity_effects=True).fit()
res_re = RandomEffects(df['y'], df[['x']]).fit()
print(f"FE beta: {res_fe.params['x']:.3f}")
print(f"RE beta: {res_re.params['x']:.3f}")
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: RE assumptions and GLS partial demeaning

The RE estimator is a GLS procedure that exploits both within-unit and between-unit variation. Its validity hinges on one critical assumption.

#### 1) Intuition (plain English)

In a panel, each unit $i$ has an unobserved "type" $\alpha_i$ (e.g., a firm's management quality, a county's geography). FE eliminates $\alpha_i$ entirely by comparing each unit only to itself over time. This is safe but wasteful -- it throws away all cross-sectional variation and cannot estimate effects of time-invariant variables.

RE assumes $\alpha_i$ is random noise uncorrelated with the regressors. Under this assumption, RE uses *both* within and between variation, producing more efficient estimates.

#### 2) The panel model

$$
y_{it} = x_{it}'\beta + \alpha_i + \varepsilon_{it}
$$

where:
- $\alpha_i$ is the unobserved unit effect (time-invariant),
- $\varepsilon_{it}$ is the idiosyncratic error,
- the composite error is $v_{it} = \alpha_i + \varepsilon_{it}$.

#### 3) The RE assumption

$$
\mathrm{Cov}(\alpha_i, x_{it}) = 0 \quad \text{for all } t.
$$

If this holds, RE is consistent *and* more efficient than FE. If this fails, RE is inconsistent -- it inherits omitted variable bias from the correlation between $\alpha_i$ and $x_{it}$.

#### 4) The RE transformation: partial demeaning

The RE estimator applies a GLS transformation. Define:

$$
\hat\theta = 1 - \sqrt{\frac{\hat\sigma_\varepsilon^2}{\hat\sigma_\varepsilon^2 + T\,\hat\sigma_\alpha^2}}
$$

where $\hat\sigma_\varepsilon^2$ is the idiosyncratic variance and $\hat\sigma_\alpha^2$ is the between-unit variance.

The transformed model:

$$
(y_{it} - \hat\theta\,\bar{y}_i) = (x_{it} - \hat\theta\,\bar{x}_i)'\beta + \text{error}
$$

- When $\hat\theta = 1$ (large $\sigma_\alpha^2$): RE $\to$ FE (full demeaning).
- When $\hat\theta = 0$ (small $\sigma_\alpha^2$): RE $\to$ pooled OLS (no demeaning).

So RE is a *weighted average* of within and between variation, with the weight determined by the relative importance of the unit effect.

#### 5) Variance components estimation

The variance components $\sigma_\alpha^2$ and $\sigma_\varepsilon^2$ are typically estimated from the FE residuals:

$$
\hat\sigma_\varepsilon^2 = \frac{1}{NT - N - K} \sum_i \sum_t \hat\varepsilon_{it,FE}^2
$$

$$
\hat\sigma_\alpha^2 = \hat\sigma_{\text{between}}^2 - \frac{\hat\sigma_\varepsilon^2}{T}
$$

where $\hat\sigma_{\text{between}}^2$ comes from the between regression of $\bar{y}_i$ on $\bar{x}_i$.

#### Exercises

- [ ] Estimate FE and RE on the same panel data. Compare the coefficient estimates. Which are more precise (smaller SE)?
- [ ] Compute $\hat\theta$ from the variance components and verify it matches the RE software output.
- [ ] Show that when $T$ is very large, $\hat\theta \to 1$ and RE converges to FE.
- [ ] Estimate the between regression ($\bar{y}_i$ on $\bar{x}_i$) and compare its coefficients with FE and RE. What does the between estimator capture?

---

### Deep Dive: Hausman test statistic and interpretation

The Hausman (1978) test is the standard tool for choosing between FE and RE. It exploits the fact that FE is consistent whether or not the RE assumption holds, while RE is efficient only when the assumption holds.

#### 1) Intuition (plain English)

The logic is simple: if RE is valid, FE and RE should produce similar coefficient estimates (their difference is just noise). If RE is invalid (because $\alpha_i$ is correlated with $x_{it}$), FE and RE will systematically disagree.

The Hausman test formalizes "how different is too different?"

#### 2) The test statistic

Let $\hat\beta_{FE}$ and $\hat\beta_{RE}$ be the FE and RE coefficient vectors, and let:

$$
\hat{q} = \hat\beta_{FE} - \hat\beta_{RE}
$$

Under $H_0$ (RE is valid), the Hausman statistic is:

$$
H = \hat{q}'\left[\widehat{\mathrm{Var}}(\hat\beta_{FE}) - \widehat{\mathrm{Var}}(\hat\beta_{RE})\right]^{-1}\hat{q} \sim \chi^2(K)
$$

where $K$ is the number of time-varying regressors.

**Key property**: under $H_0$, $\mathrm{Var}(\hat\beta_{FE}) - \mathrm{Var}(\hat\beta_{RE})$ is positive semi-definite because RE is efficient and FE has weakly larger variance.

#### 3) Interpreting the result

- **Fail to reject $H_0$** (large p-value): the data are consistent with the RE assumption. Prefer RE for efficiency.
- **Reject $H_0$** (small p-value): FE and RE disagree significantly. The RE assumption is suspect. Prefer FE (or investigate further with Mundlak).

**Caution**: a non-rejection does not prove RE is valid. The test may lack power, especially with:
- small $N$ (few entities),
- short $T$ (few time periods),
- weak between-variation in the regressors.

#### 4) Python implementation

```python
from linearmodels.panel import PanelOLS, RandomEffects, compare

# Estimate both models
res_fe = PanelOLS(df['y'], df[['x1', 'x2']], entity_effects=True).fit()
res_re = RandomEffects(df['y'], df[['x1', 'x2']]).fit()

# Manual Hausman test
import numpy as np
from scipy import stats

b_fe = res_fe.params
b_re = res_re.params
q = b_fe - b_re
V_diff = res_fe.cov - res_re.cov

# Ensure we use only time-varying regressors (drop constant if present)
H_stat = float(q.T @ np.linalg.inv(V_diff) @ q)
p_val = 1 - stats.chi2.cdf(H_stat, df=len(q))
print(f"Hausman H = {H_stat:.2f}, p-value = {p_val:.4f}")
```

#### 5) Practical issues

- **Negative test statistic**: if $\widehat{\mathrm{Var}}(\hat\beta_{FE}) - \widehat{\mathrm{Var}}(\hat\beta_{RE})$ is not positive semi-definite (due to finite-sample issues), the test statistic can be negative. In this case, the test is inconclusive; consider the Mundlak approach instead.
- **Robust version**: the classical Hausman test assumes homoskedasticity. With clustered or robust SE, use a robust version or the Mundlak test (which is a standard Wald test in a single regression).

#### Exercises

- [ ] Run the Hausman test on a panel dataset and interpret the result. Do you prefer FE or RE?
- [ ] Simulate data where $\mathrm{Cov}(\alpha_i, x_{it}) = 0$ (RE valid) and verify the Hausman test does not reject.
- [ ] Simulate data where $\mathrm{Cov}(\alpha_i, x_{it}) \neq 0$ (RE invalid) and verify the Hausman test rejects.
- [ ] Compare the Hausman test conclusion with a visual comparison of FE vs RE coefficients. Do they tell the same story?

---

### Deep Dive: Mundlak (1978) compromise

The Mundlak approach nests both FE and RE in a single regression, making the Hausman test equivalent to a standard F-test.

#### 1) Intuition (plain English)

Instead of choosing between FE and RE, Mundlak (1978) proposes: run the RE model but add group means of all time-varying regressors as additional controls. If the group means are jointly significant, RE is invalid (the Hausman test would reject). The beauty is that the resulting slope estimates on $x_{it}$ are numerically identical to FE, while also allowing time-invariant regressors.

#### 2) The Mundlak model

$$
y_{it} = x_{it}'\beta + \bar{x}_i'\gamma + \alpha_i^* + \varepsilon_{it}
$$

where $\bar{x}_i = T^{-1}\sum_t x_{it}$ is the within-unit mean and $\alpha_i^* = \alpha_i - \bar{x}_i'\gamma$ is the "purged" unit effect.

Under this specification:
- $\hat\beta$ is identical to the FE estimator of $\beta$.
- $\hat\gamma$ captures the correlation between $\bar{x}_i$ and $\alpha_i$.
- $H_0: \gamma = 0$ is equivalent to the Hausman test.

#### 3) Why Mundlak is often preferred in practice

1. **Time-invariant regressors**: unlike FE, Mundlak can estimate effects of time-invariant variables (e.g., gender, region) by including them alongside the group means.
2. **Robust testing**: the test $H_0: \gamma = 0$ is a standard Wald/F-test, which is easy to make robust to heteroskedasticity or clustering.
3. **Transparency**: you can see exactly how much the between-variation coefficient ($\beta + \gamma$) differs from the within-variation coefficient ($\beta$).

#### 4) Python implementation

```python
import statsmodels.api as sm
import pandas as pd

# Add group means of time-varying regressors
for col in ['x1', 'x2']:
    df[f'{col}_mean'] = df.groupby('entity')[col].transform('mean')

# Estimate RE with Mundlak correction
from linearmodels.panel import RandomEffects

exog_cols = ['x1', 'x2', 'x1_mean', 'x2_mean']
res_mundlak = RandomEffects(df['y'], df[exog_cols]).fit()
print(res_mundlak.summary)

# Test gamma = 0 (equivalent to Hausman)
from linearmodels.panel import PanelOLS
# Or simply check p-values on the _mean variables
```

#### 5) Correlated Random Effects (CRE)

The Mundlak approach is a special case of the broader Correlated Random Effects (CRE) framework. Chamberlain (1984) generalizes it by projecting $\alpha_i$ onto each period's $x_{it}$ separately rather than onto $\bar{x}_i$. In balanced panels, the Mundlak specification is usually sufficient.

#### Exercises

- [ ] Estimate the Mundlak model and verify that $\hat\beta$ on the time-varying regressors matches the FE estimates.
- [ ] Test $H_0: \gamma = 0$ using a Wald test. Compare the conclusion with the classical Hausman test.
- [ ] Add a time-invariant regressor (e.g., a regional dummy) to the Mundlak model and interpret its coefficient.
- [ ] Explain in your own words why the Mundlak approach makes the FE vs RE choice less binary.

### Project Code Map
- `src/causal.py`: panel setup (`to_panel_index`, `make_fips`), FE estimation (`fit_twfe_panel_ols`)
- `src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`)
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/census_api.py`: Census data fetching for county-level panels
- `linearmodels.panel`: `PanelOLS`, `RandomEffects`, `compare`

### Common Mistakes
- Assuming RE is always more efficient than FE. It is only more efficient *if* the RE assumption holds.
- Interpreting a non-rejection of the Hausman test as proof that RE is valid; the test may lack power.
- Forgetting that FE eliminates time-invariant regressors; using FE when you need to estimate those effects.
- Computing the Hausman test with robust SE but using the classical formula that assumes homoskedasticity.
- Not clustering standard errors in the Mundlak regression when observations within entities are correlated.

<a id="summary"></a>
## Summary + Suggested Readings

The FE-RE choice is one of the most consequential decisions in panel econometrics. You should now be able to:
- understand RE as a GLS partial-demeaning estimator and its critical identifying assumption,
- run and interpret the Hausman test to adjudicate between FE and RE,
- implement the Mundlak (1978) approach as a practical compromise, and
- appreciate the trade-off between efficiency (RE) and robustness (FE).

Suggested readings:
- Wooldridge, *Introductory Econometrics*, Ch. 14 (panel data: FE and RE)
- Mundlak (1978), "On the Pooling of Time Series and Cross Section Data" (*Econometrica*)
- Hausman (1978), "Specification Tests in Econometrics" (*Econometrica*)
- Chamberlain (1984), "Panel Data" in *Handbook of Econometrics*, Vol. 2
- Cameron & Trivedi, *Microeconometrics*, Ch. 21 (panel data)
