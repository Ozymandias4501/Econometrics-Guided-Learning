# Cheatsheet: Hypothesis Testing

## The Framework

| Concept | Definition | Example |
|---|---|---|
| **Null hypothesis ($H_0$)** | The default "no effect" claim you try to reject | $\beta = 0$ (the yield spread has no relationship with GDP growth) |
| **Alternative ($H_A$)** | What you conclude if you reject $H_0$ | $\beta \neq 0$ (the yield spread is associated with GDP growth) |
| **Test statistic** | A number that measures how far the observed data are from what $H_0$ predicts | $t = \frac{\hat\beta}{SE(\hat\beta)}$ — how many standard errors is $\hat\beta$ from zero? |
| **p-value** | Probability of seeing a test statistic this extreme *if $H_0$ were actually true* | $p = 0.03$ means there's a 3% chance of observing this result (or more extreme) under no true effect |
| **Significance level ($\alpha$)** | Pre-chosen threshold for rejection. By convention, typically 0.05 | Reject $H_0$ when $p < 0.05$ |
| **Confidence interval** | Range of parameter values consistent with the data at level $\alpha$ | 95% CI of $[0.12, 0.58]$ means: if you repeated the study many times, 95% of such intervals would contain the true $\beta$ |
| **Power ($1 - \beta$)** | Probability of correctly rejecting $H_0$ when it is actually false | Higher power means you're less likely to miss a real effect. Increases with sample size, effect size, and lower $\alpha$ |

## Decision Table

| | $H_0$ true (no real effect) | $H_0$ false (real effect exists) |
|---|---|---|
| **Reject $H_0$** | Type I error (false positive) — you claim an effect that doesn't exist. Probability controlled at $\alpha$ | Correct rejection. You detected a real effect (probability = power) |
| **Fail to reject $H_0$** | Correct — no effect, and you didn't claim one | Type II error (false negative) — a real effect exists but your test didn't detect it. Probability = $\beta$ |

**Key asymmetry**: Failing to reject $H_0$ does not mean $H_0$ is true. It may mean you don't have enough data to detect a real effect (low power). "Absence of evidence is not evidence of absence."

## Common Tests in This Course

### t-test (single coefficient)

$$
t = \frac{\hat\beta - \beta_0}{SE(\hat\beta)}
$$

Tests whether a single coefficient equals a hypothesized value (usually $\beta_0 = 0$). The ratio asks: how many standard errors is $\hat\beta$ from $\beta_0$?

- Under $H_0: \beta = \beta_0$, follows $t(n-k)$ distribution in finite samples
- Two-sided test: reject when $|t| > t_{\alpha/2, n-k}$ (roughly $|t| > 2$ for large samples at $\alpha = 0.05$)
- **In Python**: `res.tvalues['x']`, `res.pvalues['x']`

**Worked example**: You regress GDP growth on the yield spread and get $\hat\beta = 0.42$ with $SE = 0.15$. Then $t = 0.42 / 0.15 = 2.8$. Since $|2.8| > 2$, you reject $H_0$ at the 5% level — the yield spread has a statistically significant association with GDP growth.

### F-test (joint significance)

$$
F = \frac{(SSR_R - SSR_{UR}) / q}{SSR_{UR} / (n - k)}
$$

Tests whether $q$ coefficients are simultaneously zero. Compares a "restricted" model (coefficients forced to zero) against the "unrestricted" model (coefficients free).

- Under $H_0$: $F \sim F(q, n-k)$
- The overall F-test in a regression output tests whether *all* slope coefficients are simultaneously zero
- **In Python**: `res.f_pvalue` (overall), or `res.f_test(r_matrix)` for custom restrictions

**When to use**: You have 4 lag variables and want to test whether they jointly predict the outcome, even if no single lag is individually significant.

### Wald test (general linear restrictions)

- Tests $R\beta = r$ for any restriction matrix $R$ and vector $r$
- Asymptotically equivalent to the F-test but uses the $\chi^2$ distribution
- More flexible: can test nonlinear combinations of coefficients
- **In Python**: `res.wald_test(r_matrix)`

## Standard Error Variants

The choice of standard errors determines whether your p-values and confidence intervals are trustworthy. Using the wrong SE is one of the most common mistakes in applied work.

| Data structure | Problem | SE type | Python code |
|---|---|---|---|
| Cross-section, homoskedastic | None (assumptions satisfied) | Classical (default) | `model.fit()` |
| Cross-section, heteroskedastic | Error variance varies across observations | HC3 robust | `res.get_robustcov_results(cov_type='HC3')` |
| Time series | Errors are correlated over time | HAC / Newey-West | `fit_ols_hac(df, y_col='y', x_cols=['x'], maxlags=L)` |
| Panel data | Errors correlated within entities | Clustered | `PanelOLS(...).fit(cov_type='clustered', cluster_entity=True)` |

**Why this matters**: If errors are heteroskedastic or autocorrelated, classical SE are typically too small. This makes t-statistics too large and p-values too small, so you reject $H_0$ more often than you should — finding "significant" results that aren't really there.

**Rule of thumb for HAC maxlags**: $L = \lfloor 0.75 \cdot T^{1/3} \rfloor$. For $T = 200$ quarterly observations, $L \approx 4$. For $T = 60$ monthly observations, $L \approx 3$.

## Interpreting p-values: What They Do and Don't Mean

**A p-value IS**: the probability of seeing data this extreme or more extreme, assuming $H_0$ is true. It measures how surprising the data are under the null.

**A p-value IS NOT**:
- The probability that $H_0$ is true (that requires Bayesian reasoning and a prior)
- The probability that the result is "due to chance"
- A measure of effect size or practical importance
- Reproducible across studies (p-values are random variables — they vary from sample to sample)

### Statistical significance vs. economic significance

A large dataset can make a tiny coefficient statistically significant. Suppose you find that a 1pp increase in the yield spread predicts a 0.002pp increase in GDP growth with $p = 0.01$. This is statistically significant but economically meaningless — the effect is too small to matter for any practical purpose.

**Always report**: the coefficient estimate (effect size), its standard error or confidence interval, and the p-value. The coefficient tells you *how much*; the p-value tells you *how confident*.

### Multiple testing

If you test 20 hypotheses at $\alpha = 0.05$, you expect 1 false positive even if every null is true ($20 \times 0.05 = 1$). When screening many variables or specifications:
- Use Bonferroni correction ($\alpha / m$ for $m$ tests) for a conservative adjustment
- Use Benjamini-Hochberg for controlling the false discovery rate (less conservative)
- Or simply be transparent about how many tests you ran

## Quick Decision Flowchart

```
Is your coefficient significant?
├── p < 0.05 → Reject H₀
│   ├── Is the effect size economically meaningful?
│   │   ├── Yes → Report both: "significant and meaningful"
│   │   └── No  → "Statistically significant but economically trivial"
│   ├── Did you use the right SE?
│   │   ├── Cross-section → HC3 (heteroskedasticity-robust)
│   │   ├── Time series → HAC (autocorrelation-robust)
│   │   └── Panel → Clustered by entity
│   └── Did you run many tests?
│       ├── Yes → Adjust for multiple comparisons
│       └── No  → Report as-is
│
└── p ≥ 0.05 → Fail to reject H₀
    ├── Is your sample large enough? (check power)
    │   ├── Small sample / noisy data → Inconclusive — you may lack power to detect a real effect
    │   └── Large sample / clean data → Reasonable evidence that the effect is small or absent
    └── Is the confidence interval informative?
        ├── Narrow CI around zero → Effect is likely small
        └── Wide CI spanning large values → Too uncertain to conclude anything
```
