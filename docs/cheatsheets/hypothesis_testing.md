# Cheatsheet: Hypothesis Testing

## The Framework

| Concept | Definition |
|---|---|
| **Null hypothesis ($H_0$)** | The "no effect" claim you try to reject (e.g., $\beta = 0$) |
| **Alternative ($H_A$)** | What you conclude if you reject $H_0$ (e.g., $\beta \neq 0$) |
| **Test statistic** | A number that measures how far the data are from $H_0$ |
| **p-value** | Probability of seeing a test statistic this extreme *if $H_0$ were true* |
| **Significance level ($\alpha$)** | Threshold for rejection (typically 0.05). Reject $H_0$ when $p < \alpha$ |
| **Confidence interval** | Range of parameter values not rejected at level $\alpha$. A 95% CI means: if you repeated the study many times, 95% of CIs would contain the true value |

## Decision Table

| | $H_0$ true | $H_0$ false |
|---|---|---|
| **Reject $H_0$** | Type I error (false positive), probability = $\alpha$ | Correct (power = $1 - \beta$) |
| **Fail to reject** | Correct | Type II error (false negative), probability = $\beta$ |

## Common Tests in This Course

### t-test (single coefficient)

$$
t = \frac{\hat\beta - \beta_0}{SE(\hat\beta)}
$$

- Under $H_0: \beta = \beta_0$, follows $t(n-k)$ distribution
- Two-sided test: reject when $|t| > t_{\alpha/2, n-k}$
- **In Python**: `res.tvalues['x']`, `res.pvalues['x']`

### F-test (joint significance)

$$
F = \frac{(SSR_R - SSR_{UR}) / q}{SSR_{UR} / (n - k)}
$$

- Tests whether $q$ restrictions hold simultaneously (e.g., "are all slope coefficients zero?")
- Under $H_0$: $F \sim F(q, n-k)$
- **In Python**: `res.f_pvalue` (overall), or `res.f_test(r_matrix)` for custom restrictions

### Wald test (general linear restrictions)

- Tests $R\beta = r$ for any matrix $R$
- Asymptotically equivalent to F-test but uses $\chi^2$ distribution
- **In Python**: `res.wald_test(r_matrix)`

## Standard Error Variants

| Situation | SE type | statsmodels code |
|---|---|---|
| Homoskedastic cross-section | Classical (default) | `res = model.fit()` |
| Heteroskedastic cross-section | HC3 robust | `res.get_robustcov_results(cov_type='HC3')` |
| Time series (autocorrelation) | HAC / Newey-West | `res.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': L})` |
| Panel data (clustered) | Clustered | `PanelOLS(...).fit(cov_type='clustered', cluster_entity=True)` |

**Rule of thumb for HAC maxlags**: $L = \lfloor 0.75 \cdot T^{1/3} \rfloor$. For $T=200$ quarterly obs, $L \approx 4$.

## Interpreting p-values: What They Do and Don't Mean

**A p-value IS**: the probability of observing data this extreme under $H_0$.

**A p-value IS NOT**:
- The probability that $H_0$ is true
- The probability that the result is due to chance
- A measure of effect size or practical importance

**"Statistically significant" $\neq$ "important"**: A massive dataset can make a tiny, practically meaningless coefficient significant. Always report effect sizes alongside p-values.

## Quick Decision Flowchart

```
Is your coefficient significant?
├── p < 0.05 → Reject H₀
│   ├── Is the effect size meaningful?
│   │   ├── Yes → Report both significance and magnitude
│   │   └── No  → Significant but economically trivial
│   └── Did you use the right SE?
│       ├── Cross-section → HC3
│       ├── Time series → HAC
│       └── Panel → Clustered
└── p ≥ 0.05 → Fail to reject H₀
    └── Low power? (small n, noisy data)
        ├── Yes → Inconclusive, not "no effect"
        └── No  → Reasonable evidence of no meaningful effect
```
