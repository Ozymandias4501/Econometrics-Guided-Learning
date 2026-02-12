# Guide: 04_confidence_intervals

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_statistics_primer/04_confidence_intervals.ipynb`.

Confidence intervals quantify uncertainty about estimated parameters. They appear in every regression table (the [0.025, 0.975] columns in statsmodels), every policy report, and every empirical paper. Yet they are among the most commonly misinterpreted statistics. This guide provides the mathematical construction and, more importantly, the correct interpretation.

### Key Terms (defined)
- **Confidence interval (CI)**: a range $[\hat\theta - E, \hat\theta + E]$ constructed so that $(1-\alpha) \times 100\%$ of such intervals (across repeated samples) contain the true parameter.
- **Confidence level**: $(1-\alpha) \times 100\%$. Common choices: 90%, 95%, 99%.
- **Margin of error ($E$)**: half the width of the CI. $E = t_{\alpha/2} \cdot SE$.
- **Critical value**: the cutoff from the t (or z) distribution. For 95% CI with large n: $z_{0.025} \approx 1.96$.

### How To Read This Guide
- Use **Step-by-Step** for the implementation checklist.
- Use **Technical Explanations** for the formula derivation and interpretation nuances.
- Return to the notebook and complete the simulation exercises.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Construct a CI for a mean using the t-distribution.
- Simulate the "repeated sampling" interpretation (100 CIs, count coverage).
- Show how CI width changes with sample size and confidence level.
- Extract and interpret CIs for regression coefficients.
- Discuss margin of error and practical significance.

### Alternative Example (Not the Notebook Solution)
```python
import numpy as np
from scipy import stats

# 95% CI for mean of a sample
rng = np.random.default_rng(42)
sample = rng.normal(loc=3.0, scale=2.0, size=50)

xbar = sample.mean()
se = sample.std(ddof=1) / np.sqrt(len(sample))
t_crit = stats.t.ppf(0.975, df=len(sample) - 1)
ci = (xbar - t_crit * se, xbar + t_crit * se)
print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### CI for a population mean

Given a sample of size $n$ from a population with unknown mean $\mu$:

$$
CI_{1-\alpha} = \bar{x} \pm t_{\alpha/2, \, n-1} \cdot \frac{s}{\sqrt{n}}.
$$

This interval has the property that, across repeated samples, $(1-\alpha) \times 100\%$ of intervals constructed this way will contain $\mu$.

### The correct interpretation (and the common mistake)

**Correct**: "If we repeated this sampling procedure many times, 95% of the resulting intervals would contain the true parameter."

**Incorrect**: "There is a 95% probability that the true parameter is in this interval." The parameter is fixed (not random); it either is or is not in the interval. The randomness is in the interval itself.

### CI width and its determinants

$$
\text{Width} = 2 \cdot t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}.
$$

Width decreases with: larger $n$ (more data), smaller $s$ (less noise), lower confidence level (smaller $t_{\alpha/2}$).

### Duality with hypothesis testing

A 95% CI excludes zero if and only if the two-sided p-value for $H_0: \theta = 0$ is below 0.05. They are dual representations of the same test.

<a id="summary"></a>
## Summary + Suggested Readings

1. CIs quantify estimation uncertainty by providing a plausible range for the parameter.
2. The "repeated sampling" interpretation is the correct one — not "95% probability."
3. Width depends on sample size, variability, and confidence level.
4. CIs often convey more information than p-values (sign, magnitude, precision).

### Suggested Readings
- Wooldridge, *Introductory Econometrics* — Chapter 4 on confidence intervals for regression coefficients.
- Cumming, *Understanding the New Statistics* — excellent visual approach to CIs.
- Wasserstein & Lazar, "The ASA Statement on p-Values" (2016) — argues for CIs over p-values.
