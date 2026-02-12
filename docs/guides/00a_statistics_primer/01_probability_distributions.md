# Guide: 01_probability_distributions

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00a_statistics_primer/01_probability_distributions.ipynb`.

Probability distributions are the mathematical models that describe randomness. Every hypothesis test, every confidence interval, and every p-value in this project assumes some underlying distribution. This guide provides the mathematical details behind the distributions you will encounter: Normal, t, chi-squared, F, binomial, and Poisson.

### Key Terms (defined)
- **PDF (Probability Density Function)**: for continuous distributions, gives the relative likelihood of a value. The area under the PDF integrates to 1.
- **PMF (Probability Mass Function)**: for discrete distributions, gives the probability of each specific value.
- **CDF (Cumulative Distribution Function)**: $F(x) = P(X \leq x)$. Works for both continuous and discrete.
- **Parameters**: values that define a specific distribution (e.g., $\mu$ and $\sigma$ for the normal).
- **Degrees of freedom (df)**: a parameter of the t, chi-squared, and F distributions that determines their shape.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** for the mathematical definitions and properties.
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Simulate data from each distribution using `scipy.stats`.
- Plot PDFs/PMFs and CDFs for each distribution.
- Overlay t-distributions with different df on the standard normal.
- Simulate chi-squared as sum of squared normals.
- Simulate F as ratio of chi-squared variables.
- Compare fitted distributions to real economic data.

### Alternative Example (Not the Notebook Solution)
```python
# Visualizing how the t-distribution approaches the normal
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(-4, 4, 300)
plt.plot(x, stats.norm.pdf(x), 'k-', lw=2, label='Normal')
for df in [2, 5, 30]:
    plt.plot(x, stats.t.pdf(x, df), '--', label=f't (df={df})')
plt.legend()
plt.title('t-distribution converges to normal as df increases')
plt.show()
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### The Normal Distribution

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad x \in \mathbb{R}.
$$

Parameters: mean $\mu$ (location), standard deviation $\sigma$ (scale). Notation: $X \sim N(\mu, \sigma^2)$.

**Key properties**: symmetric, bell-shaped, fully determined by $\mu$ and $\sigma$. The standard normal is $N(0, 1)$.

### The t-Distribution

$$
f(x) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}.
$$

Parameter: degrees of freedom $\nu$. Heavier tails than normal; converges to $N(0,1)$ as $\nu \to \infty$. Arises when estimating a mean with unknown variance from a normal population.

### The Chi-Squared Distribution

If $Z_1, \ldots, Z_k \sim N(0,1)$ independently, then $Q = \sum_{i=1}^k Z_i^2 \sim \chi^2_k$.

Parameter: degrees of freedom $k$. Always non-negative and right-skewed. Appears in variance tests, goodness-of-fit tests, and heteroskedasticity diagnostics.

### The F-Distribution

If $U \sim \chi^2_{d_1}$ and $V \sim \chi^2_{d_2}$ independently, then $F = \frac{U/d_1}{V/d_2} \sim F_{d_1, d_2}$.

Parameters: numerator df $d_1$, denominator df $d_2$. Always non-negative. Appears in ANOVA, joint hypothesis tests in regression, and comparing nested models.

### Discrete: Binomial and Poisson

**Binomial**: $X \sim \text{Bin}(n, p)$ counts successes in $n$ independent trials with success probability $p$.

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}.
$$

**Poisson**: $X \sim \text{Pois}(\lambda)$ counts events in a fixed interval with rate $\lambda$.

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}.
$$

<a id="summary"></a>
## Summary + Suggested Readings

Every statistical test maps to an assumed distribution. Know the shape, parameters, and when each arises:
1. **Normal**: regression errors, CLT-justified inference.
2. **t**: small-sample coefficient tests.
3. **Chi-squared**: variance tests, goodness-of-fit, heteroskedasticity.
4. **F**: joint significance, ANOVA, model comparison.
5. **Binomial/Poisson**: count data in economics.

### Suggested Readings
- DeGroot & Schervish, *Probability and Statistics* — Chapters 5-6 on common distributions.
- Wooldridge, *Introductory Econometrics* — Appendix C.
- Rice, *Mathematical Statistics and Data Analysis* — Chapter 3-4.
