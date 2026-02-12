# Guide: 02_sampling_and_central_limit_theorem

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00a_statistics_primer/02_sampling_and_central_limit_theorem.ipynb`.

The Central Limit Theorem (CLT) is arguably the most important result in applied statistics. It explains why inference works: regardless of the shape of the underlying data, sample means become approximately normally distributed as sample size grows. This justifies t-tests, confidence intervals, and regression inference even when the raw data is skewed or non-normal.

### Key Terms (defined)
- **Population**: the complete set of values you want to learn about (often theoretical/unobserved).
- **Sample**: the subset of values you actually observe.
- **Sampling distribution**: the distribution of a statistic (e.g., $\bar{X}$) across repeated samples.
- **Standard error (SE)**: the standard deviation of a sampling distribution. For the mean: $SE = \sigma / \sqrt{n}$.
- **Law of Large Numbers (LLN)**: $\bar{X}_n \to \mu$ as $n \to \infty$.
- **Central Limit Theorem (CLT)**: $\sqrt{n}(\bar{X}_n - \mu) / \sigma \to N(0, 1)$ as $n \to \infty$.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** for formal statements and caveats.
- Then return to the notebook and verify the CLT through simulation.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Demonstrate sampling variability by drawing repeated samples.
- Show LLN convergence with a running mean plot.
- Simulate the CLT from non-normal distributions (uniform, exponential, bimodal).
- Test when the "n=30 rule" breaks down (very skewed distributions).
- Verify that the standard error equals $\sigma / \sqrt{n}$ empirically.
- Apply bootstrap resampling to real economic data.

### Alternative Example (Not the Notebook Solution)
```python
# CLT from an exponential distribution
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
population = rng.exponential(scale=5.0, size=100_000)

sample_means = [rng.choice(population, size=100).mean() for _ in range(2000)]
plt.hist(sample_means, bins=40, density=True, alpha=0.7)
plt.title('Sampling distribution of mean (n=100) from exponential data')
plt.xlabel('Sample mean')
plt.show()
# The histogram looks approximately normal despite the skewed population.
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Formal statement of the CLT

Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with mean $\mu$ and finite variance $\sigma^2 > 0$. Then:

$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1) \quad \text{as } n \to \infty.
$$

Equivalently, $\bar{X}_n$ is approximately $N(\mu, \sigma^2/n)$ for large $n$.

### When the CLT breaks down
- **Infinite variance**: distributions like the Cauchy have no finite variance, so the CLT does not apply.
- **Strong dependence**: time-series data violates the independence assumption. Modified CLTs exist for weakly dependent data, but convergence is slower.
- **Small n with extreme skew**: the "n=30" rule is a rough guideline. For very skewed data (e.g., healthcare costs), you may need $n > 100$ or more.

### The bootstrap as a CLT substitute
When analytical formulas for the SE are unavailable or unreliable, the bootstrap provides an empirical sampling distribution by resampling from the observed data with replacement. It relies on the same asymptotic theory as the CLT but does not require distributional assumptions.

<a id="summary"></a>
## Summary + Suggested Readings

1. The **LLN** guarantees that sample averages converge to population means.
2. The **CLT** guarantees that sample averages are approximately normal, enabling inference.
3. The **standard error** quantifies sampling uncertainty and shrinks with $\sqrt{n}$.
4. The CLT requires independence and finite variance; violations demand modified methods.

### Suggested Readings
- Wooldridge, *Introductory Econometrics* — Appendix C.3 on sampling distributions.
- Casella & Berger, *Statistical Inference* — Chapter 5 on convergence.
- Efron & Tibshirani, *An Introduction to the Bootstrap* — practical bootstrap methods.
