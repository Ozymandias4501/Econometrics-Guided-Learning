# Guide: 00_descriptive_statistics

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_statistics_primer/00_descriptive_statistics.ipynb`.

Descriptive statistics are the first tool you reach for with any new dataset. They summarize the central tendency (where is the data centered?), spread (how variable is it?), and shape (is it symmetric? heavy-tailed?) of your variables. These summaries inform every modeling decision downstream — from choosing transformations to detecting data quality issues.

### Key Terms (defined)
- **Mean ($\bar{x}$)**: the arithmetic average. Sensitive to outliers.
- **Median**: the middle value when data is sorted. Robust to outliers.
- **Mode**: the most frequent value. Most useful for discrete/categorical data.
- **Variance ($s^2$)**: average squared deviation from the mean. Measures spread.
- **Standard deviation ($s$)**: square root of variance. Same units as the data.
- **Skewness**: measures asymmetry. Positive skew = right tail is longer. Negative skew = left tail is longer.
- **Kurtosis**: measures tail heaviness relative to a normal distribution. Excess kurtosis > 0 means heavier tails than normal.
- **IQR (Interquartile Range)**: Q3 - Q1. Robust measure of spread.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions.
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Compute mean, median, and mode for economic variables.
- Compute variance, standard deviation, range, and IQR.
- Compute and interpret skewness and kurtosis.
- Use `df.describe()` and interpret the output.
- Create histograms, box plots, and KDE plots.
- Identify when the mean is misleading (skewed data).

### Alternative Example (Not the Notebook Solution)
```python
# Toy example: income data is right-skewed
import numpy as np

rng = np.random.default_rng(42)
income = rng.lognormal(mean=10.5, sigma=0.8, size=1000)

print(f"Mean income:   ${np.mean(income):,.0f}")
print(f"Median income: ${np.median(income):,.0f}")
# The mean is much higher than the median because of right skew.
# For policy purposes, the median is often more representative.
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: Sample vs population statistics

The **population** parameters ($\mu$, $\sigma^2$) describe the entire data-generating process. The **sample** statistics ($\bar{x}$, $s^2$) are estimates computed from the data you have.

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i, \qquad s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2.
$$

The $n-1$ in the sample variance (Bessel's correction) makes $s^2$ an unbiased estimator of $\sigma^2$.

### Deep Dive: Skewness and kurtosis

**Skewness** (Fisher's definition):

$$
g_1 = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3.
$$

- $g_1 > 0$: right-skewed (long right tail). Common in: income, housing prices, hospital costs.
- $g_1 < 0$: left-skewed (long left tail). Less common in economics.
- $g_1 \approx 0$: approximately symmetric.

**Excess kurtosis** (Fisher's definition):

$$
g_2 = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^4 - 3.
$$

- $g_2 > 0$: heavier tails than normal (leptokurtic). Common in: financial returns, GDP growth during crises.
- $g_2 < 0$: lighter tails than normal (platykurtic).
- $g_2 \approx 0$: approximately normal tails.

### Common Mistakes
- Using the mean to summarize highly skewed data (use median instead).
- Forgetting to check for missing values before computing statistics.
- Confusing sample statistics with population parameters.
- Interpreting kurtosis as "peakedness" (it measures tail heaviness).

<a id="summary"></a>
## Summary + Suggested Readings

Descriptive statistics are the foundation of data analysis. Before any modeling, you should know:
1. Where your data is centered (mean vs median).
2. How spread out it is (std dev, IQR).
3. Whether it is symmetric or skewed.
4. Whether it has heavy tails (kurtosis).

### Suggested Readings
- Freedman, Pisani & Purves, *Statistics* — Chapters 4-5 on summary statistics.
- Wooldridge, *Introductory Econometrics* — Appendix C on probability and statistics review.
- Tukey, *Exploratory Data Analysis* — The classic on visualization-first data analysis.
