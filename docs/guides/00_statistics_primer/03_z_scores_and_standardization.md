# Guide: 03_z_scores_and_standardization

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_statistics_primer/03_z_scores_and_standardization.ipynb`.

A z-score tells you how many standard deviations a value is from the mean. This simple transformation is the basis of every t-statistic in regression output: $t_j = \hat\beta_j / SE(\hat\beta_j)$ is a z-score for the coefficient. Understanding standardization here makes all of regression inference more intuitive.

### Key Terms (defined)
- **Z-score**: $z = (x - \bar{x}) / s$. Unitless measure of how far a value is from the mean.
- **Standard normal distribution**: $N(0, 1)$. The distribution of z-scores when the original data is normal.
- **Empirical rule (68-95-99.7)**: for normal data, approximately 68%/95%/99.7% of values fall within 1/2/3 standard deviations of the mean.
- **Standardization**: transforming a variable to have mean 0 and standard deviation 1.
- **Normalization**: often refers to min-max scaling to [0, 1]. Different from standardization.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** for the mathematical details.
- Then return to the notebook and complete the exercises.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Compute z-scores manually and with `scipy.stats.zscore`.
- Verify the empirical rule on normal and non-normal data.
- Standardize multiple economic indicators for comparison.
- Use z-scores for outlier detection.
- Identify when standardization is required for ML algorithms.

### Alternative Example (Not the Notebook Solution)
```python
import numpy as np
from scipy import stats

# Standardize two variables on different scales
gdp_growth = np.array([2.1, 3.5, -0.8, 1.2, 4.0])  # percent
unemployment = np.array([5.2, 4.8, 7.1, 6.0, 4.5])  # percent, different scale

z_gdp = stats.zscore(gdp_growth)
z_unemp = stats.zscore(unemployment)

# Now both are on the same scale (standard deviations from their respective means)
print("Z-scores GDP:", z_gdp.round(2))
print("Z-scores Unemp:", z_unemp.round(2))
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### The z-score transformation

For a sample $x_1, \ldots, x_n$ with mean $\bar{x}$ and standard deviation $s$:

$$
z_i = \frac{x_i - \bar{x}}{s}.
$$

**Properties**: The z-scored data has mean 0 and standard deviation 1. If the original data is normally distributed, the z-scores follow $N(0, 1)$.

### Connection to hypothesis testing

The t-statistic for a regression coefficient is:

$$
t_j = \frac{\hat\beta_j - 0}{SE(\hat\beta_j)} = \frac{\hat\beta_j}{SE(\hat\beta_j)}.
$$

This is a z-score: "how many standard errors is the coefficient from zero?" Large |t| means the coefficient is far from zero relative to its uncertainty.

### When to standardize for ML

| Algorithm | Standardize? | Why |
|---|---|---|
| OLS regression | Not required | Coefficients adjust to variable scales |
| Ridge / Lasso | **Required** | Penalty treats all coefficients equally |
| PCA | **Required** | Variance-based; scale affects components |
| K-means | **Required** | Distance-based; scale affects clusters |
| Decision trees | Not required | Split thresholds adapt to scale |

<a id="summary"></a>
## Summary + Suggested Readings

1. Z-scores provide a universal scale for comparing values across different variables.
2. The empirical rule connects z-scores to probabilities (for approximately normal data).
3. Standardization is essential for certain ML algorithms and useful for visualization.
4. Every t-statistic in regression output is a z-score.

### Suggested Readings
- Freedman, Pisani & Purves, *Statistics* — Chapter 5 on the normal curve and z-scores.
- James et al., *Introduction to Statistical Learning* — Section 6.2 on why regularization needs standardization.
