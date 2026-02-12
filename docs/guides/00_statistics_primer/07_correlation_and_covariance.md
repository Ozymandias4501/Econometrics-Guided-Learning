# Guide: 07_correlation_and_covariance

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_statistics_primer/07_correlation_and_covariance.ipynb`.

Correlation analysis is the gateway to regression. Before fitting any model, you should understand how your variables relate to each other. But correlation is also one of the most misused statistics — especially in economics, where trending time series produce spurious correlations and confounding variables create misleading associations.

### Key Terms (defined)
- **Covariance**: $\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]$. Measures joint variability; depends on scale.
- **Pearson correlation ($r$)**: $r = \text{Cov}(X,Y) / (s_X \cdot s_Y)$. Unit-free, bounded $[-1, 1]$. Measures linear association.
- **Spearman rank correlation ($r_s$)**: Pearson correlation applied to ranks. Measures monotonic (not necessarily linear) association. More robust to outliers.
- **Spurious correlation**: a high correlation between variables with no meaningful relationship, often caused by shared trends or confounders.
- **Confounder**: a variable that causally affects both $X$ and $Y$, creating a correlation between them even if $X$ does not cause $Y$.

### How To Read This Guide
- Use **Step-by-Step** for the implementation checklist.
- Use **Technical Explanations** for mathematical definitions and the spurious correlation mechanism.
- Return to the notebook and complete the exercises.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Compute covariance and Pearson correlation for economic variables.
- Compute Spearman correlation and compare to Pearson.
- Create scatter plots and correlation heatmaps.
- Simulate spurious correlation from independent random walks.
- Simulate a confounding scenario and show how "controlling" removes the association.
- Compute the covariance matrix for a set of predictors.

### Alternative Example (Not the Notebook Solution)
```python
# Spurious correlation from trending data
import numpy as np

rng = np.random.default_rng(42)
n = 200

# Two completely independent random walks
x = np.cumsum(rng.normal(size=n))
y = np.cumsum(rng.normal(size=n))

print(f"Correlation: {np.corrcoef(x, y)[0, 1]:.3f}")
# Likely high (e.g., 0.7+) despite complete independence!
# This is why stationarity matters for time-series correlation.
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Pearson correlation

$$
r_{XY} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}.
$$

**Properties**: symmetric ($r_{XY} = r_{YX}$), unit-free, bounded $[-1, 1]$. $r = \pm 1$ if and only if the points lie exactly on a line.

**Limitation**: Pearson correlation measures *linear* association only. A perfect quadratic relationship ($Y = X^2$) can have $r \approx 0$.

### Why trending time series produce spurious correlations

If $X_t = X_{t-1} + \epsilon_t$ and $Y_t = Y_{t-1} + \eta_t$ where $\epsilon$ and $\eta$ are independent, both $X$ and $Y$ are random walks. Over any finite sample, they tend to wander in the same or opposite directions, producing correlations that reflect shared stochastic trends, not meaningful relationships.

**Solution**: compute correlation on *changes* ($\Delta X_t, \Delta Y_t$) rather than levels. This is why stationarity is required before correlation analysis on time series.

### The covariance matrix in OLS

The OLS coefficient estimator is:

$$
\hat\beta = (X'X)^{-1}X'y.
$$

The covariance matrix of predictors ($X'X$) determines how stable the estimates are. High off-diagonal entries relative to diagonal entries indicate multicollinearity.

<a id="summary"></a>
## Summary + Suggested Readings

1. Correlation measures association, not causation.
2. Pearson captures linear relationships; Spearman captures monotonic ones.
3. Trending time series produce spurious correlations — always check stationarity.
4. The covariance matrix of predictors underlies regression stability.

### Suggested Readings
- Wooldridge, *Introductory Econometrics* — Chapter 2 on simple regression and correlation.
- Granger & Newbold, "Spurious Regressions in Econometrics" (1974) — the classic paper.
- Angrist & Pischke, *Mostly Harmless Econometrics* — Chapter 3 on regression as conditional expectation.
