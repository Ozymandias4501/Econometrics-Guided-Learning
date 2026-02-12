# Guide: 06_common_statistical_tests

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00a_statistics_primer/06_common_statistical_tests.ipynb`.

This guide covers the practical application of hypothesis testing through specific statistical tests. Each test answers a different question about your data: Is a mean different from a target? Do two groups differ? Are categorical variables independent? Are regression coefficients jointly significant? Choosing the right test is as important as running it correctly.

### Key Terms (defined)
- **One-sample t-test**: tests whether a sample mean equals a specified value.
- **Two-sample t-test (Welch's)**: tests whether two group means differ, allowing unequal variances.
- **Paired t-test**: tests whether the mean difference in paired observations is zero.
- **Chi-squared test of independence**: tests whether two categorical variables are associated.
- **Chi-squared goodness-of-fit**: tests whether observed frequencies match expected frequencies.
- **F-test (regression)**: tests whether a group of coefficients are jointly zero.
- **Levene's test**: tests whether two groups have equal variances (more robust than classic F-test).

### How To Read This Guide
- Use **Step-by-Step** for the implementation checklist.
- Use **Technical Explanations** for test formulas and assumptions.
- Use the decision tree to choose the right test for your question.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Run and interpret a one-sample t-test on GDP growth.
- Compare GDP growth in recession vs non-recession quarters (two-sample t-test).
- Simulate paired data and run a paired t-test.
- Build a contingency table and run a chi-squared test of independence.
- Test GDP growth for normality using chi-squared goodness-of-fit.
- Test for equal variances across groups.
- Read and interpret the F-statistic from a regression summary.

### Test Selection Guide

| Question | Data type | Test |
|---|---|---|
| Is the mean equal to X? | Continuous, one group | One-sample t-test |
| Do two group means differ? | Continuous, two groups | Two-sample t-test (Welch's) |
| Is the mean change nonzero? | Continuous, paired | Paired t-test |
| Are two categories associated? | Categorical, two variables | Chi-squared independence |
| Does data follow a distribution? | Any, one variable | Chi-squared goodness-of-fit |
| Do two groups have equal variance? | Continuous, two groups | Levene's test |
| Are coefficients jointly zero? | Regression model | F-test |

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### One-sample t-test

$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}, \quad \text{df} = n - 1.
$$

### Two-sample t-test (Welch's)

$$
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}.
$$

Degrees of freedom approximated by the Welch-Satterthwaite equation.

### Chi-squared test of independence

For an $r \times c$ contingency table with observed counts $O_{ij}$ and expected counts $E_{ij} = (\text{row}_i \text{ total} \times \text{col}_j \text{ total}) / n$:

$$
\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad \text{df} = (r-1)(c-1).
$$

**Assumption**: expected counts should be $\geq 5$ in each cell.

### Regression F-test

Tests $H_0$: all slope coefficients are jointly zero.

$$
F = \frac{(SSR_{\text{restricted}} - SSR_{\text{unrestricted}}) / q}{SSR_{\text{unrestricted}} / (n - k - 1)} \sim F_{q, n-k-1}.
$$

<a id="summary"></a>
## Summary + Suggested Readings

1. Choose the test based on your question and data type.
2. Check assumptions before running (normality for t-tests, expected counts for chi-squared).
3. The regression F-test answers "Do these variables matter collectively?" even if none is individually significant.
4. Always report the test used, the test statistic, and the p-value.

### Suggested Readings
- Wooldridge, *Introductory Econometrics* — Chapter 4 (t-tests), Chapter 4.5 (F-tests).
- Agresti, *Categorical Data Analysis* — Chapter 2 on chi-squared tests.
- Rice, *Mathematical Statistics and Data Analysis* — Chapters 9-11 on hypothesis tests.
