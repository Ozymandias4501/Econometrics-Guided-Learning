# Guide: 05_hypothesis_testing_foundations

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00a_statistics_primer/05_hypothesis_testing_foundations.ipynb`.

Hypothesis testing is the formal framework for deciding whether observed patterns in data are "real" or could have arisen by chance. Every p-value and every significance star in a regression table is the output of a hypothesis test. This guide covers the logic, the error types, and the most common misinterpretations.

### Key Terms (defined)
- **Null hypothesis ($H_0$)**: the default claim, typically "no effect" or "no difference."
- **Alternative hypothesis ($H_1$)**: what you conclude if the data is inconsistent with $H_0$.
- **Test statistic**: a number measuring how far your estimate is from the null, in standard-error units.
- **p-value**: $P(\text{data this extreme or more} \mid H_0 \text{ is true})$.
- **Significance level ($\alpha$)**: the threshold for rejection (conventionally 0.05).
- **Type I error**: rejecting $H_0$ when it is true (false positive). Probability = $\alpha$.
- **Type II error**: failing to reject $H_0$ when it is false (false negative). Probability = $\beta$.
- **Statistical power**: $1 - \beta$. The probability of correctly detecting a real effect.

### How To Read This Guide
- Use **Step-by-Step** for the implementation checklist.
- Use **Technical Explanations** for formal definitions, the error taxonomy, and power analysis.
- Return to the notebook and complete the simulation exercises.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Write $H_0$ and $H_1$ for several economic questions.
- Simulate data under $H_0$ and visualize the p-value as a tail area.
- Demonstrate that p-values are NOT the probability $H_0$ is true.
- Simulate Type I errors (should occur ~5% of the time at $\alpha = 0.05$).
- Simulate Type II errors and compute power for different sample sizes.
- Demonstrate the multiple testing problem.
- Read and interpret regression t-tests and p-values.

### Alternative Example (Not the Notebook Solution)
```python
import numpy as np
from scipy import stats

# Power simulation: how often do we detect a small effect?
rng = np.random.default_rng(42)
true_effect = 0.3
n = 50
n_sims = 5000

rejections = 0
for _ in range(n_sims):
    sample = rng.normal(loc=true_effect, scale=1.0, size=n)
    _, p = stats.ttest_1samp(sample, popmean=0)
    if p < 0.05:
        rejections += 1

print(f"Power (n={n}, effect={true_effect}): {rejections / n_sims:.2%}")
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### The hypothesis testing procedure

1. State $H_0$ and $H_1$.
2. Choose significance level $\alpha$ (before looking at the data).
3. Compute the test statistic from the data.
4. Compute the p-value (tail probability under $H_0$).
5. If $p < \alpha$, reject $H_0$. Otherwise, fail to reject.

### Error taxonomy

|  | $H_0$ true | $H_0$ false |
|---|---|---|
| **Reject $H_0$** | Type I error ($\alpha$) | Correct ($1-\beta$, power) |
| **Fail to reject** | Correct ($1-\alpha$) | Type II error ($\beta$) |

### What p-values are NOT

A p-value of 0.03 means: "If $H_0$ were true, there is a 3% chance of data this extreme."

It does **NOT** mean:
- "There is a 97% probability the effect is real."
- "The effect is large or economically meaningful."
- "The model is correctly specified."

### Power depends on four things
1. **Effect size**: larger effects are easier to detect.
2. **Sample size ($n$)**: more data = more power.
3. **Significance level ($\alpha$)**: higher $\alpha$ = more power (but more Type I errors).
4. **Noise ($\sigma$)**: less noise = more power.

<a id="summary"></a>
## Summary + Suggested Readings

1. Hypothesis testing formalizes "Is this pattern real or noise?"
2. p-values are often misinterpreted — they are NOT the probability $H_0$ is true.
3. Type I and Type II errors represent the two ways a test can fail.
4. Power analysis tells you whether your sample is large enough to detect the effect you care about.

### Suggested Readings
- Wooldridge, *Introductory Econometrics* — Chapter 4 on hypothesis testing.
- Wasserstein & Lazar, "The ASA Statement on p-Values" (2016).
- Cohen, "The Earth Is Round (p < .05)" (1994) — classic paper on misuse of significance testing.
- Greenland et al., "Statistical tests, P values, confidence intervals, and power" (2016).
