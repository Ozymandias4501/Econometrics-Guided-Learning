# Guide: 08_bayesian_vs_frequentist

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00_statistics_primer/08_bayesian_vs_frequentist.ipynb`.

The frequentist and Bayesian paradigms represent two philosophies of probability and statistical inference. This project primarily uses frequentist methods (OLS, p-values, confidence intervals), but understanding the Bayesian perspective enriches your thinking about uncertainty, prior knowledge, and regularization.

### Key Terms (defined)
- **Frequentist probability**: the long-run frequency of an event in repeated experiments.
- **Bayesian probability**: a degree of belief about a proposition, updated by evidence.
- **Prior distribution**: the belief about a parameter before observing data.
- **Likelihood**: the probability of the observed data given a parameter value.
- **Posterior distribution**: the updated belief about a parameter after observing data.
- **Bayes' theorem**: $P(\theta | \text{data}) \propto P(\text{data} | \theta) \cdot P(\theta)$.
- **Credible interval**: a Bayesian interval such that the parameter has a specified probability of lying within it (given data and prior).
- **Conjugate prior**: a prior that, combined with a specific likelihood, yields a posterior in the same family.

### How To Read This Guide
- Use **Step-by-Step** for the implementation checklist.
- Use **Technical Explanations** for Bayes' theorem and the prior-posterior mechanics.
- This is a lighter, more conceptual guide than the others.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Work through a Bayes' theorem calculation (recession detector example).
- Visualize prior → posterior updating with a Beta-Binomial model.
- Compare a frequentist CI to a Bayesian credible interval.
- Show that with large n, different priors converge to the same posterior.
- Discuss connections to regularization (ridge = Gaussian prior).

### Alternative Example (Not the Notebook Solution)
```python
# Bayes' theorem: medical test analogy
# Disease prevalence: 1%, Test sensitivity: 95%, False positive rate: 5%
# If you test positive, what's the probability you have the disease?

prevalence = 0.01
sensitivity = 0.95
false_positive_rate = 0.05

p_positive = sensitivity * prevalence + false_positive_rate * (1 - prevalence)
p_disease_given_positive = (sensitivity * prevalence) / p_positive

print(f"P(disease | positive test): {p_disease_given_positive:.1%}")
# Only about 16%! The low base rate dominates.
```

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Bayes' theorem (parameter estimation form)

$$
\underbrace{P(\theta | \text{data})}_{\text{posterior}} = \frac{\overbrace{P(\text{data} | \theta)}^{\text{likelihood}} \cdot \overbrace{P(\theta)}^{\text{prior}}}{\underbrace{P(\text{data})}_{\text{evidence}}}.
$$

The evidence $P(\text{data})$ is a normalizing constant. In practice, we often write:

$$
\text{posterior} \propto \text{likelihood} \times \text{prior}.
$$

### Beta-Binomial conjugacy

For a coin with unknown probability $\theta$:
- **Prior**: $\theta \sim \text{Beta}(\alpha, \beta)$.
- **Data**: observe $h$ heads and $t$ tails.
- **Posterior**: $\theta \sim \text{Beta}(\alpha + h, \beta + t)$.

A flat prior ($\alpha = \beta = 1$) means "all values of $\theta$ equally likely before seeing data."

### Frequentist CI vs Bayesian credible interval

| | Frequentist CI | Bayesian credible interval |
|---|---|---|
| **Interpretation** | 95% of intervals from repeated samples contain $\theta$ | Given data and prior, 95% probability $\theta$ is in the interval |
| **Parameter** | Fixed but unknown | Random variable with a distribution |
| **Depends on prior?** | No | Yes |
| **With large n** | Approximately equal | Approximately equal |

### Connection to regularization

Ridge regression adds an $L_2$ penalty: $\min_\beta \|y - X\beta\|^2 + \lambda \|\beta\|^2$.

This is equivalent to Bayesian MAP estimation with a Gaussian prior $\beta_j \sim N(0, 1/\lambda)$. The penalty "pulls" coefficients toward zero, just as a prior centered at zero does. Larger $\lambda$ = stronger prior = more shrinkage.

<a id="summary"></a>
## Summary + Suggested Readings

1. Frequentist and Bayesian approaches answer different questions about the same data.
2. Bayes' theorem provides a principled way to update beliefs with evidence.
3. With enough data, Bayesian and frequentist answers converge (the prior gets "washed out").
4. Regularization has a natural Bayesian interpretation as imposing a prior on coefficients.

### Suggested Readings
- Gelman et al., *Bayesian Data Analysis* — the standard reference for applied Bayesian methods.
- McElreath, *Statistical Rethinking* — an accessible introduction to Bayesian thinking.
- Efron, "Bayesians, Frequentists, and Scientists" (2005) — balanced perspective.
- Bishop, *Pattern Recognition and Machine Learning* — Chapter 1-3 on Bayesian foundations.
