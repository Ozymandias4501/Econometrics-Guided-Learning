### Instrumental Variables (IV) and 2SLS: isolating “as-if random” variation

IV is used when a key regressor is endogenous (correlated with the error term).

#### 1) Intuition (plain English)

Suppose you want the effect of $X$ on $Y$.
If $X$ is chosen by people/policymakers/markets, it likely reflects information that also affects $Y$.

**Story example:** Education ($X$) and earnings ($Y$).
- Ability affects both schooling and earnings.
- OLS confounds the schooling effect with ability.

IV tries to find a variable $Z$ (“instrument”) that moves $X$ for reasons unrelated to the unobserved determinants of $Y$.

#### 2) Notation + setup (define symbols)

Structural equation (what you want):

$$
Y_i = \\beta X_i + W_i'\\delta + u_i
$$

First stage (how $Z$ moves $X$):

$$
X_i = \\pi Z_i + W_i'\\rho + v_i
$$

**What each term means**
- $Y_i$: outcome.
- $X_i$: endogenous regressor (the “treatment” intensity).
- $W_i$: exogenous controls (included in both stages).
- $Z_i$: instrument(s).
- $u_i$: structural error (unobservables affecting $Y$).
- $v_i$: first-stage error.

Endogeneity means:
$$
\\mathrm{Cov}(X_i, u_i) \\neq 0.
$$

#### 3) Identification: the two IV conditions

An instrument must satisfy:

1) **Relevance**
$$
\\mathrm{Cov}(Z_i, X_i) \\neq 0
$$
The instrument meaningfully predicts $X$ (a strong first stage).

2) **Exclusion / exogeneity**
$$
\\mathrm{Cov}(Z_i, u_i) = 0
$$
$Z$ affects $Y$ only through $X$ (no direct path; no correlation with omitted determinants of $Y$).

These are design assumptions. Exclusion is especially untestable and must be defended with institutional context.

<details>
<summary>Optional: monotonicity and LATE (why IV can be “local”)</summary>

With heterogeneous treatment effects, IV often identifies a **Local Average Treatment Effect (LATE)**: the effect for “compliers” whose $X$ changes when $Z$ changes.
This adds a monotonicity assumption (“$Z$ pushes $X$ in the same direction for everyone”).

</details>

#### 4) Estimation mechanics: from IV estimand to 2SLS

**Single endogenous regressor + single instrument (no controls):**

The IV estimand is:
$$
\\beta_{IV} = \\frac{\\mathrm{Cov}(Z, Y)}{\\mathrm{Cov}(Z, X)}.
$$

**What this means**
- The numerator measures how $Z$ shifts outcomes.
- The denominator measures how $Z$ shifts the endogenous regressor.
- Their ratio interprets the outcome shift “per unit” of the regressor shift induced by the instrument.

**Two-stage least squares (2SLS) with controls:**

Stage 1 (projection):
- regress $X$ on $Z$ and $W$ to get predicted $\\hat X$ (the part of $X$ explained by instruments + controls).

Stage 2:
- regress $Y$ on $\\hat X$ and $W$.

Matrix form (compact):

$$
\\hat\\beta_{2SLS} = (X'P_Z X)^{-1} X'P_Z Y,
$$

where:
- $P_Z$ is the projection matrix onto the space spanned by instruments (and included exogenous controls).

**What each term means**
- $P_Z X$ is “the part of $X$ explained by $Z$.”
- 2SLS replaces endogenous variation in $X$ with instrumented variation.

#### 5) Inference: robust SE and weak-instrument warnings

As in OLS, you need standard errors for uncertainty.
In practice, use robust (and often clustered) SE.

Weak instruments are dangerous:
- if the first stage is weak, 2SLS estimates can be noisy and biased in finite samples.

Practical habit:
- always inspect first-stage strength (and the first-stage regression).

#### 6) Diagnostics + robustness (minimum set)

1) **First-stage strength**
- Check the magnitude and significance of $\\pi$ in the first stage.
- Use a “weak IV” diagnostic (rule-of-thumb first-stage F-type checks).

2) **Exclusion plausibility**
- Write the exclusion restriction in words and list at least 2 threats (direct effects, correlated policy, omitted shocks).

3) **Overidentification (if multiple instruments)**
- If you have more instruments than endogenous regressors, you can test instrument consistency (e.g., Hansen’s J). This is not a proof, but a diagnostic.

4) **Sensitivity**
- Try alternative instrument sets (if justified) and see if estimates are stable.

#### 7) Interpretation + reporting

Good IV reporting includes:
- structural equation and first stage written explicitly,
- relevance evidence (first stage),
- a clear statement of the exclusion restriction,
- robust/clustered SE and cluster counts (if clustered).

**What this does NOT mean**
- A strong first stage does not prove exclusion.
- A significant 2SLS coefficient is not necessarily a general ATE; it can be a LATE.

#### Exercises

- [ ] Write down one concrete endogeneity story (omitted variable or reverse causality) for a regression you care about.
- [ ] Propose a plausible instrument and write relevance + exclusion in words.
- [ ] Compute OLS and 2SLS on a simulated dataset where you control the DGP; explain which recovers the “true” effect.
- [ ] Inspect the first stage and write 3 sentences explaining whether you believe it is strong enough.
