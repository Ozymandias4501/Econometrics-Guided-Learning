# Guide: 03 — Instrumental Variables and Two-Stage Least Squares (2SLS)

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/07_causal/03_instrumental_variables_2sls.ipynb`.

**Scope.** This guide focuses exclusively on Instrumental Variables (IV) and Two-Stage Least Squares (2SLS). For the foundational causal inference framework (potential outcomes, selection bias, identification vs. estimation, the core assumptions that recur across all causal designs), see [Guide 01 — Panel Fixed Effects](01_panel_fixed_effects_clustered_se.md). For a detailed taxonomy of endogeneity sources (omitted variables, measurement error, simultaneity), see [Guide 02a — Endogeneity Sources](02a_endogeneity_sources.md). This guide assumes you have read both.

### Why IV Matters for Health Economics

Health economics is saturated with endogeneity. Almost every interesting question involves a regressor that is correlated with the error term:

- **Treatments are not randomly assigned.** Physicians select treatments based on patient severity, comorbidities, and preferences — all of which independently affect outcomes. Comparing treated and untreated patients confounds the treatment effect with selection on health status.

- **Insurance is not randomly assigned.** Sicker individuals are more likely to purchase comprehensive coverage. Comparing insured and uninsured populations confounds the insurance effect with underlying health differences (adverse selection).

- **Exposure is endogenous.** People who can afford to move away from pollution sources do so. Comparing residents in polluted vs. clean areas confounds pollution effects with socioeconomic determinants of health.

- **Healthcare utilization reflects need, not just supply.** Regions with higher hospital spending may have sicker populations. Comparing spending levels across regions confounds the effect of spending with disease burden.

In each case, OLS estimates of the causal effect are biased, sometimes severely. IV provides a way to isolate exogenous variation in the endogenous regressor and recover a consistent estimate of the causal parameter. It is one of the most widely used identification strategies in health economics, health services research, and pharmacoepidemiology.

### Key Terms (defined)

- **Instrument (Z)**: a variable that shifts the endogenous regressor but has no direct effect on the outcome except through that regressor.
- **Endogenous regressor (X)**: the "treatment" variable that is correlated with the structural error — the variable whose causal effect you want to estimate.
- **Exogenous variable (W)**: a control variable that is uncorrelated with the structural error and appears in both stages.
- **First stage**: the regression of the endogenous regressor $X$ on the instrument $Z$ (and controls $W$). Establishes that the instrument predicts $X$.
- **Second stage (structural equation)**: the regression of the outcome $Y$ on the instrumented (predicted) value $\hat{X}$ (and controls $W$).
- **Reduced form**: the regression of $Y$ directly on $Z$ (and $W$). If the instrument affects $Y$ through $X$, the reduced form coefficient should be nonzero.
- **Exclusion restriction**: the assumption that $Z$ affects $Y$ *only* through $X$, i.e., $\mathrm{Cov}(Z, u) = 0$. Cannot be tested with data.
- **Relevance condition**: the assumption that $Z$ is correlated with $X$, i.e., $\mathrm{Cov}(Z, X) \neq 0$. Can (and must) be tested with data.
- **Weak instruments**: instruments with low predictive power for $X$. Lead to biased and unreliable 2SLS estimates.
- **Overidentification**: having more instruments than endogenous regressors. Allows testing instrument consistency.
- **Just-identified**: having exactly as many instruments as endogenous regressors.
- **LATE (Local Average Treatment Effect)**: the causal effect for the subpopulation of "compliers" whose treatment status is changed by the instrument. This is what IV estimates under heterogeneous treatment effects.
- **Compliers**: individuals whose treatment status changes when the instrument changes (e.g., people who enroll in insurance when eligible but would not otherwise).
- **Always-takers**: individuals who take the treatment regardless of the instrument value.
- **Never-takers**: individuals who never take the treatment regardless of the instrument value.
- **Defiers**: individuals who do the opposite of what the instrument encourages. The monotonicity assumption rules these out.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math, assumptions, and diagnostics.
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)

1. **State the causal question.** Write out: What is the endogenous regressor ($X$)? What is the outcome ($Y$)? Why is $X$ endogenous?
2. **Propose an instrument ($Z$).** Explain in plain language why $Z$ satisfies relevance and exclusion.
3. **Simulate endogeneity.** Create a DGP where $X$ and the error are correlated. Estimate OLS and confirm that the coefficient is biased.
4. **Estimate the first stage.** Regress $X$ on $Z$ (and controls). Report the coefficient and the first-stage F-statistic. Is the instrument strong (F > 10)?
5. **Estimate 2SLS.** Use `linearmodels.iv.IV2SLS` (or the project helper `fit_iv_2sls`). Compare the 2SLS coefficient to the true parameter and to the OLS estimate.
6. **Inspect the reduced form.** Regress $Y$ on $Z$ directly. Is the coefficient nonzero? Is its sign consistent with the first stage and the 2SLS estimate?
7. **Discuss exclusion.** Write 2-3 sentences defending the exclusion restriction. List at least 2 threats.
8. **Robustness.** Try alternative specifications (different controls, different instrument definitions) and report whether the estimate is stable.
9. **Report properly.** Present both the first stage and the structural equation. Use robust or clustered standard errors. State the estimand (LATE vs. ATE) and the complier population.

### Alternative Example (Not the Notebook Solution)

This is a standalone IV simulation to build intuition before working the notebook.

```python
# Toy IV simulation: effect of X on Y with endogeneity
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)
n = 2000
TRUE_BETA = 0.5  # true causal effect of X on Y

# --- Data generating process ---
# Unobserved confounder (e.g., patient severity)
U = np.random.normal(0, 1, n)

# Instrument Z: exogenous (e.g., distance to hospital, policy shock)
Z = np.random.normal(0, 1, n)

# Endogenous regressor X: driven by Z and by the confounder U
X = 0.8 * Z + 0.6 * U + np.random.normal(0, 0.5, n)

# Outcome Y: depends on X (causal) and U (confounder) and noise
Y = TRUE_BETA * X + 0.7 * U + np.random.normal(0, 1, n)

df = pd.DataFrame({"Y": Y, "X": X, "Z": Z})

# --- OLS (biased because X is correlated with U, which is in the error) ---
ols_result = sm.OLS(df["Y"], sm.add_constant(df["X"])).fit(cov_type="HC1")
print(f"OLS estimate of beta: {ols_result.params['X']:.4f}")
print(f"  (True beta = {TRUE_BETA}; OLS is biased upward due to positive Cov(X,U))")

# --- First stage: X = pi*Z + error ---
first_stage = sm.OLS(df["X"], sm.add_constant(df["Z"])).fit(cov_type="HC1")
print(f"\nFirst-stage coefficient on Z: {first_stage.params['Z']:.4f}")
f_stat = first_stage.f_test("Z = 0").fvalue[0][0]
print(f"First-stage F-statistic: {f_stat:.1f} (want > 10)")

# --- 2SLS using linearmodels ---
from linearmodels.iv import IV2SLS

iv_formula = "Y ~ 1 + [X ~ Z]"
iv_result = IV2SLS.from_formula(iv_formula, data=df).fit(cov_type="robust")
print(f"\n2SLS estimate of beta: {iv_result.params['X']:.4f}")
print(f"  (True beta = {TRUE_BETA}; 2SLS is consistent)")
```

**What you should see:**
- OLS overestimates the true effect (~0.70-0.80 instead of 0.50) because $U$ pushes both $X$ and $Y$ in the same direction.
- The first-stage F-statistic is well above 10, confirming a strong instrument.
- 2SLS recovers approximately the true effect of 0.50.

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### 1) The Endogeneity Problem — Intuition with Health Economics Examples

The structural model is:

$$
Y_i = \beta X_i + W_i'\delta + u_i
$$

**Endogeneity** means $\mathrm{Cov}(X_i, u_i) \neq 0$: the regressor of interest is correlated with the unobserved determinants of the outcome. OLS is biased and inconsistent. (For full derivations and the OVB formula, see [Guide 02a — Endogeneity Sources](02a_endogeneity_sources.md).)

Here are four health economics examples where endogeneity is the core problem:

**Example 1: Education and health outcomes.**
You want to estimate how an additional year of schooling affects health (self-reported health, mortality risk, obesity rates). But unobserved ability and family background affect both educational attainment and health behaviors. People with higher ability tend to get more education *and* make healthier choices. OLS conflates the causal effect of education with the ability premium.

**Example 2: Insurance coverage and health spending.**
You want to estimate the effect of having health insurance on healthcare utilization and spending. But enrollment is not random: sicker individuals are more likely to seek coverage (adverse selection), and employers offering generous plans may attract different workers. OLS overstates the effect of insurance on spending because the insured are systematically different from the uninsured.

**Example 3: Hospital volume and surgical mortality.**
You want to know whether high-volume hospitals have better surgical outcomes. But referral patterns are endogenous: complex cases are referred to high-volume centers, and healthier patients may systematically sort to different hospitals. The direction of OLS bias is ambiguous and depends on which selection mechanism dominates.

**Example 4: Air pollution and respiratory health.**
You want to estimate the effect of particulate matter exposure on asthma hospitalizations. But residential location is endogenous: wealthier, healthier families can afford to live in less polluted areas. OLS may overstate the pollution effect because it captures both the direct health impact and the socioeconomic correlates of living near pollution sources.

In every case, the endogeneity has a specific, identifiable structure — and that structure determines what kind of instrument might help.

### 2) What Makes a Good Instrument — The Two Conditions

An instrument $Z$ must satisfy two conditions:

#### Condition 1: Relevance

$$
\mathrm{Cov}(Z_i, X_i) \neq 0
$$

The instrument must predict the endogenous regressor. This is **testable**: you regress $X$ on $Z$ (and controls $W$) and examine whether $Z$ has meaningful explanatory power.

The standard diagnostic is the **first-stage F-statistic** on the excluded instruments. The widely cited rule of thumb is:

> **F > 10** suggests the instrument is not dangerously weak. (Stock, Wright & Yogo 2002; Staiger & Stock 1997.)

This threshold comes from the Stock-Yogo (2005) critical values, which ensure that the 2SLS bias is no more than 10% of the OLS bias. The exact critical value depends on the number of instruments and endogenous regressors — F > 10 is a convenient approximation for the just-identified case with one endogenous regressor.

A weak instrument (low F) does not just make estimates imprecise. It makes them *biased toward OLS* in finite samples, which defeats the entire purpose of IV.

#### Condition 2: Exclusion Restriction (Exogeneity)

$$
\mathrm{Cov}(Z_i, u_i) = 0
$$

The instrument affects the outcome $Y$ **only through** $X$. There is no direct path from $Z$ to $Y$, and $Z$ is uncorrelated with the omitted determinants of $Y$.

This is the harder condition. It **cannot be tested with data**. It must be defended using institutional knowledge, economic reasoning, and careful argument about the causal structure. The exclusion restriction is where IV papers are won or lost.

#### Classic Instruments in Economics and Health Economics

| Instrument | Endogenous X | Outcome Y | Reference |
|---|---|---|---|
| Quarter of birth | Years of schooling | Earnings | Angrist & Krueger (1991) |
| Distance to college | Years of schooling | Earnings | Card (1993) |
| Vietnam draft lottery number | Military service | Earnings | Angrist (1990) |
| Judge leniency | Incarceration | Recidivism | Kling (2006) |
| Provider preference / practice style | Treatment received | Health outcomes | Brookhart et al. (2006) |
| Geographic variation in treatment intensity | Treatment received | Health outcomes | Various |

**Quarter of birth (Angrist & Krueger 1991).**
- *Relevance*: Compulsory schooling laws require attendance until a specific age. Students born earlier in the year reach the minimum dropout age at a lower grade, so quarter of birth predicts total years of schooling.
- *Exclusion*: Quarter of birth should not directly affect earnings except through education. But this has been contested: birth season correlates with family background, maternal health, and school entry age effects. Bound, Jaeger & Baker (1995) showed that the instruments are extremely weak, making inference unreliable.

**Distance to college (Card 1993).**
- *Relevance*: Growing up near a college reduces the cost of attendance, increasing the probability of enrollment.
- *Exclusion*: Distance to college should not directly affect earnings. But colleges are located in urban, economically dynamic areas — so proximity to a college may proxy for labor market opportunities. The exclusion restriction requires controlling for urbanicity and local economic conditions.

**Vietnam draft lottery (Angrist 1990).**
- *Relevance*: Men with low lottery numbers were far more likely to serve in the military.
- *Exclusion*: The lottery number was randomly assigned, so it is uncorrelated with ability, family background, or other determinants of earnings. This is one of the cleanest instruments in the literature, precisely because the randomization was literal.

**Judge leniency (Kling 2006; many subsequent papers).**
- *Relevance*: Judges vary in their sentencing severity. Defendants are (quasi-)randomly assigned to judges, so judge identity predicts incarceration.
- *Exclusion*: A judge's general leniency should not affect a defendant's future outcomes except through the sentence. Potential violation: if lenient judges also differ in how they handle probation conditions, or if judge assignment is not truly random (e.g., specialized courts).

**Provider preference / practice style (Brookhart et al. 2006).**
- *Relevance*: Physicians vary in their propensity to prescribe specific treatments (e.g., COX-2 inhibitors vs. traditional NSAIDs). A patient's assigned physician's historical prescribing rate predicts the treatment the patient receives.
- *Exclusion*: The physician's general preference should not directly affect the patient's outcome, conditional on the treatment received. Potential violation: physicians who prefer newer drugs may also be more up-to-date clinically, or may practice in settings with different patient populations.

**Geographic variation instruments.**
- *Relevance*: Distance to a specialty hospital, regional supply of specialists, or local treatment norms predict whether a patient receives a procedure.
- *Exclusion*: Geographic distance should not directly affect health outcomes. But distance correlates with rurality, income, access to other services, and travel burden — all of which could independently affect outcomes. Careful control for geography-related confounders is essential.

### 3) The IV Estimand — Wald Estimator and 2SLS

#### The Simple Wald Estimator

With a single binary instrument and no controls, the IV estimator reduces to the **Wald estimator**:

$$
\hat{\beta}_{IV} = \frac{\mathrm{Cov}(Z, Y)}{\mathrm{Cov}(Z, X)} = \frac{\bar{Y}_{Z=1} - \bar{Y}_{Z=0}}{\bar{X}_{Z=1} - \bar{X}_{Z=0}}
$$

**Intuition.** The numerator is the "reduced form" — how much does the outcome $Y$ change when the instrument $Z$ changes? The denominator is the "first stage" — how much does the endogenous regressor $X$ change when $Z$ changes? The ratio asks:

> "Per unit of instrument-induced change in $X$, how much does $Y$ change?"

This is the core logic of all IV estimation: isolate the variation in $X$ that is driven by $Z$ (and therefore exogenous), and use only that variation to estimate the effect on $Y$.

#### From the Wald Estimator to 2SLS

When you have controls $W$ or continuous instruments, you generalize to **Two-Stage Least Squares**:

**Stage 1 (first stage):** Regress $X$ on $Z$ and $W$:
$$
X_i = \pi Z_i + W_i'\rho + v_i
$$
Obtain the predicted values $\hat{X}_i = \hat{\pi} Z_i + W_i'\hat{\rho}$.

**Stage 2 (structural equation):** Regress $Y$ on $\hat{X}$ and $W$:
$$
Y_i = \beta \hat{X}_i + W_i'\delta + \text{error}_i
$$

The coefficient $\hat{\beta}_{2SLS}$ on $\hat{X}$ is the IV estimate.

In matrix notation:

$$
\hat{\beta}_{2SLS} = (X'P_Z X)^{-1} X'P_Z Y
$$

where $P_Z = \tilde{Z}(\tilde{Z}'\tilde{Z})^{-1}\tilde{Z}'$ is the projection matrix onto the column space of $\tilde{Z} = [Z, W]$ (instruments and exogenous controls).

**What each term means:**
- $P_Z X$ extracts the part of $X$ that is linearly predicted by the instruments and controls.
- 2SLS replaces the endogenous variation in $X$ with the exogenous, instrument-driven variation.
- The formula is equivalent to OLS applied to the predicted values, but with a critical SE correction.

#### Why You Cannot Just Do Two Manual OLS Regressions

A common mistake (the "forbidden regression") is to:
1. Run OLS of $X$ on $Z$ and $W$, save $\hat{X}$.
2. Run OLS of $Y$ on $\hat{X}$ and $W$, read off standard errors.

The coefficient from step 2 is numerically identical to the proper 2SLS estimate. But the **standard errors are wrong** — they are computed using the variance of $\hat{X}$ instead of the variance of $X$, which understates the true sampling variability. Proper 2SLS software (e.g., `linearmodels.iv.IV2SLS`) adjusts the variance-covariance matrix to account for the generated regressor. Always use a proper 2SLS implementation.

### 4) LATE — Why IV Estimates a "Local" Effect

Under heterogeneous treatment effects, the IV estimand is the **Local Average Treatment Effect (LATE)**: the causal effect for "compliers."

#### The Four Subpopulations

When the instrument is binary, individuals fall into four groups based on their potential treatment status under each value of $Z$:

| Type | $X$ when $Z=0$ | $X$ when $Z=1$ | Description |
|---|---|---|---|
| **Compliers** | 0 | 1 | Treatment changes with $Z$ |
| **Always-takers** | 1 | 1 | Always treated regardless of $Z$ |
| **Never-takers** | 0 | 0 | Never treated regardless of $Z$ |
| **Defiers** | 1 | 0 | Do the opposite of what $Z$ encourages |

The IV estimate applies to **compliers only** — the people whose treatment status is actually moved by the instrument.

#### Concrete Health Economics Example

Suppose the instrument is **distance to a cardiac catheterization facility** and the treatment is **receiving catheterization** for heart attack patients.

- **Compliers**: Patients who receive catheterization when they live close to a facility but would not if they lived far away. These are patients on the margin — their treatment decision is sensitive to access.
- **Always-takers**: Patients who would get catheterized regardless of distance (e.g., severe cases where any physician would refer).
- **Never-takers**: Patients who would not get catheterized regardless of distance (e.g., very frail patients with contraindications).

The IV estimate tells you the effect of catheterization **for the complier subpopulation** — the marginal patients whose treatment is shifted by distance. It does not directly tell you the effect for always-takers or never-takers.

#### The Monotonicity Assumption

The LATE framework requires **monotonicity**: the instrument must push treatment in the same direction for everyone (or at least not push some people in the opposite direction). Formally:

$$
X_i(Z=1) \geq X_i(Z=0) \quad \text{for all } i \quad (\text{no defiers})
$$

If monotonicity fails (some people defy the instrument), the LATE interpretation breaks down. In the distance example, monotonicity means that no patient who would be catheterized when far from a facility would refuse catheterization when close. This is generally plausible for distance instruments.

#### Why LATE May Differ from ATE

- The complier population is defined by the instrument. Different instruments define different complier groups, which may have different treatment effects.
- If you change the instrument (e.g., from distance to physician preference), the LATE may change because the complier group changes.
- The complier population is often not directly observable — you cannot identify specific individuals as compliers in the data.
- **Policy relevance**: If a policy intervention would affect the same margin as the instrument, the LATE is directly relevant. If the policy would affect a different population, the LATE may not generalize.

For health economics, this matters enormously. A study using physician preference as an instrument identifies the effect for patients whose treatment depends on their physician's style. A study using insurance eligibility as an instrument identifies the effect for people whose insurance status depends on eligibility rules. These may be very different populations with very different treatment effects.

### 5) Weak Instruments — Why They Are Dangerous and What To Do

#### The Problem

When the first stage is weak ($\pi \approx 0$), the 2SLS estimator is:

1. **Biased toward OLS** in finite samples. The whole point of IV is to remove OLS bias, but a weak instrument brings the bias back.
2. **Unreliable inference.** Confidence intervals have incorrect coverage. The t-statistic does not follow the usual distribution. Hypothesis tests reject far too often (or too rarely).
3. **Noisy and unstable.** Small changes in the sample or specification produce wildly different estimates.

The intuition is simple: if $Z$ barely moves $X$, you are dividing the reduced form by a near-zero number, amplifying noise.

#### Diagnosis: Stock-Yogo Critical Values

Stock and Yogo (2005) provide critical values for the first-stage F-statistic under different criteria for acceptable bias. The most commonly cited:

- **F > 10**: The 2SLS bias is no more than 10% of the OLS bias (for the case of one endogenous regressor).

For the just-identified case (one instrument for one endogenous variable), the effective F-statistic is simply the squared t-statistic on the excluded instrument. For multiple instruments, the effective F-statistic from the Cragg-Donald or Kleibergen-Paap test is the relevant quantity.

| Number of instruments | Critical F (10% maximal bias) | Critical F (15% maximal bias) |
|---|---|---|
| 1 | 16.38 | 8.96 |
| 2 | 19.93 | 11.59 |
| 3 | 22.30 | 12.83 |

(Source: Stock & Yogo 2005, Table 5.2.)

The "F > 10" heuristic is a rough approximation. The exact critical value depends on the number of instruments, the acceptable bias level, and whether you use the bias or size criterion.

#### What To Do If Instruments Are Weak

1. **Anderson-Rubin (AR) confidence sets.** The AR test is robust to weak instruments: it has correct size regardless of instrument strength. If the AR confidence set is wide, you have a weak instrument problem. If it does not contain zero but your Wald-based confidence interval does, weak instruments may be distorting your inference.

2. **Report the first-stage F and let the reader judge.** Transparency is essential. Always report the first-stage regression, the F-statistic, and the Stock-Yogo critical values. Let readers assess whether the instrument is strong enough.

3. **Find better instruments.** Ultimately, the best solution to weak instruments is stronger instruments. This requires deeper institutional knowledge and more creative thinking about sources of exogenous variation.

4. **Reduced form as a bound.** If the first stage is weak, you can still report the reduced form (the regression of $Y$ on $Z$). This is an intention-to-treat (ITT) effect that is valid regardless of instrument strength. The structural effect $\beta$ is the reduced form divided by the first stage, so the reduced form gives a lower bound on $|\beta|$ if the first stage is less than one.

### 6) Overidentification — Multiple Instruments

#### Just-Identified vs. Overidentified

- **Just-identified**: one instrument for one endogenous regressor. The 2SLS estimate is unique and equals the Wald / IV estimator.
- **Overidentified**: more instruments than endogenous regressors. Different instruments may imply different estimates. 2SLS optimally combines them.

#### Hansen's J Test

When overidentified, you can test whether all instruments give consistent estimates of $\beta$. The null hypothesis is that all instruments are valid (satisfy the exclusion restriction). The test statistic:

$$
J = n \cdot \hat{u}'P_Z\hat{u} / \hat{u}'\hat{u} \sim \chi^2(m - k)
$$

where $m$ is the number of instruments, $k$ is the number of endogenous regressors, and $\hat{u}$ are the 2SLS residuals.

- If J is large (p-value < 0.05), reject the null: at least one instrument is invalid.
- If J is small, you cannot reject consistency — but this does not prove all instruments are valid. The test has limited power, and if all instruments are invalid in the same direction, J will not detect the problem.

#### The Tradeoff

More instruments provide:
- **More power** (tighter first stage, smaller standard errors).
- **More efficiency** (optimal weighting across instruments).

But also:
- **More chances for exclusion violations** (each instrument must independently satisfy exclusion).
- **Bias toward OLS** in finite samples with many weak instruments (the "many instruments" problem).

**Practical guidance**: Be suspicious of studies with many instruments, especially if the instruments are not individually strong or do not have independent economic justifications. In health economics, using dozens of geographic or temporal dummies as instruments often signals a weak overall identification strategy.

### 7) Practical Code Template

#### Complete 2SLS Estimation Using linearmodels

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# --- Simulate data ---
np.random.seed(123)
n = 5000

# Confounder (unobserved patient severity)
severity = np.random.normal(0, 1, n)

# Instrument: physician prescribing preference (exogenous to patient outcome)
physician_pref = np.random.normal(0, 1, n)

# Observed control: patient age
age = np.random.normal(60, 10, n)

# Endogenous treatment: drug prescribed (depends on preference AND severity)
treatment = (0.6 * physician_pref
             + 0.5 * severity
             + 0.02 * age
             + np.random.normal(0, 0.8, n))

# Outcome: health improvement (true treatment effect = 1.5)
TRUE_BETA = 1.5
health = (TRUE_BETA * treatment
          - 0.8 * severity
          + 0.01 * age
          + np.random.normal(0, 1, n))

df = pd.DataFrame({
    "health": health,
    "treatment": treatment,
    "physician_pref": physician_pref,
    "age": age,
})

# --- OLS (biased) ---
ols = sm.OLS(
    df["health"],
    sm.add_constant(df[["treatment", "age"]])
).fit(cov_type="HC1")
print("=== OLS (biased) ===")
print(f"  treatment coeff: {ols.params['treatment']:.4f}  (true = {TRUE_BETA})")
print(f"  OLS is biased because severity confounds treatment and health.\n")

# --- First stage ---
first_stage = sm.OLS(
    df["treatment"],
    sm.add_constant(df[["physician_pref", "age"]])
).fit(cov_type="HC1")
print("=== First Stage: treatment ~ physician_pref + age ===")
print(f"  physician_pref coeff: {first_stage.params['physician_pref']:.4f}")
f_stat = first_stage.f_test("physician_pref = 0").fvalue[0][0]
print(f"  First-stage F-statistic: {f_stat:.1f}")
print(f"  Strong instrument? {'Yes' if f_stat > 10 else 'NO — weak instrument!'}\n")

# --- 2SLS ---
iv_model = IV2SLS.from_formula(
    "health ~ 1 + age + [treatment ~ physician_pref]", data=df
)
iv_result = iv_model.fit(cov_type="robust")
print("=== 2SLS ===")
print(f"  treatment coeff: {iv_result.params['treatment']:.4f}  (true = {TRUE_BETA})")
print(f"  95% CI: [{iv_result.conf_int().loc['treatment', 'lower']:.4f}, "
      f"{iv_result.conf_int().loc['treatment', 'upper']:.4f}]")
print(f"  2SLS is consistent and close to the true effect.\n")

# --- Reduced form (Y on Z) ---
reduced_form = sm.OLS(
    df["health"],
    sm.add_constant(df[["physician_pref", "age"]])
).fit(cov_type="HC1")
print("=== Reduced Form: health ~ physician_pref + age ===")
print(f"  physician_pref coeff: {reduced_form.params['physician_pref']:.4f}")
implied_beta = reduced_form.params["physician_pref"] / first_stage.params["physician_pref"]
print(f"  Implied structural beta (reduced form / first stage): {implied_beta:.4f}")
```

#### Using the Project Helper

```python
from src.causal import fit_iv_2sls

result = fit_iv_2sls(
    df,
    y_col="health",
    x_endog=["treatment"],
    x_exog=["age"],
    z_cols=["physician_pref"],
    cov_type="robust",
)
print(result.summary)
```

### 8) Health Economics IV Examples — Detailed

#### Example A: Provider Preference Instruments

**Setting.** You want to estimate the effect of receiving Drug A (vs. Drug B) on patient outcomes (e.g., cardiovascular events).

**Endogeneity problem.** Physicians choose Drug A for patients they believe will benefit, based on clinical judgment that incorporates unobserved patient characteristics. Sicker or more complex patients may systematically receive one drug over the other.

**Instrument.** The prescribing physician's historical preference for Drug A (measured as the physician's Drug A prescribing rate for *other* patients, excluding the focal patient — a "leave-one-out" measure).

**Relevance argument.** Physicians vary substantially and persistently in their prescribing preferences, often driven by training, habit, and exposure to drug representatives. A physician who prescribes Drug A to 80% of similar patients is much more likely to prescribe it to the focal patient than a physician who prescribes it to 20%.

**Exclusion argument.** The physician's general prescribing tendency for other patients should not directly affect the focal patient's cardiovascular outcomes — it operates only through the treatment received. Conditional on patient characteristics, the physician's practice style for other patients is plausibly exogenous.

**Potential violations.** (1) Physicians who prefer Drug A may also differ in other aspects of care quality (monitoring, follow-up). (2) Patient sorting to physicians may not be random — sicker patients may seek out certain physicians. (3) The leave-one-out measure can be contaminated if the physician's panel is small.

#### Example B: Geographic Variation Instruments

**Setting.** You want to estimate the effect of receiving a surgical procedure (e.g., CABG vs. medical management) on survival.

**Endogeneity problem.** Patients who receive surgery are selected based on clinical severity, anatomy, and physician judgment. Observed and unobserved patient characteristics simultaneously determine treatment and outcomes.

**Instrument.** The procedure rate in the patient's hospital referral region (HRR) — a measure of local treatment intensity that varies due to practice norms, capacity, and supply-side factors.

**Relevance argument.** There is well-documented, persistent geographic variation in procedure rates that is not explained by patient characteristics or disease severity (the "Dartmouth Atlas" literature). Living in a high-intensity region substantially increases the probability of receiving the procedure.

**Exclusion argument.** Regional treatment intensity should not directly affect the patient's survival except through the procedure received. The instrument operates through local practice norms and supply, not through patient health.

**Potential violations.** (1) Regional variation may correlate with other aspects of healthcare quality (ICU availability, nursing ratios). (2) Patient migration: healthier or wealthier patients may move to regions with better care. (3) Regional characteristics (poverty, pollution, diet) may independently affect both procedure rates and outcomes.

#### Example C: Policy Instruments (Medicaid Eligibility)

**Setting.** You want to estimate the effect of Medicaid coverage on health outcomes (e.g., infant mortality, emergency department utilization).

**Endogeneity problem.** Medicaid enrollment is not random — people enroll because they are poor, pregnant, disabled, or sick. These characteristics independently affect health outcomes.

**Instrument.** State and federal Medicaid eligibility expansions that shift the income threshold for coverage. Some states expanded eligibility at different times, creating variation in who is eligible that is plausibly independent of individual health.

**Relevance argument.** Eligibility expansions mechanically increase insurance coverage among the newly eligible population. The first stage is typically strong because eligibility directly determines the option to enroll.

**Exclusion argument.** The eligibility rule change affects health outcomes only by changing insurance status. The specific income cutoff at which a state sets eligibility should not directly affect health outcomes except through insurance.

**Potential violations.** (1) Eligibility expansions may coincide with other policy changes (e.g., funding increases for community health centers). (2) Individuals near the eligibility threshold may adjust their reported income or labor supply in response to the expansion, creating selection. (3) The expansion may affect healthcare markets (provider supply, wait times) in ways that are separate from individual coverage.

### 9) Diagnostics — Expanded

#### First-Stage F-Statistic

Always report the first-stage regression and the F-statistic on the excluded instruments.

```python
# First-stage F-statistic
first_stage = sm.OLS(df["X"], sm.add_constant(df[["Z", "W"]])).fit(cov_type="HC1")
f_test = first_stage.f_test("Z = 0")
print(f"First-stage F: {f_test.fvalue[0][0]:.2f}")
print(f"p-value: {f_test.pvalue:.4f}")
```

Interpretation:
- F > 10: Instrument is likely strong enough for reliable 2SLS inference.
- F between 5 and 10: Caution. Report Anderson-Rubin confidence sets as a robustness check.
- F < 5: Serious weak instrument concern. 2SLS may be more biased than OLS.

#### First-Stage Scatter Plot

A visual diagnostic: plot $X$ vs. $Z$ (residualized on controls if applicable). You should see a clear relationship. If the scatter is a shapeless cloud, the instrument is weak.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["Z"], df["X"], alpha=0.1, s=5)
ax.set_xlabel("Instrument (Z)")
ax.set_ylabel("Endogenous Regressor (X)")
ax.set_title("First-Stage Relationship: X vs Z")
# Add regression line
z_grid = np.linspace(df["Z"].min(), df["Z"].max(), 100)
ax.plot(z_grid, first_stage.params["const"] + first_stage.params["Z"] * z_grid,
        color="red", linewidth=2, label=f"slope = {first_stage.params['Z']:.3f}")
ax.legend()
plt.tight_layout()
plt.show()
```

#### Exclusion Restriction — Argued, Not Tested

The exclusion restriction ($\mathrm{Cov}(Z, u) = 0$) cannot be tested with data alone. You must argue it using:

1. **Institutional knowledge.** Explain *why* the instrument has no direct effect on $Y$. What is the causal mechanism, and why does it run exclusively through $X$?
2. **Falsification tests.** If $Z$ should not affect $Y$ in a subpopulation where $X$ is constant, test this. For example, if distance to hospital is the instrument and the treatment is surgery, check whether distance predicts outcomes for patients who are never candidates for surgery (never-takers). A significant effect in this group suggests a direct path from $Z$ to $Y$.
3. **Balance checks.** If $Z$ is as-good-as-random, it should be uncorrelated with observed predetermined characteristics. Test this by regressing baseline covariates on $Z$.

#### Reduced Form

The reduced form regression ($Y$ on $Z$ and $W$) is informative:

- If the reduced form coefficient on $Z$ is significantly different from zero, the instrument predicts $Y$. This is necessary (but not sufficient) for IV relevance.
- The 2SLS estimate equals the reduced form coefficient divided by the first-stage coefficient: $\hat{\beta}_{2SLS} = \hat{\pi}_{RF} / \hat{\pi}_{FS}$.
- Check that the sign and magnitude are consistent across the first stage, reduced form, and 2SLS. Inconsistency suggests a problem.

#### Overidentification Test (When Applicable)

If you have more instruments than endogenous regressors:

```python
# linearmodels reports the Sargan/Hansen J test automatically
iv_result = IV2SLS.from_formula(
    "Y ~ 1 + W + [X ~ Z1 + Z2]", data=df
).fit(cov_type="robust")
print(iv_result.summary)
# Look for the "Overidentification test" (Sargan or Hansen J) in the output.
```

A significant J statistic (p < 0.05) rejects the null that all instruments are valid. A non-significant J statistic does not prove validity — the test has limited power.

#### Sensitivity to Instrument Choice

If you have multiple candidate instruments, estimate the model using each instrument separately and compare the results. If different instruments give very different estimates, at least one instrument likely violates the exclusion restriction — or the LATE differs across complier populations.

### 10) Common Mistakes

1. **Using weak instruments and not checking first-stage strength.** Always report the first-stage F-statistic. If F < 10, your 2SLS estimates may be more biased than OLS.

2. **Claiming exclusion is "obvious" without careful argument.** The exclusion restriction is the most important and most contestable assumption in IV. It requires detailed institutional reasoning, not a one-sentence assertion. Every serious IV paper devotes paragraphs to defending exclusion.

3. **Interpreting 2SLS as ATE when it is LATE.** Under heterogeneous treatment effects, 2SLS estimates a Local Average Treatment Effect for compliers. If you write "the effect of treatment is..." without qualifying the population, you are overstating the generalizability of your result.

4. **Using too many instruments.** More instruments can improve efficiency but also increase finite-sample bias and multiply the chances of exclusion violations. The "many instruments" problem is well-documented. Prefer fewer, individually strong instruments with clear economic justification.

5. **Not reporting the first stage.** The first stage is not a nuisance — it is the foundation of your identification strategy. Always report the first-stage regression, the coefficient on the excluded instrument, and the F-statistic.

6. **Forgetting to cluster standard errors.** If the instrument or treatment varies at a group level (e.g., regional policy instruments, physician-level prescribing), standard errors must be clustered at that level. Failing to cluster produces confidence intervals that are too narrow.

7. **Using manually computed X-hat instead of proper 2SLS.** As noted above, running two separate OLS regressions and plugging $\hat{X}$ into the second stage gives the correct point estimate but incorrect standard errors. Always use a proper 2SLS implementation that adjusts the variance-covariance matrix.

8. **Ignoring the reduced form.** The reduced form ($Y$ on $Z$) is a useful sanity check. If the reduced form coefficient is insignificant, your instrument does not predict the outcome, which means either the first stage is weak or the true effect is zero.

9. **Confusing "instrument validity" with "instrument strength."** A strong first stage (high F) does not imply a valid instrument. The instrument can be strong and still violate exclusion. Relevance is testable; exclusion is not.

10. **Failing to consider the complier population.** When using IV for policy evaluation, ask: who are the compliers? If the policy would affect a different population than the complier group, the LATE may not be externally valid for the policy question.

### Project Code Map
- `src/causal.py`: panel + IV helpers (`to_panel_index`, `fit_twfe_panel_ols`, `fit_iv_2sls`)
- `scripts/build_datasets.py`: ACS panel builder (writes data/processed/census_county_panel.csv)
- `src/census_api.py`: Census/ACS client (`fetch_acs`)
- `configs/census_panel.yaml`: panel config (years + variables)
- `data/sample/census_county_panel_sample.csv`: offline panel dataset
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

### Exercises

- [ ] Write down one concrete endogeneity story (omitted variable, reverse causality, or measurement error) for a health economics regression you care about.
- [ ] Propose a plausible instrument. Write the relevance argument and the exclusion argument in 3-4 sentences each. Then list 2 specific threats to exclusion.
- [ ] Compute OLS and 2SLS on the simulated dataset where you control the DGP. Confirm that OLS is biased and 2SLS recovers the true effect.
- [ ] Inspect the first stage. Report the F-statistic and write 3 sentences explaining whether you believe the instrument is strong enough.
- [ ] Run the reduced form regression ($Y$ on $Z$). Verify that the ratio of the reduced form coefficient to the first-stage coefficient equals the 2SLS estimate.
- [ ] Identify the "complier" population for your proposed instrument. Describe in plain language who these people are and whether the LATE for this group is policy-relevant.
- [ ] Describe one falsification test you could run to probe the exclusion restriction, and what a "failure" would look like.

<a id="summary"></a>
## Summary + Suggested Readings

**What you should take away.** IV/2SLS is a powerful identification strategy for recovering causal effects when the treatment variable is endogenous. Its credibility rests on two conditions: the instrument must be relevant (strong first stage) and satisfy the exclusion restriction (no direct effect on the outcome). Relevance is testable; exclusion is not. Under heterogeneous treatment effects, IV estimates a Local Average Treatment Effect for compliers, which may differ from the average treatment effect. The first stage is not a nuisance — it is the foundation of the identification strategy and must always be reported and defended.

### Core References

- **Angrist & Pischke (2009).** *Mostly Harmless Econometrics*, Chapter 4: Instrumental Variables in Action. The standard graduate-level treatment of IV, LATE, and weak instruments.
- **Angrist, Imbens & Rubin (1996).** "Identification of Causal Effects Using Instrumental Variables." *JASA*. The foundational paper on LATE and the complier framework.
- **Stock & Yogo (2005).** "Testing for Weak Instruments in Linear IV Regression." In *Identification and Inference for Econometric Models*. Critical values for weak instrument diagnostics.
- **Staiger & Stock (1997).** "Instrumental Variables Regression with Weak Instruments." *Econometrica*. Established the F > 10 rule of thumb and characterized the finite-sample behavior of 2SLS with weak instruments.
- **Wooldridge (2010).** *Econometric Analysis of Cross Section and Panel Data*, Chapters 5-6. Rigorous treatment of IV mechanics and asymptotic theory.
- **Bound, Jaeger & Baker (1995).** "Problems with Instrumental Variables Estimation When the Correlation Between the Instruments and the Endogenous Explanatory Variable is Weak." *JASA*. A cautionary tale about weak instruments, using the quarter-of-birth example.

### Health Economics IV References

- **Brookhart et al. (2006).** "Evaluating Short-Term Drug Effects Using a Physician-Specific Prescribing Preference as an Instrumental Variable." *Epidemiology*. The key reference for provider preference instruments.
- **McClellan, McNeil & Newhouse (1994).** "Does More Intensive Treatment of Acute Myocardial Infarction in the Elderly Reduce Mortality?" *JAMA*. Uses distance to catheterization-capable hospitals as an instrument for treatment intensity. One of the landmark IV papers in health economics.
- **Newhouse & McClellan (1998).** "Econometrics in Outcomes Research: The Use of Instrumental Variables." *Annual Review of Public Health*. Accessible overview of IV in health services research.
- **Currie & Gruber (1996a, 1996b).** "Saving Babies" and "Health Insurance Eligibility." Use Medicaid eligibility expansions as instruments for insurance coverage. Classic policy instrument papers in health economics.
- **Angrist (1990).** "Lifetime Earnings and the Vietnam Era Draft Lottery." *AER*. The cleanest IV example in economics, using literal random assignment.
