### Panel Fixed Effects (FE): “Compare a unit to itself over time”

Fixed effects are one of the most common ways to reduce bias from **time-invariant unobservables** in panel data.

#### 1) Intuition (plain English)

Panels are powerful because they let you control for factors that differ across units but are roughly constant over time:
- geography,
- long-run institutions,
- baseline demographics,
- “culture” or other hard-to-measure features.

**Story example:** Counties differ in baseline poverty for many persistent reasons.  
If you want to study how changes in unemployment relate to changes in poverty, it is more credible to compare:
- *the same county in different years*  
than to compare two different counties once.

#### 2) Notation + setup (define symbols)

Let:
- $i = 1,\\dots,N$ index entities (counties),
- $t = 1,\\dots,T$ index time (years),
- $Y_{it}$ be the outcome,
- $X_{it}$ be a $K \\times 1$ vector of regressors,
- $\\alpha_i$ be an entity-specific intercept (unit FE),
- $\\gamma_t$ be a time-specific intercept (time FE),
- $\\varepsilon_{it}$ be the remaining error.

The two-way fixed effects (TWFE) model is:

$$
Y_{it} = X_{it}'\\beta + \\alpha_i + \\gamma_t + \\varepsilon_{it}.
$$

**What each term means**
- $X_{it}'\\beta$: the part explained by observed covariates.
- $\\alpha_i$: all time-invariant determinants of $Y$ for entity $i$ (observed or unobserved).
- $\\gamma_t$: shocks common to all entities in time $t$ (recessions, nationwide policy, measurement changes).
- $\\varepsilon_{it}$: idiosyncratic shocks not captured above.

Matrix form (stack observations by time within entity):

$$
\\mathbf{y} = \\mathbf{X}\\beta + \\mathbf{A}\\alpha + \\mathbf{G}\\gamma + \\varepsilon
$$

where:
- $\\mathbf{A}$ is the entity-dummy design matrix,
- $\\mathbf{G}$ is the time-dummy design matrix.

#### 3) What FE “assumes” (identification conditions)

FE is not magic; it removes *time-invariant* confounding, not everything.

A common identification condition is **strict exogeneity**:

$$
\\mathbb{E}[\\varepsilon_{it} \\mid X_{i1}, \\dots, X_{iT}, \\alpha_i] = 0
\\quad \\text{for all } t.
$$

**What this means**
- After controlling for unit FE, time FE, and the full history of $X$, the remaining shocks are mean-zero.
- If today’s shock changes future $X$ (feedback), strict exogeneity can fail.

Other requirements:
- **Within-unit variation:** the regressor must change over time within entities.
- **No perfect multicollinearity:** you cannot include variables fully determined by FE (e.g., a pure time trend plus full time FE can collide depending on spec).

<details>
<summary>Optional: dynamic panels (why lagged Y can be tricky)</summary>

If you include $Y_{i,t-1}$ as a regressor, FE can create “Nickell bias” when $T$ is small.
This repo does not focus on dynamic panel estimators (Arellano–Bond), but you should know this pitfall exists.

</details>

#### 4) Estimation mechanics: the “within” transformation (derivation)

The FE estimator can be understood as OLS after removing means.

**Entity FE only (simpler to see):**

Start with:
$$
Y_{it} = X_{it}'\\beta + \\alpha_i + \\varepsilon_{it}.
$$

Take the time average within each entity:
$$
\\bar{Y}_i = \\bar{X}_i'\\beta + \\alpha_i + \\bar{\\varepsilon}_i.
$$

Subtract the entity mean equation from the original:

$$
Y_{it} - \\bar{Y}_i = (X_{it} - \\bar{X}_i)'\\beta + (\\varepsilon_{it} - \\bar{\\varepsilon}_i).
$$

**What this achieves**
- The term $\\alpha_i$ disappears (it is constant within entity).
- Identification comes from deviations from the entity’s own average.

TWFE (“two-way”) uses a version of demeaning that removes both:
- entity averages, and
- time averages,
often implemented by “absorbing” FE rather than explicitly constructing every dummy.

#### 5) Frisch–Waugh–Lovell (FWL) interpretation (how software implements FE)

FWL says: if you want $\\beta$ in a regression with controls, you can:
1) residualize $Y$ on the controls,
2) residualize $X$ on the controls,
3) regress residualized $Y$ on residualized $X$.

In FE models, “controls” include thousands of entity/time dummies. Software (like `linearmodels`) uses the same logic efficiently.

#### 6) Inference: why clustering is common in panels

Panel errors are often correlated:
- within entity over time (serial correlation),
- within higher-level groups (e.g., state shocks),
- within time (common shocks).

Robust (HC) SE handle heteroskedasticity but not arbitrary within-cluster correlation.
That’s why you often see **clustered SE** in applied FE work.

#### 7) Diagnostics + robustness (at least 3)

1) **Pooled vs FE comparison**
- Fit pooled OLS and FE; if coefficients move a lot, unobserved heterogeneity likely mattered.

2) **Within-variation check**
- For each regressor, confirm it varies within entity. If not, FE cannot estimate it.

3) **Sensitivity to time FE**
- Add/remove time FE; if results change materially, time shocks matter and must be modeled.

4) **Influential units**
- Check whether a handful of entities drive the estimate (outliers / leverage).

#### 8) Interpretation + reporting (what FE coefficients mean)

In a TWFE regression, interpret $\\beta_j$ as:

> the association between *within-entity changes* in $x_j$ and *within-entity changes* in $Y$, after removing common time shocks.

**What this does NOT mean**
- FE does **not** guarantee causality (time-varying omitted variables can remain).
- FE does **not** solve reverse causality (shocks can affect both $X$ and $Y$).
- FE does **not** “fix” measurement error (it can make some bias worse).

#### Exercises

- [ ] Take one regressor in the panel notebook and compute $x_{it} - \\bar{x}_i$ manually; confirm the mean is ~0 within each entity.
- [ ] Explain in words what variation identifies $\\beta$ in a TWFE model.
- [ ] Name one time-invariant variable you *wish* you could estimate and explain why FE absorbs it.
- [ ] Fit pooled OLS and TWFE; write 3 sentences explaining why the coefficients differ.
- [ ] Try adding time FE and report whether the coefficient on your main regressor is stable.
