### Panel Fixed Effects (Entity + Time)

> **Definition:** A **panel dataset** follows the same units over time (or repeated cross-sections at a unit level).

In this project, the “unit” is a county and time is year.

#### Two-way fixed effects (TWFE)
A common model is:

$$
Y_{it} = \\beta'X_{it} + \\alpha_i + \\gamma_t + \\varepsilon_{it}
$$

- $\\alpha_i$: **entity fixed effects** (time-invariant county differences)
- $\\gamma_t$: **time fixed effects** (common shocks in a given year)

Interpretation:
- $\\beta$ is identified from **within-county changes over time**, after subtracting common year shocks.

Key implications:
- Time-invariant regressors (e.g., a county’s latitude) are absorbed by $\\alpha_i$ and cannot be estimated.
- If treatment varies only at the time level, it is absorbed by $\\gamma_t$.

