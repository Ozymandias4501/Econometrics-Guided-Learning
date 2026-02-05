### Event Study (Leads/Lags) and Pre-Trends Diagnostics

An event study expands DiD by estimating dynamic effects around treatment adoption.

You create indicators for event time:
- lead indicators (pre-treatment)
- lag indicators (post-treatment)

Then estimate:

$$
Y_{it} = \\sum_{k \\ne -1} \\beta_k \\cdot 1[t - T_i = k] + \\alpha_i + \\gamma_t + \\varepsilon_{it}
$$

where $T_i$ is the adoption time for unit $i$ (if it ever adopts).

#### Why leads matter
- Lead coefficients ($k < 0$) test for **pre-trends**.
- If you see strong pre-trends, parallel trends is doubtful.

#### What to do in practice
- Plot $\\beta_k$ with confidence intervals.
- Run placebo tests (fake treatment dates) as a falsification.
- Clearly state: “This is evidence consistent/inconsistent with parallel trends,” not proof.

