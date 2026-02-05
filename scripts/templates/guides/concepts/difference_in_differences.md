### Difference-in-Differences (DiD)

DiD is a design for estimating treatment effects when some units become treated and others do not.

#### Canonical 2×2 setup
- Two groups: treated vs control
- Two periods: pre vs post

The DiD estimand is:

$$
(\\bar{Y}^{T}_{post} - \\bar{Y}^{T}_{pre}) - (\\bar{Y}^{C}_{post} - \\bar{Y}^{C}_{pre})
$$

#### Key assumption: parallel trends
> **Definition:** **Parallel trends** means: in the absence of treatment, treated and control units would have changed similarly over time.

This is not guaranteed by regression output; it is a design assumption you must defend with context and diagnostics.

#### TWFE implementation
With more periods, DiD is often implemented as:

$$
Y_{it} = \\beta \\cdot D_{it} + \\alpha_i + \\gamma_t + \\varepsilon_{it}
$$

$\\beta$ is interpreted as the average treatment effect under assumptions.

