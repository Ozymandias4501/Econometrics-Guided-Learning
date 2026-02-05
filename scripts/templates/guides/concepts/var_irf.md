### VAR and Impulse Responses (IRFs)

> **Definition:** A **VAR(p)** models each variable as a linear function of $p$ lags of all variables.

For a vector $y_t$:
$$
y_t = A_1 y_{t-1} + \\dots + A_p y_{t-p} + \\varepsilon_t
$$

#### When VARs are useful
- You care about **dynamics** and feedback between variables.
- You want to study how shocks propagate over time.

#### Key decisions
- **Transformations**: stationarity often requires differencing/log-differencing.
- **Lag length**: choose with information criteria (AIC/BIC) and sanity checks.

#### Granger causality (predictive, not causal)
> **Definition:** $x$ “Granger-causes” $y$ if past $x$ improves prediction of $y$ beyond past $y$ alone.

This is about forecasting information, not structural causality.

#### Impulse response functions
IRFs trace the effect of a one-time shock over time.
In practice, you often use orthogonalized shocks (Cholesky), which means:
- the **ordering matters**,
- and the IRF is conditional on that identification choice.

