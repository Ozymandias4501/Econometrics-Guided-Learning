### Cointegration and Error Correction Models (ECM)

Two series can be individually nonstationary but move together in the long run.

> **Definition:** $x_t$ and $y_t$ are **cointegrated** if some linear combination is stationary:
$$
y_t - \\beta x_t \\text{ is stationary}
$$

#### Engle–Granger (two-step) idea
1) Regress $y_t$ on $x_t$ in levels to estimate $\\hat\\beta$.
2) Test whether the residual $\\hat u_t = y_t - \\hat\\beta x_t$ is stationary.

#### Error correction model
An ECM links short-run changes to long-run deviations:

$$
\\Delta y_t = \\alpha( y_{t-1} - \\beta x_{t-1}) + \\Gamma \\Delta x_t + \\varepsilon_t
$$

Interpretation:
- $(y_{t-1} - \\beta x_{t-1})$ is the “error” from the long-run relationship.
- $\\alpha$ is the speed of adjustment back to equilibrium.

