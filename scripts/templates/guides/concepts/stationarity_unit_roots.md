### Stationarity and Unit Roots (ADF/KPSS)

> **Definition:** A time series is **stationary** if its statistical properties (mean/variance/autocovariance) are stable over time.

Many macro series in levels are not stationary (they trend).

#### Unit root intuition
A unit root process behaves like a random walk:
- shocks accumulate,
- the series does not “mean revert” in levels.

#### Why this matters: spurious regression
Regressing one trending series on another can produce:
- high $R^2$,
- significant t-stats,
even when there is no meaningful relationship.

#### Common tools
- **ADF test**: null = unit root (nonstationary)
- **KPSS test**: null = stationary

Practical habit:
- Plot the series.
- Try differences / growth rates.
- Use tests as supporting evidence, not as the only decision.

