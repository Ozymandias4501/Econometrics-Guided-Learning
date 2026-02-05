## Primer: Classical time-series econometrics with statsmodels (ADF/KPSS, VAR)

This repo already uses time-aware evaluation for ML.
This primer introduces the “classical” time-series econometrics toolkit in `statsmodels`.

### Stationarity and unit roots (ADF / KPSS)
Two common tests:
- **ADF**: null = unit root (nonstationary)
- **KPSS**: null = stationary

```python
from statsmodels.tsa.stattools import adfuller, kpss

# x is a 1D array-like (no missing)
# adf_stat, adf_p, *_ = adfuller(x)
# kpss_stat, kpss_p, *_ = kpss(x, regression='c', nlags='auto')
```

Interpretation habit:
- If ADF p-value is small → evidence against unit root.
- If KPSS p-value is small → evidence against stationarity.

### VAR: multivariate autoregression
VAR models multiple series together:
```python
from statsmodels.tsa.api import VAR

# df: DataFrame of stationary-ish series with a DatetimeIndex
# model = VAR(df)
# res = model.fit(maxlags=8, ic='aic')  # or choose lags manually
# print(res.summary())
```

Useful tools:
```python
# res.test_causality('y', ['x1', 'x2'])      # Granger causality tests
# irf = res.irf(12)                         # impulse responses to 12 steps
# irf.plot(orth=True)                       # orthogonalized (ordering matters)
```

### Practical cautions
- Nonstationary series can create **spurious regression** results.
- IRFs depend on identification choices (e.g., Cholesky ordering).
- Macro series are revised and can have structural breaks; treat results as conditional and fragile.

