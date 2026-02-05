## Primer: Classical time-series econometrics with `statsmodels` (ADF/KPSS, cointegration, VAR/IRF)

This repo already teaches time-aware evaluation for ML. This primer introduces the classical econometrics toolkit for time series:
- stationarity / unit roots,
- cointegration and error correction ideas,
- VARs and impulse responses.

Deep theory is in the guides; this primer focuses on “how to use the tools correctly.”

### Before you start: what you should always do

1) **Plot the series in levels** (look for trends, breaks).
2) **Choose transformations** (diff/logdiff) for stationarity.
3) **Drop missing values** before tests/models.

### Stationarity tests (ADF / KPSS)

Two common tests:
- **ADF**: null = unit root (nonstationary)
- **KPSS**: null = stationary

```python
from statsmodels.tsa.stattools import adfuller, kpss

x = df["SERIES"].dropna().to_numpy()

adf_stat, adf_p, *_ = adfuller(x)
kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")

print("ADF p:", adf_p, "KPSS p:", kpss_p)
```

**Expected output / sanity check**
- trending level series often: ADF p not small, KPSS p small
- differenced series often: ADF p small, KPSS p not small

### Cointegration (Engle–Granger test)

If two series are individually nonstationary but move together long-run, they may be cointegrated.

```python
from statsmodels.tsa.stattools import coint

y = df["Y"].dropna()
x = df["X"].dropna()

stat, p, _ = coint(y, x)
print("coint p:", p)
```

### VAR (vector autoregression)

VAR models multiple stationary-ish series jointly.

```python
from statsmodels.tsa.api import VAR

X = df[["UNRATE", "FEDFUNDS", "INDPRO"]].astype(float).dropna()
X = X.diff().dropna()  # common stationarity transform

model = VAR(X)
res = model.fit(maxlags=8, ic="aic")  # or choose lags manually
print("lags chosen:", res.k_ar)
print(res.summary())
```

**Expected output / sanity check**
- `res.k_ar` is the chosen lag length
- `res.is_stable(verbose=False)` should be True for a stable VAR

### Granger causality (predictive, not causal)

```python
res.test_causality("UNRATE", ["FEDFUNDS"]).summary()
```

Interpretation: “do lagged FEDFUNDS help predict UNRATE beyond lagged UNRATE?”

### Impulse responses (IRFs)

```python
irf = res.irf(12)
irf.plot(orth=True)  # orthogonalized IRFs (ordering matters)
```

**Important:** orthogonalized IRFs depend on a Cholesky ordering.

### Common pitfalls (and quick fixes)

- **Nonstationary inputs:** VAR on levels can be nonsense.
  - Fix: difference/logdiff; or use cointegration/VECM logic.
- **Too many lags:** eats degrees of freedom and can destabilize the model.
  - Fix: try smaller maxlags, compare AIC/BIC, check diagnostics.
- **Misinterpreting Granger causality:** it is about predictive content, not structural causality.
- **Forgetting ordering:** orth IRFs change when you reorder variables.
