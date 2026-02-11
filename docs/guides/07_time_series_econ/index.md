# Part 8: Time-Series Econometrics (Unit Roots → VAR)

This part complements the “time-aware ML” workflow with classical time-series econometrics:
stationarity, cointegration, and VAR-based dynamics (IRFs).

## What You Will Learn
- Stationarity and unit roots (ADF / KPSS) and why differencing helps
- Spurious regression (why levels-on-levels can lie)
- Cointegration and error correction models (ECM)
- VAR modeling, lag selection, Granger causality
- Impulse response functions (IRFs) and forecast-based evaluation

## Prerequisites
- Foundations (Part 0) time-series basics + leakage prevention
- Macro monthly panel (Part 1)

## How To Study This Part
- Always inspect plots before tests.
- Use multiple diagnostics (ADF + KPSS) and explain conflicts.
- When fitting VARs, explain:
  - how you chose lags,
  - which transformations made series “stationary enough”,
  - and what identification assumption your IRF uses (ordering).

## Chapters
- [00_stationarity_unit_roots](00_stationarity_unit_roots.md) — Notebook: [00_stationarity_unit_roots.ipynb](../../../notebooks/07_time_series_econ/00_stationarity_unit_roots.ipynb)
- [01_cointegration_error_correction](01_cointegration_error_correction.md) — Notebook: [01_cointegration_error_correction.ipynb](../../../notebooks/07_time_series_econ/01_cointegration_error_correction.ipynb)
- [02_var_impulse_responses](02_var_impulse_responses.md) — Notebook: [02_var_impulse_responses.ipynb](../../../notebooks/07_time_series_econ/02_var_impulse_responses.ipynb)
