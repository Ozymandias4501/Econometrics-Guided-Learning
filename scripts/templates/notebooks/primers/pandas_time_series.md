## Primer: pandas Time Series Essentials

### DatetimeIndex
Most time series work in pandas assumes your DataFrame is indexed by time.

```python
import pandas as pd

# idx = pd.to_datetime(df['date'])
# df = df.set_index(idx).sort_index()
# assert isinstance(df.index, pd.DatetimeIndex)
```

If you see weird behavior (resample errors, merges not aligning), check:
- `df.index.dtype`
- `df.index.min(), df.index.max()`
- `df.index.is_monotonic_increasing`

### Resampling
> **Goal:** convert one frequency to another.

Common examples:
- daily -> month-end
- monthly -> quarter-end

```python
# month-end last value
# df_me_last = df.resample('ME').last()

# month-end mean
# df_me_mean = df.resample('ME').mean()

# quarter-end mean
# df_q_mean = df.resample('QE').mean()
```

**Interpretation matters.** For economic series:
- using `.last()` treats the end-of-period value as “the period’s value”
- using `.mean()` treats the period average as “the period’s value”

Neither is universally correct; you should choose based on measurement and use case.

### Alignment and merging
When joining multiple time series, you need to ensure they share:
- the same index type (`DatetimeIndex`)
- the same frequency convention (month-end vs month-start; quarter-end vs quarter-start)

```python
# Example: join two series and inspect missingness
# df = df1.join(df2, how='outer').sort_index()
# print(df.isna().sum())
```

### Lags and rolling windows
> **Lag:** use past values as features.

```python
# lag 1 period
# df['x_lag1'] = df['x'].shift(1)

# rolling mean (past-only)
# df['x_roll12'] = df['x'].rolling(12).mean()
```

### Common gotchas
- `shift(-1)` uses the future (leakage for forecasting).
- `rolling(..., center=True)` uses future values.
- Always `dropna()` after creating lags/rolls to get clean modeling rows.

One more gotcha:
- If you resample daily -> monthly and then create lags, your lag is “one month” (not one day). Lags are measured in the current index frequency.
