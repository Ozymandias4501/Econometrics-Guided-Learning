## Primer: pandas time series essentials (indexing, resampling, lags)

Most “mysterious bugs” in time series work come from index and alignment mistakes. This primer gives you the minimum patterns to avoid them.

### 1) DatetimeIndex (the first thing to verify)

Most time-series operations assume a `DatetimeIndex`:

```python
import pandas as pd

df = df.copy()
df.index = pd.to_datetime(df.index)
df = df.sort_index()
assert isinstance(df.index, pd.DatetimeIndex)
```

**Expected output / sanity checks**
- `df.index.min(), df.index.max()` look reasonable
- `df.index.is_monotonic_increasing` is `True`

### 2) Resampling (frequency alignment)

Resampling converts one frequency to another. Choose the aggregation rule intentionally.

```python
# month-end last value (end-of-period)
df_me_last = df.resample("ME").last()

# month-end mean (average-of-period)
df_me_mean = df.resample("ME").mean()

# quarter-end mean
df_q_mean = df.resample("QE").mean()
```

**Interpretation matters**
- `.last()` treats end-of-period value as “the period’s value.”
- `.mean()` treats the period average as “the period’s value.”

### 3) Alignment and merging

When joining series, always check missingness after the join:

```python
merged = df1.join(df2, how="outer").sort_index()
print(merged.isna().sum().sort_values(ascending=False).head(10))
```

### 4) Lags and rolling windows (watch for leakage!)

```python
# lag 1 period (past-only)
df["x_lag1"] = df["x"].shift(1)

# rolling mean using past values ending at t
df["x_roll12"] = df["x"].rolling(12).mean()
```

**Leakage pitfalls**
- `shift(-1)` uses the future.
- `rolling(..., center=True)` uses the future.

### 5) A quick workflow you should repeat

1) Set and verify DatetimeIndex.
2) Resample intentionally (mean vs last).
3) Join and inspect missingness.
4) Add lags/rolls (past-only).
5) `dropna()` to build a clean modeling table.
