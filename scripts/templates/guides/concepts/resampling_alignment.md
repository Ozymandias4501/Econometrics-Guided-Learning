### Deep Dive: Frequency Alignment (Daily/Monthly/Quarterly)

Economic indicators arrive at different frequencies:
- GDP is quarterly.
- CPI/unemployment are monthly.
- Yield curve spreads can be daily.

To build a single modeling table, you must choose a timeline and convert everything onto it.

#### Key terms (defined)
> **Definition:** **Resampling** converts a time series to a new frequency (e.g., daily -> monthly).

> **Definition:** **Aggregation** combines many observations into one (mean/last/sum).

> **Definition:** **Forward-fill (ffill)** carries the last known value forward until updated.

> **Definition:** A **quarter-end timestamp** represents the quarter by its final date (used for merges).

#### Why month-end is a pragmatic choice
- Many macro series are monthly.
- Daily series can be summarized to month-end (`last`) or month-average (`mean`).
- Month-end timestamps make quarterly aggregation easier.

#### Python demo: daily -> month-end (last vs mean)
```python
import numpy as np
import pandas as pd

idx = pd.date_range('2020-01-01', periods=120, freq='D')

# Toy daily series
x_daily = pd.Series(np.random.default_rng(0).normal(size=len(idx)).cumsum(), index=idx)

# Two common summaries
x_me_last = x_daily.resample('ME').last()
x_me_mean = x_daily.resample('ME').mean()
```

#### Forward-fill (what it assumes)
Forward-fill assumes the last published value remains "true" until updated.
This is often reasonable for slow-moving series, but can hide missingness or create fake stability.

#### Debug checks
1. Index is datetime, sorted, unique.
2. After resampling, frequency looks right (ME for monthly, QE for quarterly).
3. Joins do not introduce unexpected NaNs.
4. You can explain why you chose mean vs last.
