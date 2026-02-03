### Deep Dive: Monthly -> Quarterly Features (No-Leakage Engineering)

Your target is quarterly, but many predictors are monthly/daily.
You must transform predictors into quarterly features that were available at the time.

#### Key terms (defined)
> **Definition:** **Aggregation** summarizes higher-frequency data into lower-frequency data (monthly -> quarterly).

> **Definition:** A **lag** uses past values as features (e.g., $x_{t-1}$).

> **Definition:** The **prediction horizon** is how far ahead you predict (here: next quarter).

#### Two common quarterly feature definitions
- Quarter-average: mean of monthly values inside the quarter.
- Quarter-end: last available value in the quarter.

Both can be defensible. The goal is to choose intentionally and document it.

#### Leakage risk: "inside the target quarter"
Be explicit about the timestamp meaning.
- If you predict recession_{t+1} using quarter t features, features can use info up to the end of quarter t.
- If you want a mid-quarter prediction (nowcasting), you need partial-quarter features.

#### Python demo: quarterly aggregation + lags (commented)
```python
import pandas as pd

panel_monthly = pd.read_csv('data/sample/panel_monthly_sample.csv', index_col=0, parse_dates=True)

# 1) Aggregate
q_mean = panel_monthly.resample('QE').mean()
q_last = panel_monthly.resample('QE').last()

# 2) Choose one representation (or keep both with prefixes)
q = q_mean.add_prefix('mean_')

# 3) Add lags (past-only)
q['mean_UNRATE_lag1'] = q['mean_UNRATE'].shift(1)
q['mean_UNRATE_lag2'] = q['mean_UNRATE'].shift(2)

# 4) Drop rows created by lagging
q = q.dropna()
```

#### Debug checks for leakage
1. All shifts should be non-negative (past-only): `shift(+k)`.
2. Target is shifted the correct direction (`shift(-1)` for next period label).
3. After dropping NaNs, the final table has aligned indices for X and y.
