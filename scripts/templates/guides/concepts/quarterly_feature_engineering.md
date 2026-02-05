### Deep Dive: Quarterly feature engineering — lags, changes, and rolling windows

Feature engineering on macro panels is largely about **timing discipline**.

#### 1) Intuition (plain English)

In macro forecasting, the most useful features are often:
- lags (what happened recently),
- changes (growth rates),
- rolling summaries (recent averages/volatility).

But every transform can introduce leakage if done incorrectly.

#### 2) Notation + setup (define symbols)

Let $x_t$ be a quarterly series.

Common transforms:

- Lag:
$$
x_{t-1}
$$

- Difference:
$$
\\Delta x_t = x_t - x_{t-1}
$$

- Percent change (approx):
$$
\\%\\Delta x_t \\approx \\frac{x_t - x_{t-1}}{x_{t-1}}
$$

- Rolling mean over $k$ quarters (past-only):
$$
\\overline{x}_{t,k} = \\frac{1}{k}\\sum_{j=0}^{k-1} x_{t-j}
$$

#### 3) Assumptions

These transforms assume:
- the series is measured comparably over time,
- timestamps reflect when values are “known,”
- rolling windows use past data only (no centering).

#### 4) Mechanics: practical feature-building rules

1) Build features in a copy of the DataFrame and keep column naming consistent.
2) After creating lags/rolls, `dropna()` to get a clean modeling table.
3) Always validate that a lag feature at time $t$ uses data from $t-1$ (not $t+1$).

#### 5) Inference: transforms change dependence

Differencing can reduce nonstationarity but can increase noise.
Rolling means smooth noise but can increase persistence.

So transforms affect:
- stationarity tests,
- SE choices (HAC),
- model stability.

#### 6) Diagnostics + robustness (minimum set)

1) **Spot-check rows**
- print a few timestamps and verify lag alignment manually.

2) **Compare transforms**
- levels vs differences vs growth rates; do relationships change?

3) **Overfitting guard**
- more features increase variance; use time-aware evaluation.

#### 7) Interpretation + reporting

When reporting features, always specify:
- “lag 1 quarter,”
- “12-quarter rolling mean,” etc.

#### Exercises

- [ ] Create lag and rolling features for two macro indicators and verify alignment by printing rows.
- [ ] Compare a model using levels vs differences; interpret the change.
- [ ] Create a leakage feature intentionally (shift the wrong way) and show how it inflates performance.
