### Deep Dive: Resampling + alignment — making mixed-frequency data mean what you think it means

Economics data often arrives at different frequencies (daily, monthly, quarterly) and with different timestamp conventions. Alignment is part of the econometric specification.

#### 1) Intuition (plain English)

If you merge time series incorrectly, you can create:
- artificial lead/lag relationships,
- leakage (future information),
- wrong interpretations (“end of quarter” vs “average of quarter”).

**Story example:** GDP is quarterly, unemployment is monthly.  
Is “quarterly unemployment” the average unemployment *during* the quarter, or the unemployment rate *at the end* of the quarter? Those are different variables.

#### 2) Notation + setup (define symbols)

Let:
- $x_t^{(M)}$ be a monthly series,
- $y_q^{(Q)}$ be a quarterly series.

Resampling defines a function that maps monthly values inside quarter $q$ into a quarterly feature:

$$
\\tilde x_q = g\\left(\\{x_t^{(M)} : t \\in q\\}\\right).
$$

Common choices:
- $g=\\text{mean}$ (quarter average),
- $g=\\text{last}$ (end-of-quarter value).

#### 3) Assumptions (what your resampling choice implies)

Choosing `.mean()` assumes:
- the quarter’s “typical” level matters.

Choosing `.last()` assumes:
- the quarter-end level is what matters (or is what was known at quarter-end).

Neither is universally correct; you must choose based on measurement and use case.

#### 4) Mechanics: practical alignment steps

1) Ensure you have a proper DatetimeIndex and sorting.
2) Resample lower-frequency series to match higher frequency *or vice versa* with an explicit rule.
3) Join on the aligned index.
4) Inspect missingness and boundaries after the join.

#### 5) Inference: alignment affects serial correlation and effective sample size

Aggregation changes time-series dependence:
- averaging smooths noise and can increase persistence,
- end-of-period values can be more volatile.

So alignment also affects inference (HAC choices, stationarity checks).

#### 6) Diagnostics + robustness (minimum set)

1) **Plot before/after resampling**
- confirm the resampled series looks like what you intended.

2) **Check timestamp conventions**
- month-end vs month-start; quarter-end vs quarter-start.

3) **Compare mean vs last**
- run both and see if key results are sensitive.

#### 7) Interpretation + reporting

Always state:
- the resampling rule (mean/last/sum),
- the timestamp convention,
- and the intended economic interpretation.

#### Exercises

- [ ] Resample a monthly series to quarterly using `.mean()` and `.last()`; plot both.
- [ ] Merge with a quarterly target and verify no unexpected missingness appears.
- [ ] Choose one resampling rule and defend it in 5 sentences for your modeling goal.
