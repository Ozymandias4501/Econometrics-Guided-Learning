### Core Data: build datasets you can trust (schema, timing, reproducibility)

Good modeling is downstream of good data. In econometrics, “good data” means more than “no missing values”:
it means the dataset matches the real timing and measurement of the economic problem.

#### 1) Intuition (plain English)

Most downstream mistakes trace back to one of these upstream issues:
- mixing frequencies incorrectly (monthly vs quarterly),
- misaligning timestamps (month-start vs month-end),
- using revised data as if it were known in real time,
- silently changing transformations (growth rate vs level) mid-project.

**Story example:** You merge monthly unemployment to quarterly GDP growth.
If you accidentally align the *end-of-quarter* unemployment with *start-of-quarter* GDP, you change the meaning of “what was known when.”

#### 2) Notation + setup (define symbols)

We will use the “time + horizon” language throughout the repo:
- $t$: time index (month/quarter/year),
- $X_t$: features available at time $t$,
- $y_{t+h}$: outcome/label defined $h$ periods ahead.

Data pipeline layers (repo convention):

1) **Raw data** (`data/raw/`)
- closest representation of the source response (API output, raw CSV)
- should be cacheable and re-creatable

2) **Processed data** (`data/processed/`)
- cleaned, aligned frequencies, derived columns
- “analysis-ready” but not necessarily “model-ready”

3) **Modeling table**
- final table where:
  - features are past-only,
  - labels are shifted to the forecast horizon,
  - missingness from lags/rolls is handled,
  - you can split without leakage.

#### 3) Assumptions (and why we state them explicitly)

Every dataset embeds assumptions:
- measurement units (percent vs fraction),
- seasonal adjustment,
- timing conventions (month-end, quarter-end),
- whether revisions matter (real-time vs revised),
- whether the sample composition changes over time.

You cannot remove assumptions; you can only make them explicit and test sensitivity.

#### 4) Mechanics: the minimum reliable pipeline

**(a) Ingest + cache**
- Always cache API responses to disk.
- Your code should be able to re-run without changing results.

**(b) Parse + type**
- Parse dates, set a proper index (DatetimeIndex for macro; MultiIndex for panels).
- Coerce numeric columns to numeric types (watch out for strings).

**(c) Align frequency**
- Resample to a common timeline (month-end, quarter-end).
- Decide and document whether you use `.last()` or `.mean()` (interpretation differs).

**(d) Create derived variables**
- growth rates, log differences, rolling windows, lag features.
- each transform changes interpretation; keep a small “data dictionary.”

**(e) Build the modeling table**
- define labels (e.g., next-quarter recession),
- shift targets forward and features backward (lags),
- drop missing rows created by lags/rolls.

#### 5) Inference and data (why timing affects standard errors)

Even inference topics (SE, p-values) depend on data structure:
- time series residuals are autocorrelated → HAC SE,
- panels share shocks within groups → clustered SE.

So “data” is not just a preprocessing step; it determines the correct inference method.

#### 6) Diagnostics + robustness (minimum set)

1) **Schema + units check**
- print `df.dtypes`, confirm units (percent vs fraction), and inspect summary stats.

2) **Index + frequency check**
- confirm sorted index, expected frequency, and no duplicate timestamps.

3) **Missingness check**
- print missingness per column before/after merges and transforms.

4) **Timing check**
- for a few rows, manually verify that features come from the past relative to the label.

5) **Sensitivity check**
- re-run a result using an alternative alignment (mean vs last) and see if conclusions change.

#### 7) Interpretation + reporting

When you present a result downstream, always include:
- which dataset version you used (processed vs sample),
- the frequency and timestamp convention,
- key transformations (diff, logdiff, growth rates),
- any known limitations (revisions, breaks).

**What this does NOT mean**
- “More features” does not equal “better data.”
- A perfectly clean dataset can still be conceptually wrong if timing is wrong.

#### Exercises

- [ ] For one dataset, write a 5-line data dictionary (column meaning + units + frequency).
- [ ] Demonstrate how `.last()` vs `.mean()` changes a resampled series and interpret the difference.
- [ ] Pick one merge/join and verify alignment by printing a few timestamps and values.
- [ ] Show one example where a shift direction would create leakage, and explain why.
