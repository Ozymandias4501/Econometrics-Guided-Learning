# Guide: 01_build_macro_monthly_panel

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01_data/01_build_macro_monthly_panel.ipynb`.

This guide focuses on constructing a monthly macro panel from multiple FRED series that arrive at different frequencies. The central challenge is *frequency alignment*: how to combine monthly, quarterly, and daily series into a single coherent panel without introducing timing errors or information leakage.

> **Prerequisite:** This guide assumes familiarity with the Core Data primer (schema, timing, reproducibility) and FRED API caching patterns covered in [Guide 00: FRED API and Caching](00_fred_api_and_caching.md). That guide is the canonical reference for data pipeline concepts, notation, and diagnostics that apply across all data guides.

### Key Terms (defined)
- **Frequency alignment**: converting mixed-frequency series (daily/monthly/quarterly) onto a common timeline. The choice of aggregation method (`.mean()` vs `.last()`) changes the economic interpretation.
- **Resampling**: the pandas operation that converts a time series from one frequency to another (e.g., daily to monthly, monthly to quarterly).
- **Forward-fill (`ffill`)**: propagating the last observed value forward to fill gaps. Common for quarterly series observed at month-end when you need monthly observations.
- **Interpolation**: estimating missing values between observed points using a mathematical rule (linear, cubic, etc.). More sophisticated than forward-fill but introduces assumptions about the data-generating process.
- **DatetimeIndex**: pandas index type that enables time-aware operations (resampling, shifting, rolling windows). Essential for macro panels.
- **Month-end convention**: indexing all monthly observations to the last calendar day of the month (e.g., 2023-01-31, 2023-02-28). This ensures unambiguous alignment across series.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Load cached FRED series (multiple series with different native frequencies)
- Complete notebook section: Align all series to month-end timestamps
- Complete notebook section: Handle missingness introduced by frequency conversion
- Complete notebook section: Save the processed monthly panel to `data/processed/`
- Resample daily series (e.g., `T10Y2Y`) to monthly using an explicit rule (`.last()` or `.mean()`)
- Forward-fill quarterly series (e.g., `GDP`) to monthly frequency and document the assumption
- Verify the final panel has a consistent DatetimeIndex with no duplicate or missing months
- Inspect missingness per column before and after alignment

### Alternative Example (Not the Notebook Solution)
```python
# Demonstrate monthly resampling + alignment (not the notebook's exact code):
import pandas as pd
import numpy as np

# --- Simulate mixed-frequency data ---
# Monthly unemployment (already at target frequency)
monthly_idx = pd.date_range("2020-01-31", "2021-12-31", freq="ME")
unrate = pd.Series(np.random.uniform(3.5, 6.0, len(monthly_idx)),
                    index=monthly_idx, name="UNRATE")

# Daily treasury spread (needs downsampling to monthly)
daily_idx = pd.bdate_range("2020-01-01", "2021-12-31")
spread_daily = pd.Series(np.random.uniform(-0.5, 2.5, len(daily_idx)),
                          index=daily_idx, name="T10Y2Y")

# Quarterly GDP (needs upsampling to monthly)
quarterly_idx = pd.date_range("2020-03-31", "2021-12-31", freq="QE")
gdp = pd.Series([21000, 19500, 21200, 21800, 22100, 22400, 22700, 23000],
                 index=quarterly_idx, name="GDP")

# --- Resample daily -> monthly (end-of-month last value) ---
spread_monthly = spread_daily.resample("ME").last()

# --- Upsample quarterly -> monthly (forward-fill) ---
gdp_monthly = gdp.resample("ME").ffill()

# --- Combine into panel ---
panel = pd.DataFrame({
    "UNRATE": unrate,
    "T10Y2Y": spread_monthly,
    "GDP": gdp_monthly,
})

print(f"Panel shape: {panel.shape}")
print(f"Frequency: {pd.infer_freq(panel.index)}")
print(f"Missing values:\n{panel.isna().sum()}")
```

### Comparing `.mean()` vs `.last()` for Resampling
```python
# Concrete example: monthly unemployment -> quarterly
monthly_unemp = pd.Series(
    [3.8, 3.9, 4.1, 4.0, 3.8, 3.7],
    index=pd.date_range("2023-04-30", periods=6, freq="ME"),
    name="UNRATE"
)

quarterly_mean = monthly_unemp.resample("QE").mean()
quarterly_last = monthly_unemp.resample("QE").last()

print("Q2 2023 via .mean():", quarterly_mean.iloc[0])  # 3.933...
print("Q2 2023 via .last():", quarterly_last.iloc[0])   # 4.1
# .mean() = quarter average (typical condition)
# .last() = quarter-end snapshot (most recent information)
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: Resampling + alignment -- making mixed-frequency data mean what you think it means

Economics data often arrives at different frequencies (daily, monthly, quarterly) and with different timestamp conventions. Alignment is part of the econometric specification, not just a preprocessing step.

#### 1) Intuition (plain English)

If you merge time series incorrectly, you can create:
- artificial lead/lag relationships,
- leakage (future information),
- wrong interpretations ("end of quarter" vs "average of quarter").

**Story example:** GDP is quarterly, unemployment is monthly.
Is "quarterly unemployment" the average unemployment *during* the quarter, or the unemployment rate *at the end* of the quarter? Those are different variables with different economic interpretations.

**Health economics example:** You want to study whether monthly hospital admission rates predict quarterly CMS spending reports. Hospital admissions are reported monthly; CMS publishes quarterly. If you use `.mean()` to aggregate admissions, you get the quarter's average patient volume. If you use `.last()`, you get June's admission rate for Q2 -- which may spike due to end-of-fiscal-year effects. The choice changes your regression coefficient's interpretation.

#### 2) Notation + setup (define symbols)

Let:
- $x_t^{(M)}$ be a monthly series,
- $y_q^{(Q)}$ be a quarterly series.

Resampling defines a function that maps monthly values inside quarter $q$ into a quarterly feature:

$$
\tilde x_q = g\left(\{x_t^{(M)} : t \in q\}\right).
$$

Common choices:
- $g=\text{mean}$ (quarter average),
- $g=\text{last}$ (end-of-quarter value),
- $g=\text{first}$ (start-of-quarter value),
- $g=\text{sum}$ (quarter total -- appropriate for flow variables like admissions or spending).

**Concrete example:** Monthly unemployment rates for Q2: April = 3.8%, May = 3.9%, June = 4.1%.
- `.mean()` gives 3.93% -- the quarter's *typical* labor market condition.
- `.last()` gives 4.1% -- the most recent reading available at the quarter boundary.

For predicting next-quarter GDP growth, `.last()` is often preferred because it captures the most recent information that a forecaster would actually have at the decision point. But if you are modeling the quarter's *average* economic environment (e.g., for a structural regression), `.mean()` may be more appropriate.

**When to use each:**

| Method    | Interpretation                          | Best for                                      |
|-----------|-----------------------------------------|-----------------------------------------------|
| `.mean()` | Average condition during the period     | Structural models, policy evaluation           |
| `.last()` | Most recent observation at period end   | Forecasting (captures latest info)             |
| `.first()`| Condition at period start               | Lagged information sets                        |
| `.sum()`  | Total over the period                   | Flow variables (spending, admissions, output)  |

#### 3) Assumptions (what your resampling choice implies)

Choosing `.mean()` assumes:
- the quarter's "typical" level matters,
- within-quarter variation is noise (or irrelevant to your question).

Choosing `.last()` assumes:
- the quarter-end level is what matters (or is what was known at quarter-end),
- within-quarter dynamics are captured by the endpoint.

Neither is universally correct; you must choose based on the economic question and the information set you want to represent.

**Forward-fill assumptions:** When you forward-fill quarterly GDP to monthly frequency, you assume GDP is constant within the quarter. This is obviously wrong in reality -- economic activity varies month to month -- but it avoids the stronger assumptions embedded in interpolation (which imposes a functional form on within-quarter dynamics). Forward-fill is conservative: it says "use the last known value" rather than "guess what happened between observations."

**Interpolation assumptions:** Linear interpolation assumes the variable changes at a constant rate between observations. Cubic spline interpolation assumes smooth, continuous changes. Both are stronger assumptions than forward-fill. Use interpolation only when you have a substantive reason to believe the within-period dynamics follow the assumed pattern.

#### 4) Mechanics: practical alignment steps

1. **Establish a target DatetimeIndex**: `target_idx = pd.date_range("2000-01-31", "2023-12-31", freq="ME")`
2. **Downsample** high-frequency series: `spread_monthly = spread_daily.resample("ME").last()`
3. **Upsample** low-frequency series: `gdp_monthly = gdp_quarterly.resample("ME").ffill()`
4. **Combine** on aligned index: `panel = pd.concat([...], axis=1).reindex(target_idx)`
5. **Inspect** missingness and boundaries: `panel.isna().sum()` and `panel.head(15)`

#### 5) Inference: alignment affects serial correlation and effective sample size

Aggregation changes time-series dependence:
- averaging smooths noise and can increase persistence (higher autocorrelation),
- end-of-period values can be more volatile (lower autocorrelation).

So alignment also affects inference (HAC choices, stationarity checks). If you aggregate monthly data to quarterly using `.mean()`, the resulting quarterly series will be smoother than the underlying monthly process. This induced smoothness increases serial correlation, which means Newey-West standard errors may require more lags than you would expect from the quarterly frequency alone.

**Effective sample size.** Forward-filling quarterly data to monthly does not create new information. You still have only one independent quarterly observation per three months. Treating the forward-filled monthly values as independent observations in a regression would overstate your effective sample size and understate standard errors.

#### 6) Diagnostics + robustness (minimum set)

1) **Plot before/after resampling**
- confirm the resampled series looks like what you intended.

2) **Check timestamp conventions**
- month-end vs month-start; quarter-end vs quarter-start. One common pitfall: `pd.date_range(freq="MS")` gives month-*start* dates, while `freq="ME"` gives month-*end*. Merging series with different conventions will produce NaN-filled joins.

3) **Compare mean vs last**
- run both and see if key results are sensitive. If a regression coefficient changes sign depending on the resampling rule, that is a red flag about the robustness of the result.

4) **Verify no future information leaks**
- after forward-filling, check that the GDP value for January 2023 comes from Q4 2022 (the last completed quarter), not Q1 2023 (which is not yet observed). This is the most common source of subtle leakage in mixed-frequency panels.

5) **Confirm DatetimeIndex properties** -- assert sorted, no duplicates, `pd.infer_freq(panel.index) == "ME"`.

#### 7) Interpretation + reporting

Always state:
- the resampling rule (mean/last/sum/ffill),
- the timestamp convention (month-end),
- and the intended economic interpretation.

Example: "Monthly panel uses month-end timestamps. Daily treasury spread is resampled using `.last()` (end-of-month value). Quarterly GDP is forward-filled to monthly frequency, so within-quarter months carry the prior quarter's GDP level."

#### Exercises

- [ ] Resample a monthly series to quarterly using `.mean()` and `.last()`; plot both and describe the difference in economic terms.
- [ ] Forward-fill a quarterly series to monthly; then compare to linear interpolation. Plot both and discuss the assumptions.
- [ ] Merge a monthly and a quarterly series into a single DataFrame; verify no unexpected missingness appears at quarter boundaries.
- [ ] Choose one resampling rule and defend it in 5 sentences for your specific modeling goal.
- [ ] Align monthly hospital admissions with quarterly CMS spending data and describe which aggregation method is appropriate for predicting spending.

### Date Indexing Best Practices

- Always use `DatetimeIndex` (not string dates): `df.index = pd.to_datetime(df.index)`.
- Use `freq="ME"` (month-end) consistently. `freq="MS"` (month-start) will cause NaN-filled joins with month-end series.
- After filtering/subsetting, restore frequency with `df.asfreq("ME")` -- pandas may lose the `freq` attribute.
- Prefer `DatetimeIndex` over `PeriodIndex` for this project: resampling, shifting, and merging are more predictable.

### Forward-Fill vs Interpolation

- **Forward-fill**: repeats last known value. Conservative -- does not fabricate data. Use for quarterly GDP -> monthly.
- **Linear interpolation**: assumes constant rate of change between observations. Use only when domain knowledge supports it.
- **Rule of thumb:** if unsure, use forward-fill. It is the safer default.

### Project Code Map
- `src/fred_api.py`: FRED client (`fetch_series_meta`, `fetch_series_observations`, `observations_to_frame`)
- `src/macro.py`: GDP + labels (`gdp_growth_qoq`, `gdp_growth_yoy`, `technical_recession_label`, `monthly_to_quarterly`)
- `scripts/build_datasets.py`: end-to-end dataset builder
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)

### Common Mistakes
- Merging quarterly GDP with monthly predictors without explicit aggregation (silent misalignment).
- Using `freq="MS"` (month-start) when you mean `freq="ME"` (month-end), causing NaN-filled joins.
- Forward-filling without checking that the fill comes from the *prior* quarter, not the *current* quarter (leakage).
- Treating forward-filled monthly values as independent observations (overstates effective sample size).
- Forgetting that daily series may have gaps (weekends, holidays) -- `.last()` handles this, but `.loc` on a specific date may return NaN.
- Not documenting the resampling rule, so a collaborator cannot tell if "quarterly unemployment" means the average or the endpoint.

<a id="summary"></a>
## Summary + Suggested Readings

This guide covered the mechanics and interpretation of building a monthly macro panel from mixed-frequency FRED data. The core decisions are: (1) what resampling rule to apply for each series, (2) how to handle missing observations from frequency conversion, and (3) how to verify that the resulting panel has correct timing and no leakage.

Key takeaways:
- `.mean()` and `.last()` are not interchangeable -- they answer different economic questions.
- Forward-fill is conservative but produces stale values; interpolation is smoother but adds assumptions.
- DatetimeIndex with month-end convention is the standard for this project.
- Always verify alignment by inspecting a few rows manually, especially at quarter boundaries.

For the Core Data primer (schema, timing, reproducibility fundamentals), see [Guide 00](00_fred_api_and_caching.md).

Suggested readings:
- Ghysels, E., Sinko, A., & Valkanov, R. (2007), "MIDAS Regressions" -- on mixed-frequency data in econometrics
- pandas documentation: `DataFrame.resample`, `DataFrame.asfreq`, `DataFrame.interpolate`
- Stock, J.H. & Watson, M.W. (2002), "Macroeconomic Forecasting Using Diffusion Indexes" -- on constructing macro panels for forecasting
