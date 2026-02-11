# Guide: 00_fred_api_and_caching

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01_data/00_fred_api_and_caching.ipynb`.

This data module introduces the FRED (Federal Reserve Economic Data) API, demonstrates how to fetch and cache economic time series, and establishes the data engineering foundations used throughout the project.

### Key Terms (defined)
- **API endpoint**: a URL path that returns a specific dataset. For FRED, the two main endpoints are `fred/series` (metadata) and `fred/series/observations` (data values).
- **Caching**: saving raw API responses locally so experiments are reproducible and fast, even if the API is unavailable or the data is revised.
- **Frequency alignment**: converting mixed-frequency series (daily/monthly/quarterly) onto a common timeline.
- **Quarter-end timestamp**: representing a quarter by its final date (e.g., 2023-03-31 for Q1) to make merges unambiguous.
- **Series ID**: FRED's unique identifier for each economic time series (e.g., `GDP`, `UNRATE`, `FEDFUNDS`).
- **Vintage / real-time data**: the version of data that was available at a given historical date, before subsequent revisions.

### FRED Series Reference

The following series IDs are used throughout this project:

| Series ID     | Description                        | Frequency  | Units              |
|---------------|------------------------------------|------------|--------------------|
| `GDP`         | Gross Domestic Product             | Quarterly  | Billions of $      |
| `UNRATE`      | Civilian Unemployment Rate         | Monthly    | Percent            |
| `CPIAUCSL`    | Consumer Price Index (All Urban)   | Monthly    | Index (1982=100)   |
| `FEDFUNDS`    | Federal Funds Effective Rate       | Monthly    | Percent            |
| `GS10`        | 10-Year Treasury Constant Maturity | Monthly    | Percent            |
| `T10Y2Y`      | 10Y-2Y Treasury Spread             | Daily      | Percent            |
| `INDPRO`      | Industrial Production Index        | Monthly    | Index (2017=100)   |
| `PAYEMS`      | Total Nonfarm Payrolls             | Monthly    | Thousands          |

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to learn the math/assumptions (open any `<details>` blocks for optional depth).
- Then return to the notebook and write a short interpretation note after each section.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Choose series (select FRED series IDs for your macro panel)
- Complete notebook section: Fetch metadata (use `fetch_series_meta` to inspect units, frequency, seasonal adjustment)
- Complete notebook section: Fetch + cache observations (use `fetch_series_observations` with local caching)
- Complete notebook section: Fallback to sample (load from `data/sample/` if FRED API key is unavailable)
- Inspect the raw JSON response structure: confirm `observations` key, `date`/`value` fields, and "." for missing values
- Convert raw observations to a pandas DataFrame with proper DatetimeIndex and numeric dtype
- Verify cached files exist on disk and that re-running the pipeline produces identical output

### Alternative Example (Not the Notebook Solution)
```python
# Demonstrate FRED API fetch + cache pattern (not the notebook's exact code):
import os, json, requests, pandas as pd
from pathlib import Path

FRED_BASE = "https://api.stlouisfed.org/fred"
API_KEY = os.environ.get("FRED_API_KEY", "")

def fetch_fred_observations(series_id: str, cache_dir: str = "data/raw") -> pd.DataFrame:
    """Fetch FRED series observations with local JSON caching."""
    cache_path = Path(cache_dir) / f"{series_id}_obs.json"
    if cache_path.exists():
        with open(cache_path) as f:
            payload = json.load(f)
    else:
        url = f"{FRED_BASE}/series/observations"
        params = {"series_id": series_id, "api_key": API_KEY, "file_type": "json"}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(payload, f)

    df = pd.DataFrame(payload["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[["value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." -> NaN
    df.columns = [series_id]
    return df
```

### Inspecting FRED Metadata
```python
# Always check metadata BEFORE fetching observations:
# fetch_series_meta("UNRATE")
# -> {'id': 'UNRATE', 'title': 'Unemployment Rate', 'units': 'Percent',
#     'frequency': 'Monthly', 'seasonal_adjustment': 'Seasonally Adjusted'}
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

> **Canonical Data Engineering Primer.** This section is the canonical data engineering primer for the project. Guides 01-04 cover specific topics (monthly panel construction, GDP/recession labels, Census data, feature engineering) and cross-reference this section rather than duplicating it.

### Core Data: build datasets you can trust (schema, timing, reproducibility)

Good modeling is downstream of good data. In econometrics, "good data" means more than "no missing values":
it means the dataset matches the real timing and measurement of the economic problem.

#### 1) Intuition (plain English)

Most downstream mistakes trace back to one of these upstream issues:
- mixing frequencies incorrectly (monthly vs quarterly),
- misaligning timestamps (month-start vs month-end),
- using revised data as if it were known in real time,
- silently changing transformations (growth rate vs level) mid-project.

**Story example:** You merge monthly unemployment to quarterly GDP growth.
If you accidentally align the *end-of-quarter* unemployment with *start-of-quarter* GDP, you change the meaning of "what was known when."

#### 2) Notation + setup (define symbols)

We will use the "time + horizon" language throughout the repo:
- $t$: time index (month/quarter/year),
- $X_t$: features available at time $t$,
- $y_{t+h}$: outcome/label defined $h$ periods ahead.

Data pipeline layers (repo convention):

1) **Raw data** (`data/raw/`)
- closest representation of the source response (API output, raw CSV)
- should be cacheable and re-creatable

2) **Processed data** (`data/processed/`)
- cleaned, aligned frequencies, derived columns
- "analysis-ready" but not necessarily "model-ready"

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
- each transform changes interpretation; keep a small "data dictionary."

**(e) Build the modeling table**
- define labels (e.g., next-quarter recession),
- shift targets forward and features backward (lags),
- drop missing rows created by lags/rolls.

#### 5) Inference and data (why timing affects standard errors)

Even inference topics (SE, p-values) depend on data structure:
- time series residuals are autocorrelated → HAC SE,
- panels share shocks within groups → clustered SE.

So "data" is not just a preprocessing step; it determines the correct inference method.

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
- "More features" does not equal "better data."
- A perfectly clean dataset can still be conceptually wrong if timing is wrong.

#### Exercises

- [ ] For one dataset, write a 5-line data dictionary (column meaning + units + frequency).
- [ ] Demonstrate how `.last()` vs `.mean()` changes a resampled series and interpret the difference.
- [ ] Pick one merge/join and verify alignment by printing a few timestamps and values.
- [ ] Show one example where a shift direction would create leakage, and explain why.

### Deep Dive: FRED API patterns -- reproducible data ingestion for economic research

The FRED API is the standard programmatic interface to Federal Reserve Economic Data. Understanding its structure, quirks, and caching patterns is essential for reproducible econometric work.

#### 1) Intuition (plain English)

FRED hosts over 800,000 economic time series. The API lets you:
- search for series by keyword or category,
- retrieve metadata (units, frequency, seasonal adjustment status),
- download observations (date-value pairs) for any date range.

Without caching, every notebook run hits the API, which means:
- rate limits can break your workflow (FRED allows ~120 requests/minute),
- data revisions change values silently between runs,
- you cannot work offline or guarantee reproducibility.

**Story example:** You fetch GDP on Monday and get a preliminary estimate. BEA revises the number on Thursday. Without caching, your Tuesday and Friday results differ even though your code did not change.

#### 2) Notation + setup (define terms)

Think of your ingestion as a function:

$$
\text{data} = F(\text{endpoint}, \text{params}).
$$

Caching adds a persistent mapping:

$$
\text{cache\_key} = H(\text{endpoint}, \text{params}),
\quad
\text{cache}[\text{cache\_key}] = \text{data}.
$$

**What each term means**
- $F$: API fetch function (e.g., `requests.get` to FRED).
- $H$: hash/key function that uniquely identifies a request (typically `{series_id}_{endpoint_type}.json`).
- cache: local file storage (JSON for raw responses, CSV/Parquet for processed data).

**FRED API endpoints used in this project:**
- `fred/series` -- metadata (units, frequency, seasonal adjustment)
- `fred/series/observations` -- time series data (date-value pairs)
- `fred/series/search` -- search by keyword

#### 3) Assumptions (and what caching does/does not solve)

Caching assumes:
- you want repeatability for a given request,
- you can store raw responses (or cleaned versions) locally.

Caching does not solve:
- conceptual mistakes (wrong variables, wrong frequency),
- revisions vs real-time availability (you still must decide which you want).

**FRED-specific gotchas:**
- FRED returns `"."` for missing values (not `NaN`). You must parse explicitly.
- Seasonal adjustment matters: `UNRATE` vs `UNRATENSA` are different series.
- Frequency codes in metadata (`"Monthly"`, `"Quarterly"`, etc.) -- always check before assuming.

#### 4) Mechanics: FRED-specific caching patterns

Best practice in this repo:

**Step 1: Fetch metadata first.**
Before downloading observations, inspect the series metadata to confirm units, frequency, and seasonal adjustment. This prevents silent misinterpretation.

**Step 2: Cache raw JSON responses.**
Save the complete API response (including metadata fields like `realtime_start`, `realtime_end`) so you can reconstruct exactly what was available at fetch time.

**Step 3: Parse with explicit type handling.**
FRED returns `"."` for missing values. Use `pd.to_numeric(df["value"], errors="coerce")` to convert to NaN.

**Step 4: Write processed datasets.**
Save processed versions (CSV/Parquet) so downstream notebooks do not need API access. Cache naming: `data/raw/{SERIES_ID}_obs.json` (raw), `data/processed/macro_panel.csv` (processed).

#### 5) Inference: reproducibility affects credibility

Inference is not just math; it is also:
- "Can someone reproduce the exact table/figure?"
- "Can we trace a result to a specific dataset version?"

Caching is therefore part of scientific validity, especially in health economics where policy conclusions may depend on specific data vintages.

#### 6) Diagnostics + robustness (minimum set)

1) **Cache hit/miss logging** -- print whether you loaded from disk or fetched from API.
2) **Schema checks after loading** -- validate columns, dtypes, and expected date range.
3) **Re-run consistency** -- run the pipeline twice and confirm identical outputs.
4) **Staleness check** -- flag if cached data is older than your analysis window.

#### 7) Interpretation + reporting

When presenting results, state:
- whether data came from cached raw responses or fresh API calls,
- the FRED series IDs used (so others can reproduce),
- the date range of observations,
- and where the cached artifacts live (relative to the repo root).

#### Exercises

- [ ] Run a dataset build twice; confirm the second run uses cached raw data.
- [ ] Delete one cached file and confirm the pipeline re-fetches and re-caches it.
- [ ] Fetch two series with different frequencies (e.g., `GDP` quarterly, `UNRATE` monthly) and inspect how their raw date ranges differ.

### Project Code Map
- `scripts/fetch_fred.py`: CLI fetch for FRED
- `src/fred_api.py`: FRED client (`fetch_series_meta`, `fetch_series_observations`, `observations_to_frame`)
- `src/macro.py`: GDP + labels (`gdp_growth_qoq`, `gdp_growth_yoy`, `technical_recession_label`, `monthly_to_quarterly`)
- `scripts/build_datasets.py`: end-to-end dataset builder
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)

### Common Mistakes
- Fetching data without checking metadata first (wrong units, unexpected frequency).
- Not handling FRED's `"."` missing value sentinel, leading to string columns instead of numeric.
- Confusing seasonally adjusted and non-adjusted variants of the same series.
- Merging quarterly GDP with monthly predictors without explicit aggregation (silent misalignment).
- Using future quarterly features (e.g., lag -1) by accident.
- Forgetting that daily series need resampling before joining.
- Assuming FRED data is final; GDP and employment data are revised multiple times.

<a id="summary"></a>
## Summary + Suggested Readings

This guide established two foundations: (1) the Core Data primer for building trustworthy econometric datasets, and (2) the FRED API caching patterns that ensure reproducibility. Guides 01-04 cross-reference this section for shared concepts.

Suggested readings:
- [FRED API documentation](https://fred.stlouisfed.org/docs/api/fred/) (series, observations, search endpoints)
- pandas documentation: `pd.to_datetime`, `pd.to_numeric`, `DataFrame.resample`
- Croushore, D. (2011), "Frontiers of Real-Time Data Analysis" -- on data revisions and real-time vs revised data
