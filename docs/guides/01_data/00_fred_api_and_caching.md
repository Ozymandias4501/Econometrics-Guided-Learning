# Guide: 00_fred_api_and_caching

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01_data/00_fred_api_and_caching.ipynb`.

This data module builds the datasets used throughout the project: a macro panel (FRED) and a micro cross-section (Census/ACS).

### Key Terms (defined)
- **API endpoint**: a URL path that returns a specific dataset.
- **Caching**: saving raw responses locally so experiments are reproducible and fast.
- **Frequency alignment**: converting mixed-frequency series (daily/monthly/quarterly) onto a common timeline.
- **Quarter-end timestamp**: representing a quarter by its final date to make merges unambiguous.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Choose series
- Complete notebook section: Fetch metadata
- Complete notebook section: Fetch + cache observations
- Complete notebook section: Fallback to sample
- Fetch or load sample data and inspect schemas (columns, dtypes, index).
- Build the GDP growth series and recession label exactly as specified.
- Create a quarterly modeling table with `target_recession_next_q` and no obvious leakage.

### Alternative Example (Not the Notebook Solution)
```python
# Toy GDP growth + technical recession label (not real GDP):
import pandas as pd

idx = pd.date_range('2018-03-31', periods=12, freq='QE')
gdp = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 102, 103, 104, 105], index=idx)

growth_qoq = 100 * (gdp / gdp.shift(1) - 1)
recession = ((growth_qoq < 0) & (growth_qoq.shift(1) < 0)).astype(int)
target_next_q = recession.shift(-1)
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Deep Dive: APIs + Caching (How Real-World Data Ingestion Works)

This project intentionally makes you interact with real APIs because data ingestion is where many ML projects fail.
In practice, you need to understand *protocols*, *schemas*, and *reproducibility*.

#### Key Terms (defined)
- **API (Application Programming Interface)**: a contract for requesting and receiving structured data.
- **HTTP**: the protocol used for requests/responses (what `requests.get(...)` uses under the hood).
- **Endpoint**: a specific API path (e.g., `series/observations`).
- **Query parameters**: key/value inputs in the URL (e.g., `series_id=UNRATE`).
- **Status code**: tells you whether a request succeeded (200) or failed (4xx/5xx).
- **Timeout**: how long you wait before giving up (prevents hanging forever).
- **Retry/backoff**: re-attempting requests after transient failures.
- **Schema**: expected structure of the JSON payload.
- **Cache**: stored copy of responses used to avoid repeated calls and make runs reproducible.

#### Why caching matters for learning (and for production)
- **Speed**: you can iterate without waiting on the network.
- **Reproducibility**: your results do not change because an endpoint changed or the data was revised.
- **Debuggability**: you can inspect raw payloads when parsing fails.

#### Raw vs processed data
- `data/raw/`: the API's response (JSON) with minimal transformation.
- `data/processed/`: tables you created (CSV) after cleaning and aligning time.

#### Python demo: minimal caching pattern
```python
from __future__ import annotations

import json
from pathlib import Path
import requests

def load_or_fetch_json(path: Path, fetch_fn):
    if path.exists():
        return json.loads(path.read_text())
    payload = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload

def fetch_example(url: str, params: dict):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()  # raises on 4xx/5xx
    return r.json()
```

#### Debug playbook: when your API code fails
1. Print the full URL + params (so you can reproduce outside Python).
2. Inspect the status code and response body.
3. Cache the raw payload and re-run parsing offline.
4. Validate schema assumptions (`payload.keys()`, sample rows).
5. Add defensive parsing (type conversion, missing markers, etc.).

#### Economics caveat: revisions and vintages
- Some macro series are revised after initial release (GDP is a classic example).
- If you re-fetch later, historical values can change; caching avoids silent drift.


### Common Mistakes
- Merging quarterly GDP with monthly predictors without explicit aggregation (silent misalignment).
- Using future quarterly features (e.g., lag -1) by accident.
- Forgetting that daily series need resampling before joining.

<a id="summary"></a>
## Summary + Suggested Readings

You now have a reproducible macro dataset with an explicit recession label and a micro dataset for cross-sectional inference.
From here, the project focuses on modeling and interpretation.


Suggested readings:
- FRED API documentation (series, observations)
- US Census API documentation (ACS endpoints, geography parameters)
- pandas documentation: resampling, merging/joining time series
