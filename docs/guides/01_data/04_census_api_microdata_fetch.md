# Guide: 04_census_api_microdata_fetch

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/01_data/04_census_api_microdata_fetch.ipynb`.

This data module builds the datasets used throughout the project: a macro panel (FRED) and a micro cross-section (Census/ACS).

### Key Terms (defined)
- **API endpoint**: a URL path that returns a specific dataset.
- **Caching**: saving raw responses locally so experiments are reproducible and fast.
- **Frequency alignment**: converting mixed-frequency series (daily/monthly/quarterly) onto a common timeline.
- **Quarter-end timestamp**: representing a quarter by its final date to make merges unambiguous.


<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Browse variables
- Complete notebook section: Fetch county data
- Complete notebook section: Derived rates
- Complete notebook section: Save processed data
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

### Core Data: Build Datasets You Can Trust

Modeling is downstream of data.
If you do not trust your data processing, you cannot trust your model output.

#### The three layers of a real data pipeline
> **Definition:** **Raw data** is the closest representation of what the source returned.

> **Definition:** **Processed data** is cleaned and aligned for analysis.

> **Definition:** A **modeling table** is the final table with features and targets aligned and ready for splitting.

#### Timing is part of the schema
When you build features, you are also defining "what was known when".
That is why frequency alignment and target shifting are first-class topics in this project.

#### A practical habit
Keep a simple data dictionary as you go:
- what is each column?
- what are its units?
- what frequency is it observed?
- what transformations did you apply?

### Deep Dive: APIs + Caching (How Real-World Data Ingestion Works)

This project makes you interact with real APIs because data ingestion is where many ML projects fail.
In practice, you need to understand protocols, schemas, error handling, and reproducibility.

#### Key terms (defined)
> **Definition:** An **API (Application Programming Interface)** is a contract for requesting and receiving structured data.

> **Definition:** An **endpoint** is a specific API path that returns a particular dataset.

> **Definition:** **Query parameters** are inputs you pass to an endpoint (like `series_id=UNRATE`).

> **Definition:** A **schema** is the expected structure of the response payload (field names, types).

> **Definition:** A **cache** is a local copy of responses used to avoid repeated calls and make runs reproducible.

#### Why caching matters
- Speed: iterate without waiting on the network.
- Reproducibility: your results do not change because the API changed or the data was revised.
- Debuggability: you can inspect raw payloads when parsing fails.

#### Raw vs processed data
- `data/raw/` should hold cached JSON responses (closest to the API output).
- `data/processed/` should hold your cleaned, aligned, analysis-ready tables.

#### Python demo: minimal caching pattern (commented)
```python
from __future__ import annotations

import json
from pathlib import Path
import requests

def load_or_fetch_json(path: Path, fetch_fn):
    """Load JSON from disk if present; otherwise fetch and write it."""
    if path.exists():
        return json.loads(path.read_text())

    payload = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload

def fetch_example(url: str, params: dict):
    """Minimal HTTP GET with a timeout and basic error handling."""
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()  # raises on 4xx/5xx
    return r.json()
```

#### Debug playbook (when API calls fail)
1. Print the full URL and params.
2. Check HTTP status code.
3. Cache the raw response and debug parsing offline.
4. Inspect payload keys (`payload.keys()`) and sample rows.
5. Add defensive parsing (type conversion, missing markers).

#### Economics caveat: revisions and vintages
Many macro series are revised (GDP is a classic example). If you re-fetch later, historical values can change.
Caching avoids silent drift.

### Project Code Map
- `scripts/fetch_census.py`: CLI fetch for Census/ACS
- `src/fred_api.py`: FRED client (`fetch_series_meta`, `fetch_series_observations`, `observations_to_frame`)
- `src/census_api.py`: Census/ACS client (`fetch_variables`, `fetch_acs`)
- `src/macro.py`: GDP + labels (`gdp_growth_qoq`, `gdp_growth_yoy`, `technical_recession_label`, `monthly_to_quarterly`)
- `scripts/build_datasets.py`: end-to-end dataset builder
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)

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
