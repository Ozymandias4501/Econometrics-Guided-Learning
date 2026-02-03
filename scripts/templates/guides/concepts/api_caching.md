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
