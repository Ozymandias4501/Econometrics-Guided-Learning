## Primer: Paths, files, and environment variables (how this repo stays reproducible)

You will see a few patterns repeatedly in notebooks and scripts.

### Environment variables (API keys)

Environment variables are key/value settings provided by your shell to Python.
This repo uses them for API keys:
- `FRED_API_KEY`
- `CENSUS_API_KEY` (optional)

```python
import os

fred_key = os.getenv("FRED_API_KEY")
print("FRED key set?", fred_key is not None)
```

If you set a key in a terminal, restart the Jupyter kernel so Python sees it.

### Paths (why `pathlib.Path` is the default)

Use `Path` to build OS-safe file paths:

```python
from pathlib import Path

p = Path("data") / "sample" / "macro_quarterly_sample.csv"
print(p, "exists?", p.exists())
```

### Repo bootstrap variables (defined in every notebook)

The notebook bootstrap cell defines:
- `PROJECT_ROOT` (repo root)
- `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `SAMPLE_DIR`

Prefer these over hard-coded relative paths.

### Sample vs processed data (offline-first)

Most notebooks follow this pattern:
1) try `data/processed/*` (real pipeline output)
2) fall back to `data/sample/*` (small offline dataset)

This keeps notebooks runnable without network access.

### Common “file not found” fixes

- Print the path and check `.exists()`
- Print current working directory:
  - `import os; print(os.getcwd())`
- Start Jupyter from the repo root (so bootstrap can find `src/` and `docs/`)
