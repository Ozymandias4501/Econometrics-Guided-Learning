## Primer: Paths, Files, and Environment Variables

You will see a few patterns repeatedly in this repo.

### Environment variables
> **What this is:** Environment variables are key/value settings provided by your shell to your Python process.

We use them for API keys and configuration defaults.

```python
import os

# Reads an environment variable or returns None
fred_key = os.getenv('FRED_API_KEY')
print('FRED key set?', fred_key is not None)
```

If you're running from a terminal, you can set a key like this:

```bash
export FRED_API_KEY="your_key_here"
```

Then restart the Jupyter kernel (so Python picks up the new env var).

### Paths (why `pathlib.Path`)
> **What this is:** A Path is a safe way to build file paths without worrying about OS-specific separators.

```python
from pathlib import Path

p = Path('data') / 'sample' / 'macro_quarterly_sample.csv'
print(p)
print('exists?', p.exists())
```

In these notebooks, the bootstrap cell defines:
- `PROJECT_ROOT` (repo root)
- `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `SAMPLE_DIR`

Prefer those over hard-coding paths.

### Reading and writing CSV files
```python
import pandas as pd

# Read
# df = pd.read_csv(p, index_col=0, parse_dates=True)

# Write
# out = Path('data') / 'processed' / 'my_dataset.csv'
# out.parent.mkdir(parents=True, exist_ok=True)
# df.to_csv(out)
```

### Tip
If you get a "file not found" error:
- `print(path)` to confirm you're reading what you think you're reading
- `print(path.exists())` to confirm the file exists
- if you're using a relative path, confirm your current working directory: `import os; print(os.getcwd())`
