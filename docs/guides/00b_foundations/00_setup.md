# Guide: 00_setup

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

This guide accompanies the notebook `notebooks/00b_foundations/00_setup.ipynb`.

This notebook configures your environment, verifies API keys, loads sample data, and introduces the validation patterns you will use in every subsequent notebook. No modeling happens here — the goal is to make sure everything runs and you understand the project layout before moving on.

### Key Terms (defined)
- **PROJECT_ROOT**: the absolute path to the top-level repo directory. Every notebook computes this so that imports and file paths work regardless of where Jupyter was launched.
- **Environment variable**: a key-value setting provided by your shell (e.g., `FRED_API_KEY`). Python reads them with `os.getenv()`. They keep secrets out of code.
- **Sample data**: small offline datasets bundled in `data/sample/` so notebooks run without network access. Real pipeline outputs go to `data/processed/`.
- **Data validation / assertions**: runtime checks (shape, dtype, range, index monotonicity) that catch silent errors early — before they propagate into modeling results.

### How To Read This Guide
- Use **Step-by-Step** to understand what you must implement in the notebook.
- Use **Technical Explanations** to understand the repo structure and data patterns.
- Then return to the notebook and complete the TODOs.

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
- Complete notebook section: Environment bootstrap — run the cell and confirm `PROJECT_ROOT` prints the repo root.
- Complete notebook section: Verify API keys — read `FRED_API_KEY` and `CENSUS_API_KEY` with `os.getenv()` and print whether each is set (without printing the full key).
- Complete notebook section: Load sample data — load `data/sample/macro_quarterly_sample.csv`, inspect its shape, columns, and dtypes.
- Complete notebook section: Checkpoints — write assertions that validate the loaded DataFrame (monotonic index, minimum rows, expected columns).
- Create `data/raw/` and `data/processed/` directories if they don't exist.
- Complete all TODOs (no `...` left).

### Alternative Example (Not the Notebook Solution)
```python
# Loading and validating a CSV (not the notebook data):
from pathlib import Path
import pandas as pd

path = Path("data/sample/macro_quarterly_sample.csv")
assert path.exists(), f"File not found: {path}"

df = pd.read_csv(path, index_col=0, parse_dates=True)

# Validation checks you should build the habit of running:
assert isinstance(df.index, pd.DatetimeIndex), "Index should be datetime"
assert df.index.is_monotonic_increasing, "Index should be sorted"
assert df.shape[0] > 20, f"Expected >20 rows, got {df.shape[0]}"
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Missing values:\n{df.isna().sum().sort_values(ascending=False).head(5)}")
```


<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

### Project structure: how guides, notebooks, and source code relate

```
stats_learning/
├── notebooks/          ← where you work (Jupyter, section by section)
│   ├── 00b_foundations/
│   ├── 01_data/
│   ├── 02_regression/
│   └── ...
├── docs/guides/        ← companion reference for each notebook (this file)
├── docs/cheatsheets/   ← quick-reference sheets for key topics
├── src/                ← shared Python modules (imported by notebooks + scripts)
│   ├── features.py     ← lag, rolling, diff, log-diff feature helpers
│   ├── evaluation.py   ← train/test splits, walk-forward CV, metrics
│   ├── econometrics.py ← OLS, HC3, HAC, VIF wrappers
│   ├── fred_api.py     ← FRED API client with retry + caching
│   └── data.py         ← JSON caching helpers
├── scripts/            ← CLI entry points (build datasets, train models)
├── configs/            ← YAML configuration files
├── data/
│   ├── sample/         ← small offline datasets (committed to repo)
│   ├── raw/            ← API responses (gitignored)
│   └── processed/      ← cleaned datasets built by scripts (gitignored)
└── tests/              ← pytest test suite for src/ modules
```

**The workflow for each topic:**
1. Open the notebook (`notebooks/XX/YY.ipynb`).
2. Read the companion guide (`docs/guides/XX/YY.md`) for the math, assumptions, and deeper context.
3. Work through the notebook's TODO cells, using the guide and cheatsheets as reference.
4. Run the checkpoint cells to validate your work.

### Environment variables and API keys

This project uses two external APIs:

| API | Env var | What it provides | Required? |
|---|---|---|---|
| **FRED** (Federal Reserve Economic Data) | `FRED_API_KEY` | Macro time series (GDP, unemployment, yield curve, etc.) | Yes for live fetches; sample data works offline |
| **U.S. Census ACS** | `CENSUS_API_KEY` | County-level demographics (population, income, insurance, poverty) | Optional; many endpoints work without a key |

**How to set them:**

```bash
# In your shell profile (~/.bashrc, ~/.zshrc, etc.):
export FRED_API_KEY="your-key-here"
export CENSUS_API_KEY="your-key-here"
```

After setting a key, restart your Jupyter kernel so Python sees it. In a notebook:

```python
import os
fred_key = os.getenv("FRED_API_KEY")
print("FRED key set?", fred_key is not None)
# Print at most the first 4 characters to verify without exposing the full key
if fred_key:
    print("Starts with:", fred_key[:4])
```

**Security rule:** Never print full API keys or commit them to version control. Use `.env` files (gitignored) or shell environment variables.

### Sample vs processed data: the offline-first pattern

Every data notebook follows the same pattern:

```python
path = PROCESSED_DIR / "macro_quarterly.csv"
if path.exists():
    df = pd.read_csv(path, index_col=0, parse_dates=True)
else:
    df = pd.read_csv(SAMPLE_DIR / "macro_quarterly_sample.csv", index_col=0, parse_dates=True)
```

**Why:** This keeps notebooks runnable without network access. The first time you work through the project, you will use sample data. Once you run the data-building scripts (covered in Module 01), real pipeline outputs will be saved to `data/processed/` and subsequent notebooks will use those instead.

**What is different between sample and processed data:**
- Sample files are small subsets (~100 rows) committed to the repo.
- Processed files are full datasets (~200-300 rows for quarterly, ~3,200 rows for county-level) built by `scripts/build_datasets.py`. They are gitignored because they depend on API access.

### Data validation patterns

You will repeat these checks in every notebook. Build the habit here:

**1. Index checks**
```python
assert isinstance(df.index, pd.DatetimeIndex), "Expected DatetimeIndex"
assert df.index.is_monotonic_increasing, "Index must be sorted ascending"
```
These protect you from silent alignment bugs when merging or lagging.

**2. Shape checks**
```python
assert df.shape[0] > 20, "Too few rows — did the load fail?"
assert df.shape[1] >= 3, "Too few columns — check column selection"
```

**3. Column existence**
```python
expected = ["gdp_growth_qoq", "recession", "target_recession_next_q"]
missing = [c for c in expected if c not in df.columns]
assert not missing, f"Missing columns: {missing}"
```

**4. Missingness summary**
```python
print(df.isna().sum().sort_values(ascending=False).head(10))
```
Some NaNs are expected (e.g., the first few rows after creating lag features). But unexpected NaNs often indicate a join or resample error.

**5. Dtype checks**
```python
# Numeric columns should not be object/string
for col in ["gdp_growth_qoq", "UNRATE"]:
    assert df[col].dtype in [float, "float64", "int64"], f"{col} has wrong dtype: {df[col].dtype}"
```

### How to read a notebook in this project

Every notebook follows a consistent structure:

| Section | What it does | Your job |
|---|---|---|
| **Environment Bootstrap** | Sets up `PROJECT_ROOT`, `DATA_DIR`, etc. | Run the cell, confirm the path is correct |
| **Primer** | Teaches a prerequisite skill (pandas, statsmodels, etc.) | Read and understand; code examples are for reference |
| **Topic sections** (the main body) | Each has a goal, "Your Turn" code cells with TODOs, and a checkpoint | Fill in the `...` placeholders, run the cell, verify the checkpoint |
| **Checkpoint (Self-Check)** | Assertions and a prompt to write 2-3 sentences | Run the assertions; write the interpretation |
| **Extensions (Optional)** | Stretch exercises | Do these if you want more practice |
| **Reflection** | Prompts about assumptions and limitations | Write brief answers — this builds the critical thinking habit |
| **Solutions (Reference)** | Collapsed reference implementations | Try first, then compare |

### Project Code Map
- `src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)
- `src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`, `add_diff_features`, `add_log_diff_features`, `drop_na_rows`)
- `src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)
- `src/econometrics.py`: regression wrappers (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`, `vif_table`)
- `src/fred_api.py`: FRED API client with retry and caching
- `src/health_data.py`: CMS/Socrata API client for health economics datasets

### Exercises

- [ ] Run the bootstrap cell and confirm `PROJECT_ROOT` points to the repo root.
- [ ] Read `FRED_API_KEY` and `CENSUS_API_KEY` from the environment. Print whether each is set without exposing the full key.
- [ ] Create `data/raw/` and `data/processed/` directories using `Path.mkdir(parents=True, exist_ok=True)`.
- [ ] Load `data/sample/macro_quarterly_sample.csv` with `index_col=0, parse_dates=True`.
- [ ] Write at least 4 assertions: index type, monotonicity, minimum row count, and at least one expected column name.
- [ ] Print a missingness summary and identify which columns have NaNs and why (hint: lag features at the start of the series).

<a id="summary"></a>
## Summary + Suggested Readings

You now have a working environment with verified API keys, sample data loaded, and a set of validation patterns you will reuse in every notebook.

**What comes next:** The remaining foundations notebooks introduce the two concepts that govern everything in this project:
- `01_time_series_basics` — time ordering, resampling, lags, rolling windows, and the leakage problem.
- `02_stats_basics_for_ml` — hypothesis testing, confidence intervals, multicollinearity, and the bias-variance tradeoff.

Suggested readings:
- pandas documentation: `pd.read_csv`, `DatetimeIndex`, `Path` objects
- Python `pathlib` module documentation
- Python `os.getenv` documentation
