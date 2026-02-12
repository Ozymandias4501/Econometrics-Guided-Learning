# Econometrics Guided Learning

Hands-on, notebook-first curriculum that teaches statistics, econometrics, and ML using real data from the **FRED** and **US Census** APIs. Notebooks are markdown-heavy with intentional TODOs — you write the code, then check against collapsed solutions.

## Pick Your Starting Point

| Your background | Start at | Path |
|---|---|---|
| **Rusty on stats** — need a refresher on distributions, hypothesis testing, confidence intervals | **00a Statistics Primer** | `notebooks/00a_statistics_primer/` (9 notebooks) |
| **Comfortable with stats** — ready for applied econometrics and ML | **00b Foundations** | `notebooks/00b_foundations/` then `01_data/` onward |
| **Experienced** — want causal inference, time-series econ, or the capstone | **Jump ahead** | `notebooks/06_causal/`, `07_time_series_econ/`, or `08_capstone/` |

The full recommended sequence is in [`docs/index.md`](docs/index.md).

## How It Works

### Notebooks are the primary workspace

Each `.ipynb` is markdown-heavy: it explains the concept, then gives you incomplete code cells marked with `TODO`. You fill in the code, run the cell, and interpret the output. Every notebook ends with a collapsed **Solutions (Reference)** section you can expand to self-check.

### Docs mirror notebooks 1:1

Every notebook has a matching guide in `docs/guides/` with the **same name** (`.ipynb` → `.md`):

```
notebooks/02_regression/05_regularization_ridge_lasso.ipynb   ← hands-on work
docs/guides/02_regression/05_regularization_ridge_lasso.md    ← math, assumptions, deeper context
```

The notebooks keep you moving; the guides are the reference when you want the *why* behind a method. You don't need to read guides front-to-back — open them when a notebook concept needs more depth.

### Offline-first data pattern

Notebooks try to load from `data/processed/` (real pipeline output) and fall back to `data/sample/` (bundled small datasets). This means you can work through the entire curriculum without an internet connection or API keys — live data just makes it richer.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### API Keys

Two free API keys power the live-data notebooks:

| API | Env var | Where to get it |
|---|---|---|
| **FRED** (macro time series) | `FRED_API_KEY` | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) |
| **US Census ACS** (county demographics) | `CENSUS_API_KEY` | [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html) |

FRED is required for live data fetches; Census is optional. You can still work through every notebook offline using the bundled `data/sample/` datasets.

### Providing Your Keys

**Option A — Export in your shell** (quick, ephemeral)
```bash
export FRED_API_KEY="your-key"
export CENSUS_API_KEY="your-key"   # optional
```
Pros: nothing to manage, nothing to accidentally commit.
Cons: you re-type it every new terminal session (unless you add it to `~/.bashrc`).

**Option B — `.env` file** (persistent, project-scoped)

Create a `.env` file in the repo root (it is already gitignored):
```
FRED_API_KEY=your-key
CENSUS_API_KEY=your-key
```
Then load it in Python with `python-dotenv`, or source it before launching Jupyter:
```bash
set -a && source .env && set +a
jupyter notebook
```
Pros: keys persist across sessions and stay scoped to this project.
Cons: one more file to protect — never commit it or share it.

## Repo Layout

```
notebooks/          Curriculum notebooks grouped by topic
docs/index.md       Navigation hub and recommended learning path
docs/guides/        One deep-dive guide per notebook
src/                Reusable utilities (API clients, feature engineering, econ helpers)
scripts/            CLI pipeline (fetch / build / train / predict) → outputs/
configs/            YAML configs for the CLI pipeline
data/sample/        Small offline datasets (always available)
data/raw/           Raw API downloads (gitignored)
data/processed/     Pipeline output (gitignored)
apps/               Capstone Streamlit dashboard
reports/            Capstone report template
```

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Capstone Dashboard

```bash
streamlit run apps/streamlit_app.py
```
