# Econometrics Guided Learning

Hands-on, notebook-first curriculum that takes a rusty econ major from first-principles statistics to OLS regression with HAC robust inference, tree-based ML (Random Forest + XGBoost), structural-break-aware forecasting, and modern causal inference (DAGs + DiD) on real macro data from the **FRED** API. Notebooks are markdown-heavy with intentional TODOs — you write the code, then check against collapsed solutions.

## Pick Your Starting Point

| Your background | Start at | Path |
|---|---|---|
| **Rusty on stats** — need a refresher on distributions, hypothesis testing, confidence intervals, RMSE/R²/F-stats | **00a Statistics Primer** | `notebooks/00a_statistics_primer/` (9 notebooks) |
| **Comfortable with stats** — ready for applied econometrics + ML | **00b Foundations → 01 Data → 01b Diagnostics → 02 Regression → 02b ML → 03 Classification → 04 Forecasting → 05 Causal** | start at `notebooks/00b_foundations/` |

The full sequence is in [`docs/index.md`](docs/index.md).

## Curriculum Map

Nine sections, ~37 notebooks:

| # | Section | What you learn |
|---|---|---|
| 1 | **00a Statistics Primer** | Descriptive stats, distributions, sampling/CLT, CIs, hypothesis testing, correlation, and a metrics primer that maps RMSE / R² / F-stat / accuracy / precision / recall / F1 / AUC to where they appear in OLS, sklearn, and xgboost output. |
| 2 | **00b Foundations** | Project setup and time-series indexing patterns. |
| 3 | **01 Data** | FRED API + caching; building a quarterly macro panel; recession label. |
| 4 | **01b Time Series Diagnostics** | Stationarity intuition, the spurious-regression trap, ADF and KPSS tests, the differencing decision, cointegration in one paragraph. |
| 5 | **02 Regression** | OLS single- and multi-factor (micro and macro); functional forms and interactions; residual diagnostics; HAC (Newey–West) robust inference; Ridge/Lasso regularization; rolling-window stability. |
| 6 | **02b ML Regression** | Same target as section 5 (next-quarter GDP growth) with `RandomForestRegressor` and `XGBRegressor`. Walk-forward CV, hyperparameter tuning, permutation importance, and a final OLS-vs-RF-vs-XGBoost comparison table. |
| 7 | **03 Classification** | Recession prediction with logistic regression and tree/XGBoost classifiers. Confusion matrix, accuracy, precision, recall, F1, ROC-AUC, calibration, walk-forward validation. |
| 8 | **04 Honest Forecasting** | Pseudo-out-of-sample walk-forward backtests across OLS / RF / XGBoost on the same target; rolling error plots; mean ensembles; structural-break detection (Chow); the COVID problem and three mitigation strategies. |
| 9 | **05 Causal Inference** | Predictive vs causal estimands; DAGs (chain, fork, collider); omitted-variable bias by simulation; the collider trap and a control taxonomy; difference-in-differences with parallel-trends checks and clustered standard errors. |

## How It Works

### Notebooks are the primary workspace

Each `.ipynb` is markdown-heavy: it explains the concept, then gives you incomplete code cells marked with `TODO`. You fill in the code, run the cell, and interpret the output. Every notebook ends with a collapsed **Solutions (Reference)** section you can expand to self-check.

### Docs mirror notebooks 1:1

Every notebook has a matching guide in `docs/guides/` with the **same name** (`.ipynb` → `.md`):

```
notebooks/02_regression/05_regularization_ridge_lasso.ipynb   ← hands-on work
docs/guides/02_regression/05_regularization_ridge_lasso.md    ← math, assumptions, deeper context
```

The notebooks keep you moving; the guides are the reference when you want the *why* behind a method.

### Offline-first data pattern

Notebooks try to load from `data/processed/` (real pipeline output) and fall back to `data/sample/` (bundled small datasets). You can work through the entire curriculum without an internet connection or API key — live FRED data just makes it richer.

## Setup

This project uses [**uv**](https://docs.astral.sh/uv/) to manage the Python toolchain and dependencies. Install uv once (`brew install uv` on macOS, or see the docs for other platforms), then:

```bash
uv sync                  # creates .venv and installs runtime + dev deps from uv.lock
uv run jupyter notebook  # or: make notebook
```

`uv run <cmd>` executes any command inside the project's locked environment without you having to `activate` anything.

### API Key (optional)

| API | Env var | Where to get it |
|---|---|---|
| **FRED** (macro time series) | `FRED_API_KEY` | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) |

FRED is free. With a key you can fetch live data via `make fetch-fred`. Without one, every notebook still runs against the bundled `data/sample/macro_quarterly_sample.csv`.

### Providing your key

Export the key from your **shell profile** so every shell, Jupyter kernel, and editor picks it up automatically. Add to `~/.zshrc` (macOS default) or `~/.bashrc` (Linux):

```bash
export FRED_API_KEY="your-key"
```

Then `source ~/.zshrc` (or open a fresh terminal) and confirm with `echo $FRED_API_KEY`. Python reads it via `os.getenv("FRED_API_KEY")`. No `.env` file, no per-project secret config — one canonical place for credentials.

## Repo Layout

```
notebooks/          Curriculum notebooks grouped by topic
docs/index.md       Navigation hub and recommended learning path
docs/guides/        One deep-dive guide per notebook
docs/cheatsheets/   One-page references (regression diagnostics, metrics, etc.)
src/                Reusable utilities (FRED API, feature engineering, econ helpers)
scripts/            FRED fetch CLI
data/sample/        Small offline datasets (always available)
data/raw/           Raw API downloads (gitignored)
data/processed/     Pipeline output (gitignored)
```

## Running Tests

```bash
uv run pytest        # or: make test
```

`uv sync` already installs the `dev` dependency group (which includes pytest); no separate install step is needed.
