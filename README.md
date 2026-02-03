# Economic Indicators + Statistics/ML Master Tutorial

Notebook-first, hands-on curriculum using **real APIs** (FRED + US Census/ACS) to learn:
- macro + micro economic concepts
- statistics + inference (including robust standard errors)
- ML modeling (regression, classification, unsupervised)
- evaluation for time series (walk-forward, leakage prevention)
- basic model ops (reproducible runs, saved artifacts)
- a final capstone (report + Streamlit dashboard)

Most notebooks are **markdown-heavy** with TODOs. You are expected to write code yourself (with Codex as a helper).

## Setup

Prereqs:
- Python 3.10+
- FRED API key (required for real data): `FRED_API_KEY`
- Census API key (optional): `CENSUS_API_KEY`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export FRED_API_KEY="your_fred_key"
# optional:
export CENSUS_API_KEY="your_census_key"
```

Optional (tests):
```bash
pip install -r requirements-dev.txt
pytest
```

## Start Here

Open `docs/index.md` for the recommended learning path, links to notebooks, and per-notebook guides.

## Repo Layout
- `notebooks/`: curriculum notebooks (grouped by topic)
- `docs/index.md`: navigation hub
- `docs/guides/`: one deep guide per notebook (mirrors `notebooks/` subfolders)
- `src/`: reusable utilities (API clients, feature engineering, econometrics helpers)
- `scripts/`: CLI pipeline (fetch/build/train/predict) that writes to `outputs/`
- `configs/`: YAML configs for the CLI pipeline
- `outputs/`: run artifacts (models, metrics, predictions) (gitignored)
- `apps/streamlit_app.py`: capstone dashboard
- `reports/`: capstone report template/output

## Capstone Dashboard
```bash
streamlit run apps/streamlit_app.py
```

## Notes
- `data/raw/` and `data/processed/` are gitignored by default.
- `data/sample/` includes small offline datasets so you can practice without API calls.
- Each notebook ends with a collapsed **Solutions (Reference)** section so you can self-check without losing the hands-on flow.
