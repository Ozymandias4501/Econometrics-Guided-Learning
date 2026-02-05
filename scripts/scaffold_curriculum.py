"""Scaffold curriculum notebooks and per-notebook guides.

This script writes the tutorial structure described in docs/index.md:
- 35 notebooks under notebooks/
- 35 guides under docs/guides/

It is safe to re-run; it will overwrite files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class NotebookSpec:
    path: str
    title: str
    summary: str
    sections: List[str]


TEMPLATES_ROOT = Path(__file__).resolve().parent / "templates"


def load_template(rel_path: str) -> str:
    """Load a template file from scripts/templates/.

    We keep long-form educational content in template files so this generator remains maintainable.
    """

    path = TEMPLATES_ROOT / rel_path
    return path.read_text()


def render_template(text: str, mapping: dict[str, str]) -> str:
    """Render a template with {{TOKEN}} placeholders.

    Notes:
    - We intentionally avoid Python's .format() to prevent conflicts with LaTeX braces.
    - This is simple string substitution; keep tokens unique and uppercase.
    """

    out = text
    for k, v in mapping.items():
        out = out.replace(f"{{{{{k}}}}}", v)
    return out


def concept(name: str) -> str:
    """Load a guide concept block from scripts/templates/guides/concepts/."""

    return load_template(f"guides/concepts/{name}.md").rstrip()


def primer(name: str) -> str:
    """Load a notebook primer block from scripts/templates/notebooks/primers/."""

    return load_template(f"notebooks/primers/{name}.md").rstrip()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in src.split("\n")],
    }


def nb(cells: List[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "python3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def slugify(text: str) -> str:
    """Create a stable anchor slug for markdown links."""

    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "section"


BOOTSTRAP = """from __future__ import annotations

from pathlib import Path
import sys


def find_repo_root(start: Path) -> Path:
    p = start
    for _ in range(8):
        if (p / 'src').exists() and (p / 'docs').exists():
            return p
        p = p.parent
    raise RuntimeError('Could not find repo root. Start Jupyter from the repo root.')


PROJECT_ROOT = find_repo_root(Path.cwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
SAMPLE_DIR = DATA_DIR / 'sample'

PROJECT_ROOT
"""


def notebook_front_matter(stem: str, category: str, *, guide_path: str) -> str:
    """Richer notebook framing: why it matters, deliverables, success criteria, pitfalls."""

    # Defaults (used if a notebook doesn't have a concrete file deliverable).
    deliverables: list[str] = []
    prereqs: list[str] = []
    success: list[str] = [
        "You can explain what you built and why each step exists.",
        "You can run your work end-to-end without undefined variables.",
    ]
    pitfalls: list[str] = [
        "Running cells top-to-bottom without reading the instructions.",
        "Leaving `...` placeholders in code cells.",
    ]
    quick_fixes: list[str] = [
        "If you see `ModuleNotFoundError`, re-run the bootstrap cell and restart the kernel; make sure `PROJECT_ROOT` is the repo root.",
        "If a `data/processed/*` file is missing, either run the matching build script (see guide) or use the notebook’s `data/sample/*` fallback.",
        "If results look “too good,” suspect leakage; re-check shifts, rolling windows, and time splits.",
        "If a model errors, check dtypes (`astype(float)`) and missingness (`dropna()` on required columns).",
    ]

    why = "This notebook builds a core piece of the project."

    if category == "00_foundations":
        prereqs = [
            "Comfort with basic Python + pandas (reading CSVs, making plots).",
            "Willingness to write short interpretation notes as you go.",
        ]
        why = (
            "Foundations notebooks build the intuition that prevents the most common mistakes in economic ML:\n"
            "- leaking future information,\n"
            "- evaluating with the wrong split strategy,\n"
            "- over-interpreting coefficients.\n"
        )
        pitfalls += [
            "Using random splits on time series.",
            "Assuming correlation implies causation.",
        ]

    if category == "01_data":
        prereqs = [
            "Completed Part 00 (foundations) or equivalent time-series basics.",
            "FRED API key set (`FRED_API_KEY`) for real data (sample data works offline).",
        ]
        why = (
            "Data notebooks build the datasets used everywhere else. If these steps are wrong, every model result is suspect.\n"
            "You will practice:\n"
            "- API ingestion and caching,\n"
            "- frequency alignment,\n"
            "- label construction.\n"
        )
        pitfalls += [
            "Merging mixed-frequency series without explicit resampling/aggregation.",
            "Forgetting to shift targets for forecasting tasks.",
        ]

    if category == "02_regression":
        prereqs = [
            "Completed Parts 00–01 (foundations + data).",
            "Basic algebra comfort (reading coefficient tables, units).",
        ]
        why = (
            "Regression is the bridge between statistics and ML. You will learn:\n"
            "- single-factor vs multi-factor interpretation,\n"
            "- robust standard errors,\n"
            "- coefficient stability and multicollinearity.\n"
        )
        pitfalls += [
            "Treating coefficients as causal without a causal design.",
            "Ignoring multicollinearity (unstable coefficients).",
        ]

    if category == "03_classification":
        prereqs = [
            "Completed Part 02 (regression basics) or equivalent.",
            "Comfort interpreting probabilities and trade-offs (false positives vs false negatives).",
        ]
        why = (
            "Classification notebooks turn the recession label into a **probability model**.\n"
            "You will learn how to evaluate rare-event prediction and how to choose thresholds intentionally.\n"
        )
        pitfalls += [
            "Reporting only accuracy on imbalanced data.",
            "Using threshold=0.5 by default without considering costs.",
        ]

    if category == "04_unsupervised":
        prereqs = [
            "Completed Part 01 (macro panel) or equivalent.",
            "Comfort with standardization and basic linear algebra intuition (variance, distance).",
        ]
        why = (
            "Unsupervised notebooks help you understand macro structure:\n"
            "- latent factors (PCA),\n"
            "- regimes (clustering),\n"
            "- anomalies/crises.\n"
        )
        pitfalls += [
            "Forgetting to standardize features before PCA/clustering.",
            "Over-interpreting clusters as causal regimes.",
        ]

    if category == "05_model_ops":
        prereqs = [
            "Completed earlier modeling notebooks (regression/classification).",
            "Comfort running scripts and inspecting files under `outputs/`.",
        ]
        why = (
            "Model ops notebooks turn your work into reproducible runs with saved artifacts.\n"
            "The goal is: someone else can run your pipeline and see the same metrics.\n"
        )
        pitfalls += [
            "Not recording which dataset/config a model was trained on.",
            "Overwriting artifacts without run IDs.",
        ]

    if category == "06_capstone":
        prereqs = [
            "Completed (or at least attempted) Parts 00–05.",
            "Willingness to write a report with assumptions + limitations.",
        ]
        why = (
            "Capstone notebooks integrate everything:\n"
            "- data pipeline,\n"
            "- modeling + evaluation,\n"
            "- interpretation + limitations,\n"
            "- reproducible artifacts,\n"
            "- report + dashboard.\n"
        )

    if category == "07_causal":
        prereqs = [
            "Completed Part 02 (regression + robust SE).",
            "Basic familiarity with panels (same unit over time) and the idea of identification assumptions.",
        ]
        why = (
            "Causal notebooks focus on **identification**: what would have to be true for a coefficient to represent a causal effect.\n"
            "You will practice:\n"
            "- building a county-year panel,\n"
            "- fixed effects (TWFE),\n"
            "- clustered standard errors,\n"
            "- DiD + event studies,\n"
            "- IV/2SLS.\n"
        )
        pitfalls += [
            "Treating regression output as causal without stating identification assumptions.",
            "Using non-clustered SE when shocks are correlated within groups (e.g., states).",
        ]

    if category == "08_time_series_econ":
        prereqs = [
            "Completed Part 01 macro panel notebooks (or have `panel_monthly.csv` / sample available).",
            "Comfort with differencing/log transforms and reading time series plots.",
        ]
        why = (
            "Time-series econometrics notebooks build the classical toolkit you need before trusting macro regressions:\n"
            "- stationarity + unit roots,\n"
            "- cointegration + error correction,\n"
            "- VAR dynamics and impulse responses.\n"
        )
        pitfalls += [
            "Running tests without plotting or transforming the series first.",
            "Treating impulse responses as structural causality without an identification story.",
        ]

    # Concrete deliverables (where applicable)
    if stem == "01_build_macro_monthly_panel":
        deliverables = ["data/processed/panel_monthly.csv"]
    elif stem == "02_gdp_growth_and_recession_label":
        deliverables = ["data/processed/gdp_quarterly.csv"]
    elif stem == "03_build_macro_quarterly_features":
        deliverables = ["data/processed/macro_quarterly.csv"]
    elif stem == "04_census_api_microdata_fetch":
        deliverables = ["data/processed/census_county_<year>.csv"]
    elif stem == "00_build_census_county_panel":
        deliverables = ["data/processed/census_county_panel.csv"]
    elif stem == "02_build_cli_train_predict":
        deliverables = ["outputs/<run_id>/model.joblib", "outputs/<run_id>/metrics.json", "outputs/<run_id>/predictions.csv"]
    elif stem == "01_capstone_workspace":
        deliverables = ["reports/capstone_report.md", "apps/streamlit_app.py (running)"]

    if deliverables:
        success = [
            *success,
            "You can point to the concrete deliverable(s) listed below and explain how they were produced.",
        ]

    def bullets(items: list[str]) -> str:
        return "\n".join([f"- {x}" for x in items]) if items else "- (none)"

    deliverable_text = bullets(deliverables) if deliverables else "- (no file output; learning/analysis notebook)"

    return (
        "## Why This Notebook Matters\n"
        f"{why}\n"
        "\n"
        "## Prerequisites (Quick Self-Check)\n"
        f"{bullets(prereqs)}\n"
        "\n"
        "## What You Will Produce\n"
        f"{deliverable_text}\n"
        "\n"
        "## Success Criteria\n"
        f"{bullets(success)}\n"
        "\n"
        "## Common Pitfalls\n"
        f"{bullets(pitfalls)}\n"
        "\n"
        "## Quick Fixes (When You Get Stuck)\n"
        f"{bullets(quick_fixes)}\n"
        "\n"
        "## Matching Guide\n"
        f"- `{guide_path}`\n"
    )


def notebook_checkpoint_snippet(stem: str, category: str) -> str:
    """A small TODO-driven checkpoint that helps learners validate progress."""

    if category == "00_foundations":
        return (
            "# TODO: Run a quick sanity check on any DataFrame/Series you created in this notebook.\n"
            "# Example (adjust variable names):\n"
            "# assert df.index.is_monotonic_increasing\n"
            "# assert df.isna().sum().sum() == 0\n"
            "#\n"
            "# TODO: Write 2-3 sentences:\n"
            "# - What would leakage look like in YOUR code?\n"
            "# - How would you detect it?\n"
            "...\n"
        )

    if category == "01_data":
        expected = {
            "01_build_macro_monthly_panel": "panel_monthly.csv",
            "02_gdp_growth_and_recession_label": "gdp_quarterly.csv",
            "03_build_macro_quarterly_features": "macro_quarterly.csv",
            "04_census_api_microdata_fetch": "census_county_<year>.csv",
        }.get(stem)

        hint = (
            f"# Expected file: data/processed/{expected}\n" if expected else "# Expected file: (see notebook front matter)\n"
        )

        return (
            "import pandas as pd\n"
            "\n"
            + hint
            + "# TODO: After saving your processed dataset, load it and run checks.\n"
            "# df = pd.read_csv(PROCESSED_DIR / 'your_file.csv', index_col=0, parse_dates=True)\n"
            "# assert df.index.is_monotonic_increasing\n"
            "# assert df.shape[0] > 20\n"
            "# print(df.dtypes)\n"
            "...\n"
        )

    if category in {"02_regression", "03_classification"}:
        return (
            "# TODO: After you build X/y and split by time, validate the split.\n"
            "# Example (adjust variable names):\n"
            "# assert X_train.index.max() < X_test.index.min()\n"
            "# assert y_train.index.equals(X_train.index)\n"
            "# assert y_test.index.equals(X_test.index)\n"
            "# assert not X_train.isna().any().any()\n"
            "# assert not X_test.isna().any().any()\n"
            "...\n"
        )

    if category == "04_unsupervised":
        return (
            "# TODO: Confirm your feature matrix is standardized (or justify why not).\n"
            "# Example:\n"
            "# assert abs(X_scaled.mean(axis=0)).max() < 1e-6\n"
            "# assert abs(X_scaled.std(axis=0) - 1).max() < 1e-6\n"
            "...\n"
        )

    if category == "05_model_ops":
        return (
            "# TODO: Run one script end-to-end and confirm an artifact bundle exists.\n"
            "# Example:\n"
            "# - list outputs/ and pick the newest run_id\n"
            "# - assert model.joblib and metrics.json exist\n"
            "...\n"
        )

    if category == "06_capstone":
        return (
            "# TODO: Confirm your capstone deliverables exist.\n"
            "# - reports/capstone_report.md updated\n"
            "# - outputs/<run_id>/ contains model + metrics + predictions\n"
            "# - streamlit app runs and loads artifacts\n"
            "...\n"
        )

    if category == "07_causal":
        expected = {
            "00_build_census_county_panel": "census_county_panel.csv",
        }.get(stem)

        hint = (
            f"# Expected file: data/processed/{expected}\n" if expected else "# Expected output: (see notebook front matter)\n"
        )

        return (
            "import pandas as pd\n"
            "\n"
            + hint
            + "# TODO: If you created a panel DataFrame, verify the indexing + core columns.\n"
            "# Example (adjust variable names):\n"
            "# assert isinstance(panel.index, pd.MultiIndex)\n"
            "# assert panel.index.names[:2] == ['fips', 'year']\n"
            "# assert panel['year'].astype(int).between(1900, 2100).all()\n"
            "# assert panel['fips'].astype(str).str.len().eq(5).all()\n"
            "#\n"
            "# TODO: Write 2-3 sentences:\n"
            "# - What is the identification assumption for your causal estimate?\n"
            "# - What diagnostic/falsification did you run?\n"
            "...\n"
        )

    if category == "08_time_series_econ":
        return (
            "import pandas as pd\n"
            "\n"
            "# TODO: Validate your time series table is well-formed.\n"
            "# Example (adjust variable names):\n"
            "# assert isinstance(df.index, pd.DatetimeIndex)\n"
            "# assert df.index.is_monotonic_increasing\n"
            "# assert df.shape[0] > 30\n"
            "#\n"
            "# TODO: If you built transformed series (diff/logdiff), confirm no future leakage.\n"
            "# Hint: transformations should only use past/current values (shift/diff), never future.\n"
            "...\n"
        )

    return "...\n"


def solution_snippets(stem: str) -> dict[str, str]:
    """Reference solutions per notebook section.

    Notes:
    - Solutions default to offline sample datasets under data/sample/.
    - Where relevant, solutions include an optional branch for API keys.
    - These are reference implementations; learners should attempt TODOs first.
    """

    if stem == "00_setup":
        return {
            "Environment bootstrap": "print(PROJECT_ROOT)\nprint(DATA_DIR)\n",
            "Verify API keys": (
                "import os\n"
                "print('FRED_API_KEY set?', bool(os.getenv('FRED_API_KEY')))\n"
                "print('CENSUS_API_KEY set?', bool(os.getenv('CENSUS_API_KEY')))\n"
            ),
            "Load sample data": (
                "import pandas as pd\n"
                "df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "df.head()\n"
            ),
            "Checkpoints": (
                "assert df.index.is_monotonic_increasing\n"
                "assert 'target_recession_next_q' in df.columns\n"
                "print('ok')\n"
            ),
        }

    if stem == "01_time_series_basics":
        return {
            "Toy series": (
                "import numpy as np\n"
                "import pandas as pd\n"
                "\n"
                "rng = np.random.default_rng(0)\n"
                "idx = pd.date_range('2020-01-01', periods=365, freq='D')\n"
                "y = pd.Series(0.01*np.arange(len(idx)) + rng.normal(0, 1, len(idx)), index=idx, name='y')\n"
                "y.head()\n"
            ),
            "Resampling": (
                "monthly_mean = y.resample('ME').mean()\n"
                "monthly_last = y.resample('ME').last()\n"
                "monthly_mean.head(), monthly_last.head()\n"
            ),
            "Lag and rolling features": (
                "df = y.to_frame()\n"
                "df['y_lag1'] = df['y'].shift(1)\n"
                "df['y_lag7'] = df['y'].shift(7)\n"
                "df['y_roll14_mean'] = df['y'].rolling(14).mean()\n"
                "df = df.dropna()\n"
                "df.head()\n"
            ),
            "Leakage demo": (
                "from sklearn.model_selection import train_test_split\n"
                "from sklearn.linear_model import LinearRegression\n"
                "from sklearn.metrics import mean_squared_error\n"
                "\n"
                "tmp = df.copy()\n"
                "tmp['target_tomorrow'] = tmp['y'].shift(-1)\n"
                "tmp = tmp.dropna()\n"
                "\n"
                "X = tmp[['y_lag1', 'y_lag7', 'y_roll14_mean']]\n"
                "y_t = tmp['target_tomorrow']\n"
                "\n"
                "# Random split (not recommended for time series)\n"
                "X_tr, X_te, y_tr, y_te = train_test_split(X, y_t, test_size=0.2, shuffle=True, random_state=0)\n"
                "m = LinearRegression().fit(X_tr, y_tr)\n"
                "rmse_rand = mean_squared_error(y_te, m.predict(X_te), squared=False)\n"
                "\n"
                "# Time split (recommended)\n"
                "split = int(len(tmp) * 0.8)\n"
                "m2 = LinearRegression().fit(X.iloc[:split], y_t.iloc[:split])\n"
                "rmse_time = mean_squared_error(y_t.iloc[split:], m2.predict(X.iloc[split:]), squared=False)\n"
                "\n"
                "rmse_rand, rmse_time\n"
            ),
        }

    if stem == "02_stats_basics_for_ml":
        return {
            "Correlation vs causation": (
                "import numpy as np\n"
                "import pandas as pd\n"
                "\n"
                "rng = np.random.default_rng(0)\n"
                "n = 800\n"
                "# Hidden confounder\n"
                "z = rng.normal(size=n)\n"
                "\n"
                "# Two observed variables both driven by z\n"
                "x = z + rng.normal(scale=0.8, size=n)\n"
                "w = z + rng.normal(scale=0.8, size=n)\n"
                "\n"
                "# Target driven by z (not by x directly)\n"
                "y = 2.0 * z + rng.normal(scale=1.0, size=n)\n"
                "\n"
                "df = pd.DataFrame({'z': z, 'x': x, 'w': w, 'y': y})\n"
                "df.corr(numeric_only=True)\n"
            ),
            "Multicollinearity (VIF)": (
                "import statsmodels.api as sm\n"
                "from src.econometrics import vif_table\n"
                "\n"
                "# Build highly correlated predictors\n"
                "rng = np.random.default_rng(1)\n"
                "n = 600\n"
                "x1 = rng.normal(size=n)\n"
                "x2 = 0.95 * x1 + rng.normal(scale=0.2, size=n)\n"
                "y = 1.0 + 2.0 * x1 + rng.normal(scale=1.0, size=n)\n"
                "df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})\n"
                "\n"
                "vif_table(df, ['x1', 'x2'])\n"
                "\n"
                "X = sm.add_constant(df[['x1', 'x2']])\n"
                "res = sm.OLS(df['y'], X).fit()\n"
                "print(res.summary())\n"
            ),
            "Bias/variance": (
                "from sklearn.linear_model import LinearRegression\n"
                "from sklearn.tree import DecisionTreeRegressor\n"
                "from sklearn.metrics import mean_squared_error\n"
                "\n"
                "# Simple 1D regression problem\n"
                "rng = np.random.default_rng(2)\n"
                "n = 400\n"
                "x = np.linspace(-3, 3, n)\n"
                "y = np.sin(x) + rng.normal(scale=0.2, size=n)\n"
                "\n"
                "X = x.reshape(-1, 1)\n"
                "split = int(n * 0.8)\n"
                "\n"
                "lin = LinearRegression().fit(X[:split], y[:split])\n"
                "tree = DecisionTreeRegressor(random_state=0).fit(X[:split], y[:split])\n"
                "\n"
                "rmse_lin_tr = mean_squared_error(y[:split], lin.predict(X[:split]), squared=False)\n"
                "rmse_lin_te = mean_squared_error(y[split:], lin.predict(X[split:]), squared=False)\n"
                "rmse_tree_tr = mean_squared_error(y[:split], tree.predict(X[:split]), squared=False)\n"
                "rmse_tree_te = mean_squared_error(y[split:], tree.predict(X[split:]), squared=False)\n"
                "\n"
                "(\n"
                "    {'linear_train': rmse_lin_tr, 'linear_test': rmse_lin_te},\n"
                "    {'tree_train': rmse_tree_tr, 'tree_test': rmse_tree_te},\n"
                ")\n"
            ),
            "Hypothesis testing": (
                "import numpy as np\n"
                "import pandas as pd\n"
                "import statsmodels.api as sm\n"
                "from scipy import stats\n"
                "\n"
                "# One-sample t-test: mean(x) == 0?\n"
                "rng = np.random.default_rng(3)\n"
                "x = rng.normal(loc=0.1, scale=1.0, size=200)\n"
                "t_stat, p_val = stats.ttest_1samp(x, popmean=0.0)\n"
                "print('t:', t_stat, 'p:', p_val)\n"
                "\n"
                "# Regression coefficient test: slope == 0?\n"
                "rng = np.random.default_rng(4)\n"
                "n = 300\n"
                "x2 = rng.normal(size=n)\n"
                "y2 = 1.0 + 0.5 * x2 + rng.normal(scale=1.0, size=n)\n"
                "\n"
                "df = pd.DataFrame({'y': y2, 'x': x2})\n"
                "X = sm.add_constant(df[['x']], has_constant='add')\n"
                "res = sm.OLS(df['y'], X).fit()\n"
                "\n"
                "print(res.summary())\n"
            ),
        }

    # Data notebooks
    if stem == "00_fred_api_and_caching":
        return {
            "Choose series": (
                "series_ids = ['UNRATE', 'FEDFUNDS', 'CPIAUCSL', 'INDPRO', 'RSAFS', 'T10Y2Y']\n"
                "series_ids\n"
            ),
            "Fetch metadata": (
                "from src import fred_api\n"
                "\n"
                "meta = fred_api.fetch_series_meta('UNRATE')\n"
                "meta\n"
            ),
            "Fetch + cache observations": (
                "from src import data as data_utils\n"
                "from src import fred_api\n"
                "\n"
                "raw_dir = RAW_DIR / 'fred'\n"
                "raw_dir.mkdir(parents=True, exist_ok=True)\n"
                "\n"
                "frames = []\n"
                "for sid in series_ids:\n"
                "    payload = data_utils.load_or_fetch_json(\n"
                "        raw_dir / f'{sid}.json',\n"
                "        lambda sid=sid: fred_api.fetch_series_observations(sid, start_date='1980-01-01', end_date=None),\n"
                "    )\n"
                "    frames.append(fred_api.observations_to_frame(payload, sid))\n"
                "\n"
                "panel = pd.concat(frames, axis=1).sort_index()\n"
                "panel.head()\n"
            ),
            "Fallback to sample": (
                "import pandas as pd\n"
                "\n"
                "panel = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True)\n"
                "panel.head()\n"
            ),
        }

    if stem == "01_build_macro_monthly_panel":
        return {
            "Load series": (
                "import pandas as pd\n"
                "from src import data as data_utils\n"
                "from src import fred_api\n"
                "\n"
                "raw_dir = RAW_DIR / 'fred'\n"
                "series_ids = ['UNRATE', 'FEDFUNDS', 'CPIAUCSL', 'INDPRO', 'RSAFS', 'T10Y2Y']\n"
                "\n"
                "frames = []\n"
                "for sid in series_ids:\n"
                "    payload = data_utils.load_json(raw_dir / f'{sid}.json') if (raw_dir / f'{sid}.json').exists() else None\n"
                "    if payload is None:\n"
                "        # Offline fallback\n"
                "        panel = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True)\n"
                "        break\n"
                "    frames.append(fred_api.observations_to_frame(payload, sid))\n"
                "else:\n"
                "    panel = pd.concat(frames, axis=1).sort_index()\n"
                "\n"
                "panel.head()\n"
            ),
            "Month-end alignment": (
                "# Convert to month-end panel (ME) and forward-fill.\n"
                "panel_monthly = panel.resample('ME').last().ffill()\n"
                "panel_monthly.tail()\n"
            ),
            "Missingness": (
                "missing = panel_monthly.isna().sum().sort_values(ascending=False)\n"
                "missing.head(10)\n"
            ),
            "Save processed panel": (
                "from src import data as data_utils\n"
                "data_utils.save_csv(panel_monthly, PROCESSED_DIR / 'panel_monthly.csv')\n"
                "print('saved', PROCESSED_DIR / 'panel_monthly.csv')\n"
            ),
        }

    if stem == "02_gdp_growth_and_recession_label":
        return {
            "Fetch GDP": (
                "import pandas as pd\n"
                "from src import data as data_utils\n"
                "\n"
                "# Offline default\n"
                "gdp = pd.read_csv(SAMPLE_DIR / 'gdp_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "gdp.head()\n"
            ),
            "Compute growth": (
                "from src import macro\n"
                "\n"
                "levels = gdp['GDPC1']\n"
                "gdp['gdp_growth_qoq'] = macro.gdp_growth_qoq(levels)\n"
                "gdp['gdp_growth_qoq_annualized'] = macro.gdp_growth_qoq_annualized(levels)\n"
                "gdp['gdp_growth_yoy'] = macro.gdp_growth_yoy(levels)\n"
            ),
            "Define recession label": (
                "gdp['recession'] = macro.technical_recession_label(gdp['gdp_growth_qoq'])\n"
            ),
            "Define next-quarter target": (
                "gdp['target_recession_next_q'] = macro.next_period_target(gdp['recession'])\n"
                "gdp = gdp.dropna()\n"
                "gdp.tail()\n"
            ),
        }

    if stem == "03_build_macro_quarterly_features":
        return {
            "Aggregate monthly -> quarterly": (
                "import pandas as pd\n"
                "from src import macro\n"
                "\n"
                "panel_monthly = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True)\n"
                "q_mean = macro.monthly_to_quarterly(panel_monthly, how='mean')\n"
                "q_last = macro.monthly_to_quarterly(panel_monthly, how='last')\n"
                "q_mean.head(), q_last.head()\n"
            ),
            "Add lags": (
                "q = q_mean.copy()\n"
                "for col in q.columns:\n"
                "    for lag in [1, 2, 4]:\n"
                "        q[f'{col}_lag{lag}'] = q[col].shift(lag)\n"
                "q = q.dropna()\n"
            ),
            "Merge with GDP/labels": (
                "gdp = pd.read_csv(SAMPLE_DIR / 'gdp_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "df = q.join(gdp, how='inner').dropna()\n"
                "df.head()\n"
            ),
            "Save macro_quarterly.csv": (
                "from src import data as data_utils\n"
                "data_utils.save_csv(df, PROCESSED_DIR / 'macro_quarterly.csv')\n"
                "print('saved', PROCESSED_DIR / 'macro_quarterly.csv')\n"
            ),
        }

    if stem == "04_census_api_microdata_fetch":
        return {
            "Browse variables": (
                "import json\n"
                "\n"
                "# Offline default\n"
                "print('Open the Census variables metadata in data/raw/census/variables_<year>.json if available.')\n"
            ),
            "Fetch county data": (
                "import pandas as pd\n"
                "\n"
                "# Offline default sample\n"
                "df = pd.read_csv(SAMPLE_DIR / 'census_county_sample.csv')\n"
                "df.head()\n"
            ),
            "Derived rates": (
                "df['unemployment_rate'] = df['B23025_005E'] / df['B23025_002E']\n"
                "df['poverty_rate'] = df['B17001_002E'] / df['B01003_001E']\n"
                "df[['unemployment_rate', 'poverty_rate']].describe()\n"
            ),
            "Save processed data": (
                "from src import data as data_utils\n"
                "year = 2022\n"
                "data_utils.save_csv(df.set_index(['state','county'], drop=False), PROCESSED_DIR / f'census_county_{year}.csv')\n"
                "print('saved')\n"
            ),
        }

    # Regression notebooks (reference approaches)
    if stem == "00_single_factor_regression_micro":
        return {
            "Load census data": (
                "import numpy as np\n"
                "import pandas as pd\n"
                "\n"
                "df = pd.read_csv(SAMPLE_DIR / 'census_county_sample.csv')\n"
                "df.head()\n"
            ),
            "Build log variables": (
                "df = df.copy()\n"
                "df = df[(df['B19013_001E'] > 0) & (df['B25064_001E'] > 0)].copy()\n"
                "df['log_income'] = np.log(df['B19013_001E'].astype(float))\n"
                "df['log_rent'] = np.log(df['B25064_001E'].astype(float))\n"
            ),
            "Fit OLS + HC3": (
                "from src import econometrics\n"
                "\n"
                "res = econometrics.fit_ols_hc3(df, y_col='log_rent', x_cols=['log_income'])\n"
                "print(res.summary())\n"
            ),
            "Interpretation": (
                "# In a log-log model, the slope is an elasticity-style interpretation:\n"
                "# a 1% increase in income is associated with ~beta% increase in rent (under assumptions).\n"
            ),
        }

    if stem == "01_multifactor_regression_micro_controls":
        return {
            "Choose controls": (
                "import numpy as np\n"
                "import pandas as pd\n"
                "from src import econometrics\n"
                "\n"
                "df = pd.read_csv(SAMPLE_DIR / 'census_county_sample.csv')\n"
                "df = df[(df['B19013_001E'] > 0) & (df['B25064_001E'] > 0) & (df['B01003_001E'] > 0)].copy()\n"
                "df['log_income'] = np.log(df['B19013_001E'].astype(float))\n"
                "df['log_rent'] = np.log(df['B25064_001E'].astype(float))\n"
                "df['log_pop'] = np.log(df['B01003_001E'].astype(float))\n"
                "\n"
                "res = econometrics.fit_ols_hc3(df, y_col='log_rent', x_cols=['log_income', 'log_pop', 'poverty_rate'])\n"
                "print(res.summary())\n"
            ),
            "Fit model": "# See above.\n",
            "Compare coefficients": (
                "# Compare single-factor vs multi-factor slopes for log_income.\n"
                "# If it changes a lot, omitted variable bias is plausible.\n"
            ),
        }

    if stem == "02_single_factor_regression_macro":
        return {
            "Load macro data": (
                "import pandas as pd\n"
                "df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "df.head()\n"
            ),
            "Fit OLS": (
                "from src import econometrics\n"
                "\n"
                "res = econometrics.fit_ols(df, y_col='gdp_growth_qoq', x_cols=['T10Y2Y'])\n"
                "print(res.summary())\n"
            ),
            "Fit HAC": (
                "res_hac = econometrics.fit_ols_hac(df, y_col='gdp_growth_qoq', x_cols=['T10Y2Y'], maxlags=2)\n"
                "print(res_hac.summary())\n"
            ),
            "Interpretation": "# Interpret sign/magnitude carefully; time-series inference is fragile.\n",
        }

    if stem == "03_multifactor_regression_macro":
        return {
            "Choose features": (
                "import pandas as pd\n"
                "from src import econometrics\n"
                "\n"
                "df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "x_cols = ['T10Y2Y', 'UNRATE', 'FEDFUNDS']\n"
                "res = econometrics.fit_ols_hac(df, y_col='gdp_growth_qoq', x_cols=x_cols, maxlags=2)\n"
                "print(res.summary())\n"
            ),
            "Fit model": "# See above.\n",
            "VIF + stability": (
                "from src.econometrics import vif_table\n"
                "print(vif_table(df.dropna(), ['T10Y2Y', 'UNRATE', 'FEDFUNDS']))\n"
            ),
        }

    if stem == "04_inference_time_series_hac":
        return {
            "Assumptions": "# OLS assumptions (esp. independent errors) often fail in time series.\n",
            "Autocorrelation": (
                "import pandas as pd\n"
                "import numpy as np\n"
                "\n"
                "df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "y = df['gdp_growth_qoq'].to_numpy()\n"
                "print('lag1 autocorr (rough):', np.corrcoef(y[1:], y[:-1])[0,1])\n"
            ),
            "HAC SE": (
                "from src import econometrics\n"
                "\n"
                "res = econometrics.fit_ols(df, y_col='gdp_growth_qoq', x_cols=['T10Y2Y'])\n"
                "res_hac = econometrics.fit_ols_hac(df, y_col='gdp_growth_qoq', x_cols=['T10Y2Y'], maxlags=2)\n"
                "print('naive SE:', res.bse)\n"
                "print('HAC SE  :', res_hac.bse)\n"
            ),
        }

    if stem == "05_regularization_ridge_lasso":
        return {
            "Build feature matrix": (
                "import pandas as pd\n"
                "import numpy as np\n"
                "\n"
                "from sklearn.model_selection import train_test_split\n"
                "\n"
                "df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True).dropna()\n"
                "target = 'gdp_growth_qoq'\n"
                "X = df.drop(columns=[c for c in df.columns if c.startswith('gdp_') or c in {'GDPC1','recession','target_recession_next_q'}], errors='ignore')\n"
                "y = df[target]\n"
                "split = int(len(df)*0.8)\n"
                "X_tr, X_te = X.iloc[:split], X.iloc[split:]\n"
                "y_tr, y_te = y.iloc[:split], y.iloc[split:]\n"
            ),
            "Fit ridge/lasso": (
                "import numpy as np\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import Ridge, Lasso\n"
                "\n"
                "alphas = [0.01, 0.1, 1.0, 10.0]\n"
                "ridge_coefs = {}\n"
                "lasso_coefs = {}\n"
                "for a in alphas:\n"
                "    r = Pipeline([('scaler', StandardScaler()), ('m', Ridge(alpha=a))]).fit(X_tr, y_tr)\n"
                "    l = Pipeline([('scaler', StandardScaler()), ('m', Lasso(alpha=a, max_iter=5000))]).fit(X_tr, y_tr)\n"
                "    ridge_coefs[a] = r.named_steps['m'].coef_\n"
                "    lasso_coefs[a] = l.named_steps['m'].coef_\n"
                "ridge_coefs.keys(), lasso_coefs.keys()\n"
            ),
            "Coefficient paths": (
                "import matplotlib.pyplot as plt\n"
                "\n"
                "# Plot a few coefficient paths (first 5 features)\n"
                "feat_names = list(X.columns)\n"
                "for i in range(min(5, len(feat_names))):\n"
                "    plt.plot(alphas, [ridge_coefs[a][i] for a in alphas], label=f'Ridge {feat_names[i]}')\n"
                "plt.xscale('log')\n"
                "plt.legend()\n"
                "plt.title('Ridge coefficient paths (subset)')\n"
                "plt.show()\n"
            ),
        }

    if stem == "06_rolling_regressions_stability":
        return {
            "Rolling regression": (
                "import pandas as pd\n"
                "import statsmodels.api as sm\n"
                "\n"
                "df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True).dropna()\n"
                "window = 12  # quarters\n"
                "betas = []\n"
                "dates = []\n"
                "for i in range(window, len(df)+1):\n"
                "    sub = df.iloc[i-window:i]\n"
                "    X = sm.add_constant(sub[['T10Y2Y']])\n"
                "    res = sm.OLS(sub['gdp_growth_qoq'], X).fit()\n"
                "    betas.append(res.params['T10Y2Y'])\n"
                "    dates.append(sub.index[-1])\n"
                "beta_series = pd.Series(betas, index=dates)\n"
                "beta_series.tail()\n"
            ),
            "Coefficient drift": (
                "import matplotlib.pyplot as plt\n"
                "beta_series.plot(title='Rolling coefficient: GDP growth ~ yield spread')\n"
                "plt.show()\n"
            ),
            "Regime interpretation": "# Coefficient drift suggests relationships are not stable across eras.\n",
        }

    # Classification notebooks
    if stem == "00_recession_classifier_baselines":
        return {
            "Load data": (
                "import pandas as pd\n"
                "df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True).dropna()\n"
                "df[['target_recession_next_q']].value_counts(dropna=False)\n"
            ),
            "Define baselines": (
                "import numpy as np\n"
                "from src import evaluation\n"
                "\n"
                "y = df['target_recession_next_q'].astype(int).to_numpy()\n"
                "\n"
                "# Baseline 1: always predict base rate\n"
                "p_base = np.full_like(y, y.mean(), dtype=float)\n"
                "m_base = evaluation.classification_metrics(y, p_base)\n"
                "\n"
                "# Baseline 2: predict next recession = current recession (persistence)\n"
                "p_persist = df['recession'].astype(float).to_numpy()\n"
                "m_persist = evaluation.classification_metrics(y, p_persist)\n"
                "\n"
                "{'base_rate': m_base, 'persistence': m_persist}\n"
            ),
            "Evaluate metrics": "# See above (ROC-AUC, PR-AUC, Brier).\n",
        }

    if stem == "01_logistic_recession_classifier":
        return {
            "Train/test split": (
                "import pandas as pd\n"
                "import numpy as np\n"
                "\n"
                "df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True).dropna()\n"
                "target = 'target_recession_next_q'\n"
                "drop_cols = {target, 'recession', 'GDPC1', 'gdp_growth_qoq', 'gdp_growth_qoq_annualized', 'gdp_growth_yoy'}\n"
                "X = df[[c for c in df.columns if c not in drop_cols]].astype(float)\n"
                "y = df[target].astype(int)\n"
                "split = int(len(df)*0.8)\n"
                "X_tr, X_te = X.iloc[:split], X.iloc[split:]\n"
                "y_tr, y_te = y.iloc[:split], y.iloc[split:]\n"
            ),
            "Fit logistic": (
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from src import evaluation\n"
                "\n"
                "clf = Pipeline([('scaler', StandardScaler()), ('m', LogisticRegression(max_iter=5000))])\n"
                "clf.fit(X_tr, y_tr)\n"
                "p = clf.predict_proba(X_te)[:,1]\n"
                "evaluation.classification_metrics(y_te.to_numpy(), p)\n"
            ),
            "ROC/PR": (
                "from sklearn.metrics import roc_curve, precision_recall_curve\n"
                "from src import plots\n"
                "\n"
                "fpr, tpr, _ = roc_curve(y_te, p)\n"
                "prec, rec, _ = precision_recall_curve(y_te, p)\n"
                "plots.plot_roc_curve(fpr, tpr)\n"
                "plots.plot_pr_curve(rec, prec)\n"
            ),
            "Threshold tuning": (
                "from src import evaluation\n"
                "for thr in [0.3, 0.5, 0.7]:\n"
                "    print(thr, evaluation.classification_metrics(y_te.to_numpy(), p, threshold=thr))\n"
            ),
        }

    if stem == "02_calibration_and_costs":
        return {
            "Calibration": (
                "from sklearn.calibration import calibration_curve\n"
                "import matplotlib.pyplot as plt\n"
                "\n"
                "# Assume y_te and p from prior notebook\n"
                "prob_true, prob_pred = calibration_curve(y_te, p, n_bins=6)\n"
                "plt.plot(prob_pred, prob_true, marker='o')\n"
                "plt.plot([0,1],[0,1], linestyle='--', color='gray')\n"
                "plt.xlabel('Predicted')\n"
                "plt.ylabel('Observed')\n"
                "plt.title('Calibration curve')\n"
                "plt.show()\n"
            ),
            "Brier score": (
                "from sklearn.metrics import brier_score_loss\n"
                "brier_score_loss(y_te, p)\n"
            ),
            "Decision costs": (
                "import numpy as np\n"
                "\n"
                "# Simple cost model: cost_fp and cost_fn\n"
                "cost_fp = 1.0\n"
                "cost_fn = 5.0\n"
                "\n"
                "def expected_cost(thr):\n"
                "    pred = (p >= thr).astype(int)\n"
                "    fp = ((pred == 1) & (y_te.to_numpy() == 0)).sum()\n"
                "    fn = ((pred == 0) & (y_te.to_numpy() == 1)).sum()\n"
                "    return cost_fp*fp + cost_fn*fn\n"
                "\n"
                "candidates = np.linspace(0.05, 0.95, 19)\n"
                "best = min([(thr, expected_cost(thr)) for thr in candidates], key=lambda t: t[1])\n"
                "best\n"
            ),
        }

    if stem == "03_tree_models_and_importance":
        return {
            "Fit tree model": (
                "from sklearn.ensemble import RandomForestClassifier\n"
                "from src import evaluation\n"
                "\n"
                "rf = RandomForestClassifier(n_estimators=300, random_state=0)\n"
                "rf.fit(X_tr, y_tr)\n"
                "p_rf = rf.predict_proba(X_te)[:,1]\n"
                "evaluation.classification_metrics(y_te.to_numpy(), p_rf)\n"
            ),
            "Compare metrics": "# Compare to logistic metrics from earlier.\n",
            "Interpret importance": (
                "import pandas as pd\n"
                "from sklearn.inspection import permutation_importance\n"
                "\n"
                "imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)\n"
                "imp.head(10)\n"
                "\n"
                "perm = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0)\n"
                "perm_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)\n"
                "perm_imp.head(10)\n"
            ),
        }

    if stem == "04_walk_forward_validation":
        return {
            "Walk-forward splits": (
                "import numpy as np\n"
                "import pandas as pd\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from src import evaluation\n"
                "\n"
                "df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True).dropna()\n"
                "target = 'target_recession_next_q'\n"
                "drop_cols = {target, 'recession', 'GDPC1', 'gdp_growth_qoq', 'gdp_growth_qoq_annualized', 'gdp_growth_yoy'}\n"
                "X = df[[c for c in df.columns if c not in drop_cols]].astype(float)\n"
                "y = df[target].astype(int).to_numpy()\n"
                "\n"
                "clf = Pipeline([('scaler', StandardScaler()), ('m', LogisticRegression(max_iter=5000))])\n"
                "\n"
                "metrics = []\n"
                "for split in evaluation.walk_forward_splits(len(df), initial_train_size=20, test_size=4, step_size=2):\n"
                "    clf.fit(X.iloc[split.train_slice], y[split.train_slice])\n"
                "    p = clf.predict_proba(X.iloc[split.test_slice])[:,1]\n"
                "    m = evaluation.classification_metrics(y[split.test_slice], p)\n"
                "    metrics.append(m)\n"
                "\n"
                "pd.DataFrame(metrics).head()\n"
            ),
            "Metric stability": "# Plot ROC-AUC/PR-AUC over folds and inspect variance.\n",
            "Failure analysis": "# Identify folds with worst metrics and map to dates.\n",
        }

    # Unsupervised notebooks
    if stem == "01_pca_macro_factors":
        return {
            "Standardize": (
                "import pandas as pd\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "\n"
                "panel = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True).dropna()\n"
                "X = StandardScaler().fit_transform(panel)\n"
            ),
            "Fit PCA": (
                "from sklearn.decomposition import PCA\n"
                "pca = PCA(n_components=3).fit(X)\n"
                "pca.explained_variance_ratio_\n"
            ),
            "Interpret loadings": (
                "import pandas as pd\n"
                "loadings = pd.DataFrame(pca.components_.T, index=panel.columns, columns=[f'PC{i+1}' for i in range(pca.n_components_)])\n"
                "loadings.sort_values('PC1', key=abs, ascending=False).head(10)\n"
            ),
        }

    if stem == "02_clustering_macro_regimes":
        return {
            "Clustering": (
                "import pandas as pd\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.cluster import KMeans\n"
                "\n"
                "panel = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True).dropna()\n"
                "X = StandardScaler().fit_transform(panel)\n"
                "kmeans = KMeans(n_clusters=4, random_state=0).fit(X)\n"
                "labels = pd.Series(kmeans.labels_, index=panel.index, name='cluster')\n"
                "labels.value_counts()\n"
            ),
            "Choose k": "# Try k=3..6 and compare interpretability.\n",
            "Relate to recessions": (
                "# Join monthly clusters to quarterly recession labels by resampling cluster mode to quarters.\n"
                "import pandas as pd\n"
                "gdp = pd.read_csv(SAMPLE_DIR / 'gdp_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "cluster_q = labels.resample('QE').agg(lambda s: s.value_counts().index[0])\n"
                "joined = pd.DataFrame({'cluster': cluster_q}).join(gdp[['recession']], how='inner')\n"
                "pd.crosstab(joined['cluster'], joined['recession'])\n"
            ),
        }

    if stem == "03_anomaly_detection":
        return {
            "Fit detector": (
                "import pandas as pd\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.ensemble import IsolationForest\n"
                "\n"
                "panel = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True).dropna()\n"
                "X = StandardScaler().fit_transform(panel)\n"
                "iso = IsolationForest(contamination=0.05, random_state=0).fit(X)\n"
                "score = -iso.decision_function(X)\n"
                "anomaly = pd.Series(score, index=panel.index, name='anomaly_score')\n"
                "anomaly.sort_values(ascending=False).head(10)\n"
            ),
            "Inspect anomalies": "# Plot anomaly_score over time and inspect peaks.\n",
            "Compare to recessions": "# Compare anomaly peaks to recession labels.\n",
        }

    # Model ops notebooks
    if stem == "01_reproducible_pipeline_design":
        return {
            "Configs": (
                "# Open configs/recession.yaml and configs/census.yaml and explain each field.\n"
            ),
            "Outputs": (
                "# Run:\n"
                "#   python scripts/build_datasets.py --recession-config configs/recession.yaml --census-config configs/census.yaml\n"
                "#   python scripts/train_recession.py --config configs/recession.yaml\n"
                "# Then inspect outputs/<run_id>/\n"
            ),
            "Reproducibility": (
                "# Confirm run_metadata.json includes dataset hash and feature list.\n"
            ),
        }

    if stem == "02_build_cli_train_predict":
        return {
            "Training CLI": (
                "# Reference idea: add argparse flags in scripts/train_recession.py\n"
                "# - --include-gdp-features true/false\n"
                "# - --model logistic|rf\n"
                "# Then branch logic when selecting feature_cols and choosing estimator.\n"
            ),
            "Prediction CLI": (
                "# Reference idea: add --last-n option to scripts/predict_recession.py\n"
                "# Slice the output dataframe before writing.\n"
            ),
            "Artifacts": (
                "# Ensure each run writes:\n"
                "# - model.joblib\n"
                "# - metrics.json\n"
                "# - predictions.csv\n"
                "# - run_metadata.json\n"
            ),
        }

    if stem == "03_model_cards_and_reporting":
        return {
            "Model card": (
                "# Use reports/capstone_report.md as a template.\n"
                "# Fill it using one specific outputs/<run_id>.\n"
            ),
            "Reporting": (
                "# Include: metrics, calibration, top drivers, error analysis by date.\n"
            ),
            "Limitations": (
                "# Include: GDP revisions, structural breaks, leakage risks, regime shifts.\n"
            ),
        }

    # Causal notebooks (panels, DiD, IV)
    if stem == "00_build_census_county_panel":
        return {
            "Choose years + variables": (
                "import yaml\n"
                "\n"
                "cfg = yaml.safe_load((PROJECT_ROOT / 'configs' / 'census_panel.yaml').read_text())\n"
                "acs = cfg['acs_panel']\n"
                "years = list(acs['years'])\n"
                "acs_vars = list(acs['get'])\n"
                "dataset = acs.get('dataset', 'acs/acs5')\n"
                "geo_for = acs['geography']['for']\n"
                "geo_in = acs['geography'].get('in')\n"
                "\n"
                "years[:3], acs_vars[:5]\n"
            ),
            "Fetch/cache ACS tables": (
                "import pandas as pd\n"
                "\n"
                "# Offline default: load the bundled sample panel.\n"
                "panel_raw = pd.read_csv(SAMPLE_DIR / 'census_county_panel_sample.csv')\n"
                "panel_raw.head()\n"
            ),
            "Build panel + FIPS": (
                "import pandas as pd\n"
                "\n"
                "df = panel_raw.copy()\n"
                "df['state'] = df['state'].astype(str).str.zfill(2)\n"
                "df['county'] = df['county'].astype(str).str.zfill(3)\n"
                "df['fips'] = df['state'] + df['county']\n"
                "df['year'] = df['year'].astype(int)\n"
                "\n"
                "# Recompute derived rates (safe guards included)\n"
                "df['unemployment_rate'] = (\n"
                "    df['B23025_005E'].astype(float) / df['B23025_002E'].replace({0: pd.NA}).astype(float)\n"
                ").astype(float)\n"
                "df['poverty_rate'] = (\n"
                "    df['B17001_002E'].astype(float) / df['B01003_001E'].replace({0: pd.NA}).astype(float)\n"
                ").astype(float)\n"
                "\n"
                "panel = df.set_index(['fips', 'year'], drop=False).sort_index()\n"
                "panel.head()\n"
            ),
            "Save processed panel": (
                "out_path = PROCESSED_DIR / 'census_county_panel.csv'\n"
                "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "panel.to_csv(out_path, index=True)\n"
                "\n"
                "print('wrote', out_path)\n"
            ),
        }

    if stem == "01_panel_fixed_effects_clustered_se":
        return {
            "Load panel and define variables": (
                "import numpy as np\n"
                "import pandas as pd\n"
                "\n"
                "path = PROCESSED_DIR / 'census_county_panel.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'census_county_panel_sample.csv')\n"
                "\n"
                "df['fips'] = df['fips'].astype(str)\n"
                "df['year'] = df['year'].astype(int)\n"
                "df = df.set_index(['fips', 'year'], drop=False).sort_index()\n"
                "\n"
                "df['log_income'] = np.log(df['B19013_001E'].astype(float))\n"
                "df['log_rent'] = np.log(df['B25064_001E'].astype(float))\n"
                "df[['poverty_rate', 'log_income', 'unemployment_rate']].describe()\n"
            ),
            "Pooled OLS baseline": (
                "import statsmodels.api as sm\n"
                "\n"
                "tmp = df[['poverty_rate', 'log_income', 'unemployment_rate']].dropna().copy()\n"
                "y = tmp['poverty_rate'].astype(float)\n"
                "X = sm.add_constant(tmp[['log_income', 'unemployment_rate']], has_constant='add')\n"
                "res = sm.OLS(y, X).fit(cov_type='HC3')\n"
                "print(res.summary())\n"
            ),
            "Two-way fixed effects": (
                "from src.causal import fit_twfe_panel_ols\n"
                "\n"
                "res_twfe = fit_twfe_panel_ols(\n"
                "    df,\n"
                "    y_col='poverty_rate',\n"
                "    x_cols=['log_income', 'unemployment_rate'],\n"
                "    entity_effects=True,\n"
                "    time_effects=True,\n"
                ")\n"
                "print(res_twfe.summary)\n"
            ),
            "Clustered standard errors": (
                "from src.causal import fit_twfe_panel_ols\n"
                "\n"
                "res_cluster = fit_twfe_panel_ols(\n"
                "    df,\n"
                "    y_col='poverty_rate',\n"
                "    x_cols=['log_income', 'unemployment_rate'],\n"
                "    entity_effects=True,\n"
                "    time_effects=True,\n"
                "    cluster_col='state',\n"
                ")\n"
                "\n"
                "pd.DataFrame({'robust_se': res_twfe.std_errors, 'cluster_se': res_cluster.std_errors})\n"
            ),
        }

    if stem == "02_difference_in_differences_event_study":
        return {
            "Synthetic adoption + treatment": (
                "import numpy as np\n"
                "import pandas as pd\n"
                "\n"
                "df = pd.read_csv(SAMPLE_DIR / 'census_county_panel_sample.csv')\n"
                "df['fips'] = df['fips'].astype(str)\n"
                "df['year'] = df['year'].astype(int)\n"
                "\n"
                "states = sorted(df['state'].astype(str).unique())\n"
                "# Deterministic synthetic adoption schedule\n"
                "adopt = {states[0]: 2018, states[1]: 2020}  # states[2] is never-treated\n"
                "\n"
                "df['adopt_year'] = df['state'].astype(str).map(adopt)\n"
                "df['ever_treated'] = df['adopt_year'].notna().astype(int)\n"
                "df['post'] = ((df['year'] >= df['adopt_year']).fillna(False)).astype(int)\n"
                "df['treated'] = df['ever_treated'] * df['post']\n"
                "\n"
                "# Semi-synthetic outcome: add a known post effect.\n"
                "true_effect = -0.02\n"
                "df['poverty_rate_real'] = df['poverty_rate'].astype(float)\n"
                "df['poverty_rate_semi'] = (df['poverty_rate_real'] + true_effect * df['treated']).clip(0, 1)\n"
                "\n"
                "df[['state', 'year', 'treated', 'poverty_rate_real', 'poverty_rate_semi']].head()\n"
            ),
            "TWFE DiD": (
                "from src.causal import fit_twfe_panel_ols\n"
                "\n"
                "df = df.set_index(['fips', 'year'], drop=False).sort_index()\n"
                "\n"
                "res = fit_twfe_panel_ols(\n"
                "    df,\n"
                "    y_col='poverty_rate_semi',\n"
                "    x_cols=['treated'],\n"
                "    entity_effects=True,\n"
                "    time_effects=True,\n"
                "    cluster_col='state',\n"
                ")\n"
                "res.params\n"
            ),
            "Event study (leads/lags)": (
                "import numpy as np\n"
                "\n"
                "df_es = df.reset_index(drop=True).copy()\n"
                "df_es['event_time'] = df_es['year'] - df_es['adopt_year']\n"
                "\n"
                "window = list(range(-3, 4))\n"
                "base = -1\n"
                "event_cols = []\n"
                "for k in window:\n"
                "    if k == base:\n"
                "        continue\n"
                "    col = f'event_{k}'\n"
                "    df_es[col] = ((df_es['ever_treated'] == 1) & (df_es['event_time'] == k)).astype(int)\n"
                "    event_cols.append(col)\n"
                "\n"
                "df_es = df_es.set_index(['fips', 'year'], drop=False).sort_index()\n"
                "\n"
                "res_es = fit_twfe_panel_ols(\n"
                "    df_es,\n"
                "    y_col='poverty_rate_semi',\n"
                "    x_cols=event_cols,\n"
                "    entity_effects=True,\n"
                "    time_effects=True,\n"
                "    cluster_col='state',\n"
                ")\n"
                "\n"
                "coefs = res_es.params.filter(like='event_')\n"
                "ses = res_es.std_errors.filter(like='event_')\n"
                "out = (coefs.to_frame('coef').join(ses.to_frame('se')))\n"
                "out\n"
            ),
            "Diagnostics: pre-trends + placebo": (
                "# Pre-trends: inspect lead coefficients (event_-3, event_-2).\n"
                "# Placebo: shift adoption earlier and confirm estimated effect shrinks toward 0.\n"
            ),
        }

    if stem == "03_instrumental_variables_2sls":
        return {
            "Simulate endogeneity": (
                "import numpy as np\n"
                "import pandas as pd\n"
                "\n"
                "rng = np.random.default_rng(0)\n"
                "n = 2000\n"
                "z = rng.normal(size=n)          # instrument\n"
                "u = rng.normal(size=n)          # unobserved confounder\n"
                "\n"
                "x = 0.8*z + 0.8*u + rng.normal(size=n)  # endogenous regressor\n"
                "eps = 0.8*u + rng.normal(size=n)        # error correlated with x\n"
                "\n"
                "beta_true = 1.5\n"
                "y = beta_true * x + eps\n"
                "\n"
                "df = pd.DataFrame({'y': y, 'x': x, 'z': z})\n"
                "df.head()\n"
            ),
            "OLS vs 2SLS": (
                "import statsmodels.api as sm\n"
                "from src.causal import fit_iv_2sls\n"
                "\n"
                "ols = sm.OLS(df['y'], sm.add_constant(df[['x']], has_constant='add')).fit()\n"
                "print('OLS beta:', float(ols.params['x']))\n"
                "\n"
                "iv = fit_iv_2sls(df, y_col='y', x_endog='x', x_exog=[], z_cols=['z'])\n"
                "print('IV beta :', float(iv.params['x']))\n"
            ),
            "First-stage + weak IV checks": (
                "# Inspect first stage output (instrument strength):\n"
                "# iv.first_stage\n"
            ),
            "Interpretation + limitations": (
                "# Write 3-5 sentences on:\n"
                "# - relevance + exclusion in your simulated setup\n"
                "# - why IV can fix endogeneity here\n"
            ),
        }

    # Time-series econometrics notebooks
    if stem == "00_stationarity_unit_roots":
        return {
            "Load macro series": (
                "import pandas as pd\n"
                "\n"
                "path = PROCESSED_DIR / 'panel_monthly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "df = df.dropna().copy()\n"
                "df.head()\n"
            ),
            "Transformations": (
                "# Example: difference CPI and unemployment\n"
                "df_t = df[['CPIAUCSL', 'UNRATE']].astype(float).copy()\n"
                "df_t['dCPI'] = df_t['CPIAUCSL'].diff()\n"
                "df_t['dUNRATE'] = df_t['UNRATE'].diff()\n"
                "df_t = df_t.dropna()\n"
                "df_t.head()\n"
            ),
            "ADF/KPSS tests": (
                "from statsmodels.tsa.stattools import adfuller, kpss\n"
                "\n"
                "x = df['CPIAUCSL'].astype(float).dropna()\n"
                "dx = x.diff().dropna()\n"
                "\n"
                "adf_p_level = adfuller(x)[1]\n"
                "adf_p_diff = adfuller(dx)[1]\n"
                "kpss_p_level = kpss(x, regression='c', nlags='auto')[1]\n"
                "kpss_p_diff = kpss(dx, regression='c', nlags='auto')[1]\n"
                "\n"
                "{'adf_p_level': adf_p_level, 'adf_p_diff': adf_p_diff, 'kpss_p_level': kpss_p_level, 'kpss_p_diff': kpss_p_diff}\n"
            ),
            "Spurious regression demo": (
                "import statsmodels.api as sm\n"
                "\n"
                "# Levels-on-levels can look 'significant' even when dynamics are mis-specified.\n"
                "tmp = df[['CPIAUCSL', 'INDPRO']].astype(float).dropna()\n"
                "res_lvl = sm.OLS(tmp['CPIAUCSL'], sm.add_constant(tmp[['INDPRO']], has_constant='add')).fit()\n"
                "res_diff = sm.OLS(tmp['CPIAUCSL'].diff().dropna(), sm.add_constant(tmp['INDPRO'].diff().dropna(), has_constant='add')).fit()\n"
                "\n"
                "(res_lvl.rsquared, res_diff.rsquared)\n"
            ),
        }

    if stem == "01_cointegration_error_correction":
        return {
            "Construct cointegrated pair": (
                "import numpy as np\n"
                "import pandas as pd\n"
                "\n"
                "rng = np.random.default_rng(0)\n"
                "n = 240\n"
                "idx = pd.date_range('2000-01-31', periods=n, freq='ME')\n"
                "\n"
                "x = rng.normal(size=n).cumsum()  # random walk\n"
                "y = 1.0 * x + rng.normal(scale=0.5, size=n)  # cointegrated with x\n"
                "\n"
                "df = pd.DataFrame({'x': x, 'y': y}, index=idx)\n"
                "df.head()\n"
            ),
            "Engle-Granger test": (
                "from statsmodels.tsa.stattools import coint\n"
                "\n"
                "t_stat, p_val, _ = coint(df['y'], df['x'])\n"
                "{'t': t_stat, 'p': p_val}\n"
            ),
            "Error correction model": (
                "import statsmodels.api as sm\n"
                "\n"
                "# Step 1: long-run relationship\n"
                "lr = sm.OLS(df['y'], sm.add_constant(df[['x']], has_constant='add')).fit()\n"
                "df['u'] = lr.resid\n"
                "\n"
                "# Step 2: ECM\n"
                "ecm = pd.DataFrame({\n"
                "    'dy': df['y'].diff(),\n"
                "    'dx': df['x'].diff(),\n"
                "    'u_lag1': df['u'].shift(1),\n"
                "}).dropna()\n"
                "\n"
                "res = sm.OLS(ecm['dy'], sm.add_constant(ecm[['dx', 'u_lag1']], has_constant='add')).fit()\n"
                "res.params\n"
            ),
            "Interpretation": "# Explain what the error-correction coefficient implies about mean reversion.\n",
        }

    if stem == "02_var_impulse_responses":
        return {
            "Build stationary dataset": (
                "import pandas as pd\n"
                "\n"
                "panel = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True).dropna()\n"
                "df = panel[['UNRATE', 'FEDFUNDS', 'INDPRO']].astype(float).diff().dropna()\n"
                "df.head()\n"
            ),
            "Fit VAR + choose lags": (
                "from statsmodels.tsa.api import VAR\n"
                "\n"
                "res = VAR(df).fit(maxlags=8, ic='aic')\n"
                "res.k_ar\n"
            ),
            "Granger causality": (
                "# Example: do lagged FEDFUNDS help predict UNRATE?\n"
                "res.test_causality('UNRATE', ['FEDFUNDS']).summary()\n"
            ),
            "IRFs + forecasting": (
                "irf = res.irf(12)\n"
                "irf.plot(orth=True)\n"
            ),
        }

    # Capstone notebooks
    if stem == "00_capstone_brief":
        return {
            "Deliverables": (
                "# Deliverables:\n"
                "# - outputs/<run_id>/ artifacts\n"
                "# - reports/capstone_report.md completed\n"
                "# - apps/streamlit_app.py running against your artifacts\n"
            ),
            "Rubric": "# Use the rubric in the notebook markdown.\n",
            "Scope selection": "# Choose macro-only or macro+micro and justify.\n",
        }

    if stem == "01_capstone_workspace":
        return {
            "Data": (
                "# Load macro_quarterly.csv (or build it with scripts/build_datasets.py)\n"
            ),
            "Modeling": (
                "# Train at least two models (e.g., logistic and random forest) and compare walk-forward metrics.\n"
            ),
            "Interpretation": (
                "# Provide coefficient/importance + error analysis + calibration.\n"
            ),
            "Artifacts": (
                "# Save artifacts under outputs/<run_id> and update reports/capstone_report.md.\n"
            ),
        }

    return {}


def solutions_markdown(stem: str, sections: list[str]) -> str:
    snippets = solution_snippets(stem)

    blocks: list[str] = []
    for section in sections:
        code_snippet = snippets.get(section)
        if not code_snippet:
            code_snippet = "# No reference solution provided for this section yet.\n"

        blocks.append(
            "\n".join(
                [
                    f"<details><summary>Solution: {section}</summary>",
                    "",
                    "_One possible approach. Your variable names may differ; align them with the notebook._",
                    "",
                    "```python",
                    f"# Reference solution for {stem} — {section}",
                    code_snippet.rstrip(),
                    "```",
                    "",
                    "</details>",
                ]
            )
        )

    return (
        "<a id=\"solutions-reference\"></a>\n"
        "## Solutions (Reference)\n\n"
        "Try the TODOs first. Use these only to unblock yourself or to compare approaches.\n\n"
        + "\n\n".join(blocks)
        + "\n"
    )


def write_notebook(spec: NotebookSpec, root: Path) -> None:
    nb_path = Path(spec.path)
    stem = nb_path.stem
    category = nb_path.parts[1] if len(nb_path.parts) > 1 else "misc"
    guide_path = f"docs/guides/{category}/{stem}.md"

    out_path = root / spec.path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    toc_lines = ["## Table of Contents"]
    for s in spec.sections:
        toc_lines.append(f"- [{s}](#{slugify(s)})")
    toc_lines.append("- [Checkpoint (Self-Check)](#checkpoint-self-check)")
    toc_lines.append("- [Solutions (Reference)](#solutions-reference)")

    cells: List[dict] = [
        md(f"# {spec.title}\n\n{spec.summary}"),
        md("\n".join(toc_lines)),
        md(notebook_front_matter(stem, category, guide_path=guide_path)),
        md(
            "## How To Use This Notebook\n"
            "- Work section-by-section; don’t skip the markdown.\n"
            "- Most code cells are incomplete on purpose: replace TODOs and `...`, then run.\n"
            "- After each section, write 2–4 sentences answering the interpretation prompts (what changed, why it matters).\n"
            "- Prefer `data/processed/*` if you have built the real datasets; otherwise use the bundled `data/sample/*` fallbacks.\n"
            "- Use the **Checkpoint (Self-Check)** section to catch mistakes early.\n"
            "- Use **Solutions (Reference)** only to unblock yourself; then re-implement without looking.\n"
            f"- Use the matching guide (`{guide_path}`) for the math, assumptions, and deeper context.\n"
        ),
        md(
            "<a id=\"environment-bootstrap\"></a>\n"
            "## Environment Bootstrap\n"
            "Run this cell first. It makes the repo importable and defines common directories.\n"
        ),
        code(BOOTSTRAP),
    ]

    # Add notebook-specific scaffold blocks.
    #
    # NOTE: These notebooks are intentionally markdown-heavy and TODO-driven.
    # The matching guide goes deeper on math/interpretation; the notebook should be standalone.
    if spec.path.endswith("00_setup.ipynb"):
        cells += [
            md(primer("paths_and_env")),
            md(
                f"<a id=\"{slugify('Verify API keys')}\"></a>\n"
                "## Verify API keys\n\n"
                "In this project you will use two APIs:\n"
                "- **FRED** (macro time series)\n"
                "- **US Census ACS** (micro cross-sectional data)\n\n"
                "We read API keys from environment variables:\n"
                "- `FRED_API_KEY`\n"
                "- `CENSUS_API_KEY` (optional; many endpoints work without it)\n\n"
                "Your task: confirm your notebook process can see these variables.\n"
            ),
            md("### Your Turn (1): Read env vars safely"),
            code(
                "import os\n\n"
                "# TODO: Read env vars with os.getenv\n"
                "fred_key = os.getenv('FRED_API_KEY')\n"
                "census_key = os.getenv('CENSUS_API_KEY')\n"
                "\n"
                "# TODO: Print whether each key is set (do NOT print the full key)\n"
                "# Hint: show the first 4 chars only if present\n"
                "...\n"
            ),
            md("### Your Turn (2): Confirm your data folders exist"),
            code(
                "# TODO: Print the important directories defined by the bootstrap cell.\n"
                "# Confirm they exist (or create them where appropriate).\n"
                "print('PROJECT_ROOT:', PROJECT_ROOT)\n"
                "print('DATA_DIR:', DATA_DIR)\n"
                "print('RAW_DIR:', RAW_DIR)\n"
                "print('PROCESSED_DIR:', PROCESSED_DIR)\n"
                "print('SAMPLE_DIR:', SAMPLE_DIR)\n"
                "\n"
                "# TODO: Create RAW_DIR and PROCESSED_DIR if they don't exist\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Load sample data')}\"></a>\n"
                "## Load sample data\n\n"
                "This repo includes small sample datasets under `data/sample/` so you can work offline.\n"
                "In most notebooks you'll:\n"
                "1) try to load a real dataset from `data/processed/`\n"
                "2) if it doesn't exist yet, load a sample from `data/sample/`\n\n"
                "Your task: load the macro quarterly sample and inspect the schema.\n"
            ),
            md("### Your Turn (1): Load a sample dataset"),
            code(
                "import pandas as pd\n\n"
                "# TODO: Load data/sample/macro_quarterly_sample.csv\n"
                "# Hint: use index_col=0 and parse_dates=True\n"
                "df = ...\n"
                "\n"
                "# TODO: Inspect the data\n"
                "# - df.shape\n"
                "# - df.columns\n"
                "# - df.dtypes\n"
                "...\n"
            ),
            md("### Your Turn (2): Quick plot"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Pick 1-2 columns to plot over time (e.g., GDP growth + recession label)\n"
                "# Hint: df[['col1','col2']].plot(subplots=True)\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Checkpoints')}\"></a>\n"
                "## Checkpoints\n\n"
                "These are the kinds of checks you will repeat in every notebook.\n"
                "If you build this habit early, debugging becomes dramatically easier.\n"
            ),
            md("### Checkpoint (data sanity)"),
            code(
                "# TODO: Replace column names below with real ones from df\n"
                "# Example expected columns in macro_quarterly: gdp_growth_qoq, recession, target_recession_next_q\n"
                "\n"
                "# 1) Index checks\n"
                "assert df.index.is_monotonic_increasing\n"
                "assert df.index.inferred_type in {'datetime64', 'datetime64tz'}\n"
                "\n"
                "# 2) Shape checks\n"
                "assert df.shape[0] > 20\n"
                "assert df.shape[1] >= 3\n"
                "\n"
                "# 3) Missingness checks (you may allow some NaNs early due to lags)\n"
                "print(df.isna().sum().sort_values(ascending=False).head(10))\n"
                "\n"
                "# TODO: Write 2 sentences: what does each check protect you from?\n"
                "...\n"
            ),
        ]

    # Foundations
    if spec.path.endswith("01_time_series_basics.ipynb"):
        cells += [
            md(
                "## Concept\n"
                "This notebook is about *timing*.\n\n"
                "Before you touch real economic data, you will practice on a toy series where you can see mistakes clearly.\n"
                "The same mistakes (especially leakage) show up later when you build recession predictors.\n"
            ),
            md(primer("pandas_time_series")),
            md(
                f"<a id=\"{slugify('Toy series')}\"></a>\n"
                "## Toy series\n\n"
                "### Goal\n"
                "Create a daily time series with a trend + noise.\n\n"
                "### Why this matters\n"
                "Real macro indicators are noisy. You want to learn how resampling and rolling windows change the information your model sees.\n\n"
                "### Your Turn (1): Build an index + values\n"
            ),
            code(
                "import numpy as np\n"
                "import pandas as pd\n\n"
                "# TODO: Create a daily DatetimeIndex from 2010-01-01 for ~5 years.\n"
                "# Hint: pd.date_range('2010-01-01', periods=..., freq='D')\n"
                "idx = ...\n"
                "\n"
                "# TODO: Create a signal with a slow trend + seasonal-ish wiggle + noise.\n"
                "# Hint: np.linspace + np.sin + rng.normal\n"
                "rng = np.random.default_rng(0)\n"
                "trend = ...\n"
                "season = ...\n"
                "noise = ...\n"
                "\n"
                "y = pd.Series(trend + season + noise, index=idx, name='y')\n"
                "y.head()\n"
            ),
            md("### Your Turn (2): Inspect and visualize"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Print basic summary stats\n"
                "# TODO: Plot the time series\n"
                "...\n"
            ),
            md("### Checkpoint (toy series sanity)"),
            code(
                "# TODO: Confirm the index is daily and sorted\n"
                "assert isinstance(y.index, pd.DatetimeIndex)\n"
                "assert y.index.is_monotonic_increasing\n"
                "assert y.shape[0] > 1000\n"
                "assert y.isna().sum() == 0\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Resampling')}\"></a>\n"
                "## Resampling\n\n"
                "### Goal\n"
                "Convert daily data to monthly data in two different ways and compare them.\n\n"
                "### Why this matters\n"
                "When you resample, you are choosing a *measurement definition*:\n"
                "- month-end value (`last`) vs\n"
                "- monthly average (`mean`).\n\n"
                "These can lead to different model conclusions.\n"
            ),
            md("### Your Turn (1): Month-end series (mean vs last)"),
            code(
                "# TODO: Build two monthly series:\n"
                "# - y_me_last: month-end value\n"
                "# - y_me_mean: monthly average\n"
                "\n"
                "# Hint: y.resample('ME').last() and y.resample('ME').mean()\n"
                "y_me_last = ...\n"
                "y_me_mean = ...\n"
                "\n"
                "y_me_last.head(), y_me_mean.head()\n"
            ),
            md("### Your Turn (2): Compare visually and numerically"),
            code(
                "# TODO: Put them in one DataFrame and compare.\n"
                "# - plot both\n"
                "# - compute correlation\n"
                "# - compute their difference distribution\n"
                "df_m = ...\n"
                "...\n"
            ),
            md("### Checkpoint (resampling sanity)"),
            code(
                "# TODO: Assert the resampled index is month-end\n"
                "assert y_me_last.index.freqstr in {'ME', 'M'} or y_me_last.index.inferred_freq in {'M', 'ME'}\n"
                "assert y_me_last.shape == y_me_mean.shape\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Lag and rolling features')}\"></a>\n"
                "## Lag and rolling features\n\n"
                "### Goal\n"
                "Create features that only use the past: lags and rolling windows.\n\n"
                "### Why this matters\n"
                "Most macro prediction is built from lagged indicators (what you knew last month/quarter) and summaries of recent history.\n"
            ),
            md("### Your Turn (1): Lags"),
            code(
                "# TODO: Create lag features from the DAILY series y:\n"
                "# - y_lag1: yesterday\n"
                "# - y_lag7: one week ago\n"
                "df_feat = pd.DataFrame({'y': y})\n"
                "df_feat['y_lag1'] = ...\n"
                "df_feat['y_lag7'] = ...\n"
                "df_feat.head(10)\n"
            ),
            md("### Your Turn (2): Rolling features"),
            code(
                "# TODO: Create rolling mean and rolling std features (past-only!)\n"
                "# Example: 14-day rolling mean and std\n"
                "df_feat['y_roll14_mean'] = ...\n"
                "df_feat['y_roll14_std'] = ...\n"
                "\n"
                "# TODO: drop rows with NaNs created by lags/rolling\n"
                "df_feat_clean = ...\n"
                "df_feat_clean.head()\n"
            ),
            md("### Checkpoint (feature availability)"),
            code(
                "# TODO: Confirm you did not accidentally leak the future.\n"
                "# Hint: check that lagged columns align the way you expect.\n"
                "# Example: df_feat.loc[t, 'y_lag1'] should equal df_feat.loc[t - 1 day, 'y']\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Leakage demo')}\"></a>\n"
                "## Leakage demo\n\n"
                "### Goal\n"
                "Experience how leakage can make results look incredible but meaningless.\n\n"
                "### What you'll do\n"
                "1) define a 1-step-ahead prediction task\n"
                "2) build one legitimate feature set and one leaky feature set\n"
                "3) compare random split vs time split\n"
            ),
            md("### Your Turn (1): Build the prediction dataset"),
            code(
                "from sklearn.model_selection import train_test_split\n"
                "from sklearn.linear_model import LinearRegression\n"
                "from sklearn.metrics import mean_squared_error\n\n"
                "# Predict tomorrow's value\n"
                "# TODO: target = y.shift(-1)\n"
                "target = ...\n"
                "\n"
                "# Legit features: past-only\n"
                "X_ok = df_feat_clean[['y_lag1', 'y_lag7', 'y_roll14_mean', 'y_roll14_std']].copy()\n"
                "\n"
                "# LEAKY feature: tomorrow's value (do not do this in real work)\n"
                "# TODO: X_leak should include a column that equals the target (or uses shift(-1))\n"
                "X_leak = ...\n"
                "\n"
                "# Align and drop missing rows\n"
                "df_model = pd.DataFrame({'target': target}).join(X_ok).dropna()\n"
                "X_ok = df_model[X_ok.columns]\n"
                "y_arr = df_model['target']\n"
                "\n"
                "# TODO: Align X_leak to the same rows as y_arr\n"
                "...\n"
            ),
            md("### Your Turn (2): Random split vs time split"),
            code(
                "# Random split (WRONG for time series)\n"
                "X_tr, X_te, y_tr, y_te = train_test_split(X_ok, y_arr, test_size=0.2, shuffle=True, random_state=0)\n"
                "m = LinearRegression().fit(X_tr, y_tr)\n"
                "rmse_rand = mean_squared_error(y_te, m.predict(X_te), squared=False)\n"
                "\n"
                "# Time split (RIGHT for time series)\n"
                "split = int(len(X_ok) * 0.8)\n"
                "X_tr2, X_te2 = X_ok.iloc[:split], X_ok.iloc[split:]\n"
                "y_tr2, y_te2 = y_arr.iloc[:split], y_arr.iloc[split:]\n"
                "m2 = LinearRegression().fit(X_tr2, y_tr2)\n"
                "rmse_time = mean_squared_error(y_te2, m2.predict(X_te2), squared=False)\n"
                "\n"
                "rmse_rand, rmse_time\n"
            ),
            md("### Your Turn (3): Show how the leaky feature 'cheats'"),
            code(
                "# TODO: Repeat the evaluation above using X_leak.\n"
                "# What happens to the test RMSE?\n"
                "# Write 3-5 sentences explaining why this result is meaningless.\n"
                "...\n"
            ),
        ]

    if spec.path.endswith("02_stats_basics_for_ml.ipynb"):
        cells += [
            md(
                "## Concept\n"
                "This notebook gives you the statistical vocabulary you'll use throughout the project.\n\n"
                "You will build intuition for:\n"
                "- when correlations are meaningful vs misleading,\n"
                "- why coefficients can become unstable when features are correlated,\n"
                "- how overfitting shows up as a gap between train and test performance,\n"
                "- how to read hypothesis tests (p-values / confidence intervals) without over-trusting them.\n"
            ),
            md(primer("statsmodels_inference")),
            md(
                f"<a id=\"{slugify('Correlation vs causation')}\"></a>\n"
                "## Correlation vs causation\n\n"
                "### Goal\n"
                "Simulate a classic confounding scenario where variables are correlated without a direct causal relationship.\n\n"
                "### Why this matters in economics\n"
                "Macro indicators often move together. If you interpret correlations as causal effects, you'll make confident but wrong stories.\n"
            ),
            md("### Your Turn (1): Simulate a confounder"),
            code(
                "import numpy as np\n"
                "import pandas as pd\n\n"
                "# We will build: z -> x, z -> w, and z -> y.\n"
                "# That makes x and y correlated even if x doesn't directly cause y.\n"
                "\n"
                "rng = np.random.default_rng(0)\n"
                "n = 800\n"
                "\n"
                "# TODO: Simulate a hidden confounder z\n"
                "z = ...\n"
                "\n"
                "# TODO: Create x and w that both depend on z (plus noise)\n"
                "x = ...\n"
                "w = ...\n"
                "\n"
                "# TODO: Create y that depends on z (plus noise)\n"
                "y = ...\n"
                "\n"
                "df = pd.DataFrame({'z': z, 'x': x, 'w': w, 'y': y})\n"
                "df.head()\n"
            ),
            md("### Your Turn (2): Correlation matrix + interpretation"),
            code(
                "# TODO: Compute df.corr() and interpret.\n"
                "# Questions:\n"
                "# 1) Are x and y correlated?\n"
                "# 2) Does that mean x causes y?\n"
                "# 3) Which variable is the common cause?\n"
                "corr = df.corr(numeric_only=True)\n"
                "corr\n"
            ),
            md("### Optional extension: a simple regression 'control' demo"),
            code(
                "import statsmodels.api as sm\n\n"
                "# TODO: Fit two regressions and compare the coefficient on x:\n"
                "# 1) y ~ x\n"
                "# 2) y ~ x + z\n"
                "# Hint: sm.add_constant + sm.OLS\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Multicollinearity (VIF)')}\"></a>\n"
                "## Multicollinearity (VIF)\n\n"
                "### Goal\n"
                "Create highly correlated predictors and see how they affect coefficient stability.\n\n"
                "### Key term\n"
                "> **Definition:** **Multicollinearity** means your predictors are strongly correlated with each other.\n"
                "It doesn't necessarily hurt prediction, but it can make coefficient interpretation unstable.\n"
            ),
            md("### Your Turn (1): Build correlated features"),
            code(
                "from src.econometrics import vif_table\n\n"
                "# TODO: Create x1 and x2 that are almost the same\n"
                "rng = np.random.default_rng(1)\n"
                "n = 600\n"
                "x1 = rng.normal(size=n)\n"
                "x2 = ...  # make this highly correlated with x1\n"
                "\n"
                "# Target depends mostly on x1\n"
                "eps = rng.normal(scale=1.0, size=n)\n"
                "y2 = 1.0 + 2.0 * x1 + eps\n"
                "\n"
                "df2 = pd.DataFrame({'y': y2, 'x1': x1, 'x2': x2})\n"
                "df2[['x1','x2']].corr()\n"
            ),
            md("### Your Turn (2): Compute VIF + interpret"),
            code(
                "# TODO: Compute VIF for x1 and x2.\n"
                "# How large are the VIFs? What does that suggest?\n"
                "vif_table(df2, ['x1', 'x2'])\n"
            ),
            md("### Your Turn (3): Fit a regression and inspect coefficient stability"),
            code(
                "import statsmodels.api as sm\n\n"
                "# Fit y ~ x1 + x2 and inspect coefficients.\n"
                "# TODO: Compare to fitting y ~ x1 alone.\n"
                "X_both = sm.add_constant(df2[['x1', 'x2']])\n"
                "res_both = sm.OLS(df2['y'], X_both).fit()\n"
                "\n"
                "X_one = sm.add_constant(df2[['x1']])\n"
                "res_one = sm.OLS(df2['y'], X_one).fit()\n"
                "\n"
                "# TODO: Print the two coefficient estimates on x1\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Bias/variance')}\"></a>\n"
                "## Bias/variance\n\n"
                "### Goal\n"
                "See overfitting as a train/test gap by comparing a simple model vs a flexible one.\n\n"
                "### Key term\n"
                "> **Definition:** **Overfitting** happens when a model fits noise in the training data and fails to generalize.\n"
            ),
            md("### Your Turn (1): Create a non-linear dataset"),
            code(
                "import numpy as np\n"
                "from sklearn.metrics import mean_squared_error\n"
                "from sklearn.linear_model import LinearRegression\n"
                "from sklearn.tree import DecisionTreeRegressor\n\n"
                "rng = np.random.default_rng(2)\n"
                "n = 400\n"
                "\n"
                "# TODO: Create x on [-3, 3]\n"
                "x = ...\n"
                "\n"
                "# TODO: Create y = sin(x) + noise (nonlinear)\n"
                "y = ...\n"
                "\n"
                "# Train/test split (random is OK here because this is NOT time series)\n"
                "split = int(n * 0.8)\n"
                "X = x.reshape(-1, 1)\n"
                "X_tr, X_te = X[:split], X[split:]\n"
                "y_tr, y_te = y[:split], y[split:]\n"
                "\n"
                "...\n"
            ),
            md("### Your Turn (2): Fit linear vs tree and compare errors"),
            code(
                "# TODO: Fit LinearRegression and a DecisionTreeRegressor\n"
                "# Compute RMSE on train and test for both\n"
                "...\n"
            ),
            md("### Your Turn (3): Control model complexity"),
            code(
                "# TODO: Refit the tree with different max_depth values (e.g., 2, 4, 8, None)\n"
                "# Track train/test RMSE and describe the pattern.\n"
                "...\n"
            ),
            md(primer("hypothesis_testing")),
            md(
                f"<a id=\"{slugify('Hypothesis testing')}\"></a>\n"
                "## Hypothesis testing\n\n"
                "### Goal\n"
                "Make p-values and confidence intervals concrete with a toy example.\n\n"
                "### Your Turn (1): One-sample t-test\n"
                "Simulate a sample whose true mean is not 0 and test whether you can detect it.\n"
            ),
            code(
                "from scipy import stats\n\n"
                "rng = np.random.default_rng(3)\n"
                "\n"
                "# TODO: Simulate x with a small non-zero mean (e.g., 0.1) and some noise\n"
                "x = ...\n"
                "\n"
                "# TODO: Run a one-sample t-test for mean == 0\n"
                "t_stat, p_val = ...\n"
                "print('t:', t_stat, 'p:', p_val)\n"
                "\n"
                "# TODO: Explain: what does this p-value mean in words?\n"
                "...\n"
            ),
            md("### Your Turn (2): Regression coefficient test"),
            code(
                "import statsmodels.api as sm\n\n"
                "# TODO: Simulate a simple linear relationship y = 1 + 0.5*x + noise\n"
                "rng = np.random.default_rng(4)\n"
                "n = 300\n"
                "x = ...\n"
                "y = ...\n"
                "\n"
                "df_ht = pd.DataFrame({'y': y, 'x': x})\n"
                "X = sm.add_constant(df_ht[['x']])\n"
                "res = sm.OLS(df_ht['y'], X).fit()\n"
                "\n"
                "# TODO: Print coefficient, SE, p-value, and 95% CI for x\n"
                "...\n"
            ),
        ]

    # Data notebooks
    if spec.path.endswith("00_fred_api_and_caching.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Fetch a basket of economic indicators from FRED, cache the raw API responses, and build a tidy time series panel.\n\n"
                "### Why this matters\n"
                "You want to separate:\n"
                "- **data acquisition** (API calls) from\n"
                "- **analysis/modeling** (notebooks and scripts).\n\n"
                "Caching makes your work reproducible and faster.\n"
            ),
            md(primer("paths_and_env")),
            md(
                f"<a id=\"{slugify('Choose series')}\"></a>\n"
                "## Choose series\n\n"
                "### Goal\n"
                "Pick a starter basket of macro indicators.\n\n"
                "### Notes\n"
                "- Different indicators have different frequencies (monthly, daily, quarterly).\n"
                "- We'll deal with alignment in later notebooks.\n"
            ),
            md("### Your Turn (1): Define a basket of series IDs"),
            code(
                "# TODO: Define a list of FRED series IDs (strings)\n"
                "# Suggested starters:\n"
                "# - UNRATE   (Unemployment rate, monthly)\n"
                "# - FEDFUNDS (Fed funds rate, monthly)\n"
                "# - CPIAUCSL (CPI, monthly)\n"
                "# - INDPRO   (Industrial Production, monthly)\n"
                "# - RSAFS    (Retail Sales, monthly)\n"
                "# - T10Y2Y   (10Y-2Y yield spread, daily)\n"
                "series_ids = [\n"
                "    ...,\n"
                "]\n"
                "\n"
                "# TODO: Basic validation\n"
                "assert isinstance(series_ids, list)\n"
                "assert all(isinstance(x, str) and x for x in series_ids)\n"
                "assert len(series_ids) == len(set(series_ids)), 'duplicate series IDs'\n"
                "series_ids\n"
            ),
            md(
                f"<a id=\"{slugify('Fetch metadata')}\"></a>\n"
                "## Fetch metadata\n\n"
                "### Goal\n"
                "Metadata answers: what is this series, what are the units, and what is its native frequency?\n\n"
                "You will use metadata later to decide:\n"
                "- which features are meaningful,\n"
                "- how to align frequencies,\n"
                "- how to interpret coefficient units.\n"
            ),
            md("### Your Turn (1): Fetch metadata for one series"),
            code(
                "from src import fred_api\n\n"
                "# TODO: Pick one series_id and fetch metadata\n"
                "sid = series_ids[0]\n"
                "meta = fred_api.fetch_series_meta(sid)\n"
                "\n"
                "# TODO: Print the most important fields (title, units, frequency)\n"
                "...\n"
            ),
            md("### Your Turn (2): Fetch metadata for all series and build a table"),
            code(
                "import pandas as pd\n"
                "from src import fred_api\n\n"
                "# TODO: Loop over series_ids and build a DataFrame of metadata.\n"
                "# Hint: meta is a dict; you can select keys you care about.\n"
                "rows = []\n"
                "for sid in series_ids:\n"
                "    meta = fred_api.fetch_series_meta(sid)\n"
                "    rows.append({\n"
                "        'id': sid,\n"
                "        'title': meta.get('title'),\n"
                "        'units': meta.get('units'),\n"
                "        'frequency': meta.get('frequency'),\n"
                "        'seasonal_adjustment': meta.get('seasonal_adjustment'),\n"
                "    })\n"
                "\n"
                "meta_df = pd.DataFrame(rows)\n"
                "meta_df\n"
            ),
            md(
                f"<a id=\"{slugify('Fetch + cache observations')}\"></a>\n"
                "## Fetch + cache observations\n\n"
                "### Goal\n"
                "Download observations for each series and cache the raw JSON under `data/raw/fred/`.\n\n"
                "### Why cache raw JSON?\n"
                "- It's the exact raw record of what the API returned.\n"
                "- You can debug parsing issues later without re-downloading.\n"
            ),
            md("### Your Turn (1): Fetch and cache JSON payloads"),
            code(
                "from src import data as data_utils\n"
                "from src import fred_api\n\n"
                "# We'll store raw API responses here\n"
                "raw_dir = RAW_DIR / 'fred'\n"
                "raw_dir.mkdir(parents=True, exist_ok=True)\n"
                "\n"
                "# TODO: For each series_id, cache JSON under data/raw/fred/<id>.json\n"
                "# Hint: data_utils.load_or_fetch_json(path, fetch_fn)\n"
                "...\n"
            ),
            md("### Your Turn (2): Convert cached payloads to a tidy DataFrame panel"),
            code(
                "import pandas as pd\n"
                "from src import data as data_utils\n"
                "from src import fred_api\n\n"
                "# TODO: For each series, load JSON and convert to a 1-column DataFrame\n"
                "# Hint: fred_api.observations_to_frame(payload, sid)\n"
                "frames = []\n"
                "for sid in series_ids:\n"
                "    payload = data_utils.load_json(raw_dir / f'{sid}.json')\n"
                "    frames.append(fred_api.observations_to_frame(payload, sid))\n"
                "\n"
                "panel = pd.concat(frames, axis=1).sort_index()\n"
                "panel.head()\n"
            ),
            md("### Your Turn (3): Inspect missingness and basic ranges"),
            code(
                "# TODO: Print missing values per column\n"
                "# TODO: Print min/max dates\n"
                "# TODO: Describe each column\n"
                "print('date range:', panel.index.min(), '->', panel.index.max())\n"
                "print(panel.isna().sum().sort_values(ascending=False))\n"
                "panel.describe().T\n"
            ),
            md("### Checkpoint (panel sanity)"),
            code(
                "# TODO: These checks should pass if you built a valid panel.\n"
                "assert isinstance(panel.index, pd.DatetimeIndex)\n"
                "assert panel.index.is_monotonic_increasing\n"
                "assert panel.shape[0] > 200\n"
                "assert panel.shape[1] == len(series_ids)\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Fallback to sample')}\"></a>\n"
                "## Fallback to sample\n\n"
                "If you cannot fetch from the API (no key, no network), load the bundled sample panel.\n"
            ),
            code(
                "import os\n"
                "import pandas as pd\n\n"
                "# TODO: Implement a fallback:\n"
                "# - if FRED_API_KEY is missing, load data/sample/panel_monthly_sample.csv\n"
                "# - otherwise, keep using your freshly built panel\n"
                "if not os.getenv('FRED_API_KEY'):\n"
                "    panel = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "panel.head()\n"
            ),
        ]

    if spec.path.endswith("01_build_macro_monthly_panel.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Build a clean **month-end** panel of predictors.\n\n"
                "### Why this matters\n"
                "Most economic indicators are not recorded at the same frequency.\n"
                "Before modeling, you must decide:\n"
                "- the timeline you will predict on (monthly vs quarterly)\n"
                "- how to align mixed-frequency indicators onto that timeline\n\n"
                "In this notebook, you standardize everything to **month-end**.\n"
            ),
            md(primer("pandas_time_series")),
            md(
                f"<a id=\"{slugify('Load series')}\"></a>\n"
                "## Load series\n\n"
                "### Goal\n"
                "Load a macro indicator panel.\n\n"
                "Options:\n"
                "1) If you completed `00_fred_api_and_caching`, load raw series from `data/raw/fred/` JSON files.\n"
                "2) Otherwise, load the offline sample from `data/sample/panel_monthly_sample.csv`.\n"
            ),
            md("### Your Turn (1): Try to load from cached JSON (preferred)"),
            code(
                "import pandas as pd\n"
                "from src import data as data_utils\n"
                "from src import fred_api\n\n"
                "# TODO: Choose the same series_ids you used before.\n"
                "series_ids = ['UNRATE', 'FEDFUNDS', 'CPIAUCSL', 'INDPRO', 'RSAFS', 'T10Y2Y']\n"
                "\n"
                "raw_dir = RAW_DIR / 'fred'\n"
                "\n"
                "# TODO: If JSON files exist, load them and build a raw panel.\n"
                "frames = []\n"
                "for sid in series_ids:\n"
                "    path = raw_dir / f'{sid}.json'\n"
                "    if not path.exists():\n"
                "        continue\n"
                "    payload = data_utils.load_json(path)\n"
                "    frames.append(fred_api.observations_to_frame(payload, sid))\n"
                "\n"
                "panel_raw = pd.concat(frames, axis=1).sort_index() if frames else None\n"
                "panel_raw.head() if panel_raw is not None else None\n"
            ),
            md("### Your Turn (2): Fallback to sample if needed"),
            code(
                "import pandas as pd\n\n"
                "# TODO: If panel_raw is None, load the sample.\n"
                "if panel_raw is None:\n"
                "    panel_raw = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "panel_raw.head()\n"
            ),
            md(
                f"<a id=\"{slugify('Month-end alignment')}\"></a>\n"
                "## Month-end alignment\n\n"
                "### Goal\n"
                "Convert the panel to **month-end** index and decide how to handle series that update more frequently.\n\n"
                "In later notebooks we will aggregate monthly -> quarterly; for now, everything becomes monthly.\n"
            ),
            md("### Your Turn (1): Ensure month-end index"),
            code(
                "from src import features\n\n"
                "# TODO: If panel_raw is already month-end, verify it.\n"
                "# If it is daily (or mixed), resample to month-end.\n"
                "# Hint: features.to_monthly(panel_raw)\n"
                "panel_me = ...\n"
                "\n"
                "panel_me.head()\n"
            ),
            md("### Your Turn (2): Compare before/after resampling"),
            code(
                "# TODO: Print index frequency guesses before and after.\n"
                "# TODO: Print how many rows you have before/after.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Missingness')}\"></a>\n"
                "## Missingness\n\n"
                "### Goal\n"
                "Inspect missing values and choose a strategy.\n\n"
                "In macro panels, a common approach is:\n"
                "- forward-fill within a series after resampling\n"
                "- then drop early rows that are still missing because the series starts later\n"
            ),
            md("### Your Turn (1): Missingness report"),
            code(
                "# TODO: Print missing values per column and as a percent\n"
                "na_counts = panel_me.isna().sum().sort_values(ascending=False)\n"
                "na_pct = (na_counts / len(panel_me)).round(3)\n"
                "pd.DataFrame({'na': na_counts, 'na_pct': na_pct})\n"
            ),
            md("### Your Turn (2): Decide and apply a strategy"),
            code(
                "# TODO: Choose a missingness strategy.\n"
                "# Default suggestion:\n"
                "# - forward fill\n"
                "# - drop remaining NaNs\n"
                "panel_filled = panel_me.ffill()\n"
                "panel_clean = panel_filled.dropna().copy()\n"
                "\n"
                "panel_clean.head()\n"
            ),
            md(
                f"<a id=\"{slugify('Save processed panel')}\"></a>\n"
                "## Save processed panel\n\n"
                "### Goal\n"
                "Write your month-end panel to `data/processed/panel_monthly.csv`.\n"
            ),
            md("### Your Turn (1): Save"),
            code(
                "out_path = PROCESSED_DIR / 'panel_monthly.csv'\n"
                "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "\n"
                "# TODO: Save panel_clean to CSV\n"
                "...\n"
            ),
            md("### Your Turn (2): Load back and validate"),
            code(
                "import pandas as pd\n\n"
                "# TODO: Load the saved file and confirm it matches your in-memory data.\n"
                "panel_check = pd.read_csv(PROCESSED_DIR / 'panel_monthly.csv', index_col=0, parse_dates=True)\n"
                "assert panel_check.shape == panel_clean.shape\n"
                "assert panel_check.index.is_monotonic_increasing\n"
                "panel_check.head()\n"
            ),
            md("### Checkpoint"),
            code(
                "# TODO: Assert no missing values remain and index is month-end-ish.\n"
                "assert not panel_clean.isna().any().any()\n"
                "assert panel_clean.shape[0] > 100\n"
                "...\n"
            ),
        ]

    if spec.path.endswith("02_gdp_growth_and_recession_label.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Compute GDP growth (multiple definitions) and define a **technical recession** label.\n\n"
                "### Technical recession label used in this project\n"
                "We define recession as:\n"
                "- two consecutive quarters of negative **real GDP growth** (QoQ)\n\n"
                "This is a teaching proxy, not an official recession dating rule.\n"
            ),
            md(primer("pandas_time_series")),
            md(
                f"<a id=\"{slugify('Fetch GDP')}\"></a>\n"
                "## Fetch GDP\n\n"
                "### Goal\n"
                "Load quarterly real GDP levels (`GDPC1`).\n\n"
                "Options:\n"
                "1) Fetch from FRED (preferred if you have a key)\n"
                "2) Load the offline sample (`data/sample/gdp_quarterly_sample.csv`)\n"
            ),
            md("### Your Turn (1): Fetch GDPC1 (or load sample)"),
            code(
                "import os\n"
                "import pandas as pd\n"
                "from src import fred_api\n"
                "from src import data as data_utils\n\n"
                "sid = 'GDPC1'\n"
                "\n"
                "# TODO: If you have FRED_API_KEY, fetch observations and convert to a DataFrame.\n"
                "# Hint: fred_api.fetch_series_observations + fred_api.observations_to_frame\n"
                "# Otherwise, load SAMPLE_DIR / 'gdp_quarterly_sample.csv'\n"
                "if os.getenv('FRED_API_KEY'):\n"
                "    payload = fred_api.fetch_series_observations(sid, start_date='1980-01-01', end_date=None)\n"
                "    gdp = fred_api.observations_to_frame(payload, sid)\n"
                "else:\n"
                "    gdp = pd.read_csv(SAMPLE_DIR / 'gdp_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "gdp = gdp.sort_index()\n"
                "gdp.head()\n"
            ),
            md("### Checkpoint (GDP shape)"),
            code(
                "# TODO: Confirm quarterly-ish index and no obvious missingness.\n"
                "assert isinstance(gdp.index, pd.DatetimeIndex)\n"
                "assert gdp.index.is_monotonic_increasing\n"
                "assert gdp.shape[1] == 1\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Compute growth')}\"></a>\n"
                "## Compute growth\n\n"
                "### Goal\n"
                "Compute multiple growth definitions from GDP levels.\n\n"
                "You will compute and compare:\n"
                "- QoQ growth (percent)\n"
                "- Annualized QoQ growth (percent)\n"
                "- YoY growth (percent)\n"
            ),
            md("### Your Turn (1): Compute growth variants"),
            code(
                "from src import macro\n\n"
                "# GDP levels series\n"
                "gdp_levels = gdp['GDPC1'].astype(float)\n"
                "\n"
                "# TODO: Compute growth series\n"
                "gdp_growth_qoq = ...\n"
                "gdp_growth_qoq_ann = ...\n"
                "gdp_growth_yoy = ...\n"
                "\n"
                "gdp_feat = pd.DataFrame({\n"
                "    'gdp_level': gdp_levels,\n"
                "    'gdp_growth_qoq': gdp_growth_qoq,\n"
                "    'gdp_growth_qoq_annualized': gdp_growth_qoq_ann,\n"
                "    'gdp_growth_yoy': gdp_growth_yoy,\n"
                "}).dropna()\n"
                "gdp_feat.head()\n"
            ),
            md("### Your Turn (2): Compare the growth definitions"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Plot the three growth series on separate subplots.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Define recession label')}\"></a>\n"
                "## Define recession label\n\n"
                "### Goal\n"
                "Construct the technical recession label from QoQ growth.\n\n"
                "Definition used here:\n"
                "- `recession_t = 1` if `growth_t < 0` AND `growth_{t-1} < 0`\n"
            ),
            md("### Your Turn (1): Build the label and inspect it"),
            code(
                "from src import macro\n\n"
                "# TODO: Compute technical recession label\n"
                "recession = ...\n"
                "\n"
                "# TODO: Attach to gdp_feat and inspect counts\n"
                "gdp_feat['recession'] = recession\n"
                "print(gdp_feat['recession'].value_counts())\n"
                "gdp_feat[['gdp_growth_qoq', 'recession']].tail(12)\n"
            ),
            md(
                f"<a id=\"{slugify('Define next-quarter target')}\"></a>\n"
                "## Define next-quarter target\n\n"
                "### Goal\n"
                "Create the classifier target: **predict next quarter's recession label**.\n\n"
                "That target is:\n"
                "- `target_recession_next_q[t] = recession[t+1]`\n\n"
                "Be careful: this creates a missing value at the end (there is no future label for the last row).\n"
            ),
            md("### Your Turn (1): Shift label to build the target"),
            code(
                "from src import macro\n\n"
                "# TODO: Shift recession to build next-quarter target\n"
                "gdp_feat['target_recession_next_q'] = ...\n"
                "\n"
                "# TODO: Drop rows where target is missing\n"
                "gdp_feat = gdp_feat.dropna(subset=['target_recession_next_q']).copy()\n"
                "\n"
                "gdp_feat[['recession', 'target_recession_next_q']].tail(6)\n"
            ),
            md("### Your Turn (2): Save to data/processed/gdp_quarterly.csv"),
            code(
                "out_path = PROCESSED_DIR / 'gdp_quarterly.csv'\n"
                "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "\n"
                "# TODO: Save gdp_feat to CSV\n"
                "...\n"
            ),
            md("### Reflection"),
            md(
                "- Why is this only a proxy for 'recession'?\n"
                "- How could this differ from official recession dating?\n"
                "- How might you validate whether this label aligns with economic reality?\n"
            ),
        ]

    if spec.path.endswith("03_build_macro_quarterly_features.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Build a quarterly modeling table by:\n"
                "1) aggregating monthly predictors to quarterly features\n"
                "2) adding lagged predictors (past-only)\n"
                "3) merging with quarterly GDP growth + recession targets\n\n"
                "The output is `data/processed/macro_quarterly.csv`.\n"
            ),
            md(primer("pandas_time_series")),
            md(
                f"<a id=\"{slugify('Aggregate monthly -> quarterly')}\"></a>\n"
                "## Aggregate monthly -> quarterly\n\n"
                "### Goal\n"
                "Convert the month-end panel into a quarterly feature table.\n\n"
                "You will try two aggregation rules:\n"
                "- quarter-end value (`last`)\n"
                "- quarter-average value (`mean`)\n\n"
                "Then you will choose one (or keep both with suffixes).\n"
            ),
            md("### Your Turn (1): Load inputs"),
            code(
                "import pandas as pd\n\n"
                "# TODO: Load the processed monthly panel (or fallback to sample)\n"
                "panel_path = PROCESSED_DIR / 'panel_monthly.csv'\n"
                "if panel_path.exists():\n"
                "    panel_m = pd.read_csv(panel_path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    panel_m = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "# TODO: Load quarterly GDP/label table from the previous notebook\n"
                "gdp_path = PROCESSED_DIR / 'gdp_quarterly.csv'\n"
                "if gdp_path.exists():\n"
                "    gdp_q = pd.read_csv(gdp_path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    gdp_q = pd.read_csv(SAMPLE_DIR / 'gdp_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "panel_m.head(), gdp_q.head()\n"
            ),
            md("### Your Turn (2): Aggregate (mean vs last)"),
            code(
                "from src import macro\n\n"
                "# TODO: Build quarterly versions of the monthly predictors.\n"
                "# Hint: macro.monthly_to_quarterly(panel_m, how='mean'|'last')\n"
                "panel_q_last = ...\n"
                "panel_q_mean = ...\n"
                "\n"
                "# TODO: Compare them (e.g., correlation of each column)\n"
                "...\n"
            ),
            md("### Your Turn (3): Choose a quarterly feature table"),
            code(
                "# TODO: Choose which aggregation to use for modeling.\n"
                "# Option A: use quarter-end values\n"
                "# Option B: use quarter averages\n"
                "# Option C: keep both by adding suffixes\n"
                "\n"
                "Xq = ...\n"
                "Xq.head()\n"
            ),
            md("### Checkpoint (quarterly index alignment)"),
            code(
                "# TODO: Confirm both Xq and gdp_q use quarter-end timestamps.\n"
                "assert isinstance(Xq.index, pd.DatetimeIndex)\n"
                "assert isinstance(gdp_q.index, pd.DatetimeIndex)\n"
                "assert Xq.index.is_monotonic_increasing\n"
                "assert gdp_q.index.is_monotonic_increasing\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Add lags')}\"></a>\n"
                "## Add lags\n\n"
                "### Goal\n"
                "Add lagged quarterly predictors so the model only uses information available *before* the target period.\n\n"
                "Typical lags to try:\n"
                "- 1 quarter\n"
                "- 2 quarters\n"
                "- 4 quarters (one year)\n"
            ),
            md("### Your Turn (1): Add lag features"),
            code(
                "from src import features\n\n"
                "# TODO: Add lag features for all columns in Xq\n"
                "# Hint: features.add_lag_features(Xq, columns=Xq.columns, lags=[...])\n"
                "Xq_lagged = ...\n"
                "\n"
                "# TODO: Drop rows with NaNs created by lags\n"
                "Xq_lagged = Xq_lagged.dropna().copy()\n"
                "Xq_lagged.head()\n"
            ),
            md("### Checkpoint (no future lags)"),
            code(
                "# TODO: Confirm you used ONLY positive lags.\n"
                "# features.add_lag_features will raise if lags <= 0.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Merge with GDP/labels')}\"></a>\n"
                "## Merge with GDP/labels\n\n"
                "### Goal\n"
                "Join lagged predictors with GDP growth and the next-quarter recession target.\n\n"
                "Key idea:\n"
                "- predictors at time t\n"
                "- target at time t (which is recession at t+1)\n"
            ),
            md("### Your Turn (1): Merge and build the final table"),
            code(
                "# TODO: Join on the quarterly index.\n"
                "# Keep at least:\n"
                "# - gdp growth columns\n"
                "# - recession label\n"
                "# - target_recession_next_q\n"
                "# - lagged predictors\n"
                "\n"
                "df_q = ...\n"
                "\n"
                "# Drop rows with missing target or predictors\n"
                "df_q = df_q.dropna().copy()\n"
                "df_q.head()\n"
            ),
            md("### Checkpoint (target alignment)"),
            code(
                "# TODO: Confirm the target is 0/1 and shifted correctly.\n"
                "assert set(df_q['target_recession_next_q'].unique()).issubset({0, 1})\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Save macro_quarterly.csv')}\"></a>\n"
                "## Save macro_quarterly.csv\n\n"
                "### Goal\n"
                "Write the final modeling table to `data/processed/macro_quarterly.csv`.\n"
            ),
            md("### Your Turn (1): Save + reload"),
            code(
                "out_path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "\n"
                "# TODO: Save df_q\n"
                "...\n"
                "\n"
                "# Reload for sanity\n"
                "df_reload = pd.read_csv(out_path, index_col=0, parse_dates=True)\n"
                "assert df_reload.shape == df_q.shape\n"
                "df_reload.tail()\n"
            ),
        ]

    if spec.path.endswith("04_census_api_microdata_fetch.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Build a county-level micro dataset from the US Census ACS API.\n\n"
                "### Why this matters\n"
                "This micro track is deliberately different from macro time series:\n"
                "- observations are counties (not time)\n"
                "- regression interpretation focuses on cross-sectional relationships\n"
                "- robust SE (HC3) is usually more relevant than time-series HAC\n"
            ),
            md(primer("paths_and_env")),
            md(
                f"<a id=\"{slugify('Browse variables')}\"></a>\n"
                "## Browse variables\n\n"
                "### Goal\n"
                "Learn how ACS variable codes work and choose a starter set.\n\n"
                "We'll focus on a practical starter set:\n"
                "- population\n"
                "- median household income\n"
                "- median gross rent\n"
                "- median home value\n"
                "- poverty count (to build a poverty rate)\n"
                "- labor force / unemployment (to build an unemployment rate)\n"
            ),
            md("### Your Turn (1): Fetch or load variables.json"),
            code(
                "import json\n"
                "from src import census_api\n\n"
                "year = 2022  # TODO: change if you want a different year\n"
                "raw_dir = RAW_DIR / 'census'\n"
                "raw_dir.mkdir(parents=True, exist_ok=True)\n"
                "vars_path = raw_dir / f'variables_{year}.json'\n"
                "\n"
                "# TODO: Load variables metadata.\n"
                "# - If vars_path exists, load it from disk.\n"
                "# - Otherwise, fetch from the API and save it to vars_path.\n"
                "...\n"
            ),
            md("### Your Turn (2): Search for relevant variables"),
            code(
                "# The variables metadata is a nested JSON structure.\n"
                "# TODO: Explore it and search for keywords like:\n"
                "# - 'Median household income'\n"
                "# - 'Median gross rent'\n"
                "# - 'Poverty'\n"
                "# - 'Labor force'\n"
                "\n"
                "# Hint: variables are typically under payload['variables'].\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Fetch county data')}\"></a>\n"
                "## Fetch county data\n\n"
                "### Goal\n"
                "Fetch a county-level table for your chosen variables.\n\n"
                "Default geography:\n"
                "- all counties: `for=county:*`\n"
                "- within all states: `in=state:*`\n"
            ),
            md("### Your Turn (1): Choose a starter variable set"),
            code(
                "# TODO: Use a starter set.\n"
                "# These are commonly-used ACS 5-year estimate codes:\n"
                "acs_vars = [\n"
                "    'NAME',\n"
                "    'B01003_001E',  # total population\n"
                "    'B19013_001E',  # median household income\n"
                "    'B25064_001E',  # median gross rent\n"
                "    'B25077_001E',  # median home value\n"
                "    'B17001_002E',  # count below poverty level\n"
                "    'B23025_002E',  # in labor force\n"
                "    'B23025_005E',  # unemployed\n"
                "]\n"
                "\n"
                "acs_vars\n"
            ),
            md("### Your Turn (2): Fetch the ACS table"),
            code(
                "import pandas as pd\n"
                "from src import census_api\n\n"
                "# TODO: Fetch the data from the API.\n"
                "# Hint: census_api.fetch_acs(year=..., get=..., for_geo='county:*', in_geo='state:*')\n"
                "try:\n"
                "    df_raw = census_api.fetch_acs(year=year, get=acs_vars, for_geo='county:*', in_geo='state:*')\n"
                "except Exception as exc:\n"
                "    df_raw = None\n"
                "    print('Fetch failed, will use sample. Error:', exc)\n"
                "\n"
                "df_raw.head() if df_raw is not None else None\n"
            ),
            md("### Your Turn (3): Fallback to sample"),
            code(
                "import pandas as pd\n\n"
                "# TODO: If df_raw is None, load the sample dataset.\n"
                "if df_raw is None:\n"
                "    df_raw = pd.read_csv(SAMPLE_DIR / 'census_county_sample.csv')\n"
                "\n"
                "df_raw.head()\n"
            ),
            md(
                f"<a id=\"{slugify('Derived rates')}\"></a>\n"
                "## Derived rates\n\n"
                "### Goal\n"
                "Turn raw counts into rates (more comparable across counties).\n\n"
                "You will build:\n"
                "- unemployment_rate = unemployed / labor_force\n"
                "- poverty_rate = below_poverty / population\n"
            ),
            md("### Your Turn (1): Cast numeric columns"),
            code(
                "# TODO: Ensure numeric columns are numeric (some API returns strings).\n"
                "# Hint: pd.to_numeric(..., errors='coerce')\n"
                "...\n"
            ),
            md("### Your Turn (2): Build derived rates safely"),
            code(
                "import numpy as np\n\n"
                "# TODO: Compute rates with safe division.\n"
                "# Replace division-by-zero with NaN.\n"
                "\n"
                "pop = df_raw['B01003_001E'].astype(float)\n"
                "labor_force = df_raw['B23025_002E'].astype(float)\n"
                "unemployed = df_raw['B23025_005E'].astype(float)\n"
                "below_pov = df_raw['B17001_002E'].astype(float)\n"
                "\n"
                "df_raw['unemployment_rate'] = unemployed / labor_force.replace({0: np.nan})\n"
                "df_raw['poverty_rate'] = below_pov / pop.replace({0: np.nan})\n"
                "\n"
                "df_raw[['unemployment_rate', 'poverty_rate']].describe()\n"
            ),
            md(
                f"<a id=\"{slugify('Save processed data')}\"></a>\n"
                "## Save processed data\n\n"
                "### Goal\n"
                "Save a cleaned dataset to `data/processed/census_county_<year>.csv`.\n"
            ),
            md("### Your Turn (1): Save + reload"),
            code(
                "out_path = PROCESSED_DIR / f'census_county_{year}.csv'\n"
                "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "\n"
                "# TODO: Select a useful subset of columns and save.\n"
                "# Suggested: NAME, state, county, raw vars, unemployment_rate, poverty_rate\n"
                "cols = ['NAME', 'state', 'county'] + [c for c in acs_vars if c not in {'NAME'}] + ['unemployment_rate', 'poverty_rate']\n"
                "df_out = df_raw[cols].copy()\n"
                "df_out.to_csv(out_path, index=False)\n"
                "\n"
                "df_check = pd.read_csv(out_path)\n"
                "df_check.head()\n"
            ),
            md("### Checkpoint"),
            code(
                "# TODO: Validate rates are in [0, 1] for most rows.\n"
                "assert (df_out['unemployment_rate'].dropna().between(0, 1).mean() > 0.95)\n"
                "assert (df_out['poverty_rate'].dropna().between(0, 1).mean() > 0.95)\n"
                "...\n"
            ),
        ]

    # Regression notebooks
    if spec.path.endswith("00_single_factor_regression_micro.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Fit a single-factor log-log regression on county data and interpret the coefficient like an elasticity.\n\n"
                "Example question:\n"
                "- \"Across counties, how does rent scale with income?\"\n\n"
                "This is **not** a causal claim by default. It's a structured description of a relationship.\n"
            ),
            md(primer("statsmodels_inference")),
            md(primer("hypothesis_testing")),
            md(
                f"<a id=\"{slugify('Load census data')}\"></a>\n"
                "## Load census data\n\n"
                "### Goal\n"
                "Load a county-level dataset created in the Census notebook.\n\n"
                "If you haven't run the fetch notebook, use the bundled sample.\n"
            ),
            md("### Your Turn (1): Load processed county data (or sample)"),
            code(
                "import pandas as pd\n\n"
                "year = 2022  # TODO: set to the year you fetched\n"
                "path = PROCESSED_DIR / f'census_county_{year}.csv'\n"
                "\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'census_county_sample.csv')\n"
                "\n"
                "df.head()\n"
            ),
            md("### Your Turn (2): Inspect schema"),
            code(
                "# TODO: Inspect columns and dtypes.\n"
                "# Identify which columns represent:\n"
                "# - income\n"
                "# - rent\n"
                "print(df.columns.tolist())\n"
                "print(df.dtypes)\n"
                "...\n"
            ),
            md("### Checkpoint (required columns)"),
            code(
                "# TODO: Confirm the starter variables exist.\n"
                "# If you used different ACS vars, update these names.\n"
                "assert 'B19013_001E' in df.columns  # median household income\n"
                "assert 'B25064_001E' in df.columns  # median gross rent\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Build log variables')}\"></a>\n"
                "## Build log variables\n\n"
                "### Goal\n"
                "Build a clean modeling table with:\n"
                "- `log_income = log(income)`\n"
                "- `log_rent = log(rent)`\n\n"
                "Log-log regression is common in economics because it turns multiplicative relationships into additive ones.\n"
            ),
            md("### Your Turn (1): Clean and log-transform"),
            code(
                "import numpy as np\n\n"
                "# Raw variables\n"
                "income = pd.to_numeric(df['B19013_001E'], errors='coerce')\n"
                "rent = pd.to_numeric(df['B25064_001E'], errors='coerce')\n"
                "\n"
                "# TODO: Drop non-positive values before taking logs\n"
                "mask = (income > 0) & (rent > 0)\n"
                "df_m = pd.DataFrame({\n"
                "    'income': income[mask],\n"
                "    'rent': rent[mask],\n"
                "}).dropna()\n"
                "\n"
                "# TODO: Create log variables\n"
                "df_m['log_income'] = ...\n"
                "df_m['log_rent'] = ...\n"
                "\n"
                "df_m[['log_income', 'log_rent']].head()\n"
            ),
            md("### Your Turn (2): Visualize the relationship"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Make a scatter plot of log_income vs log_rent.\n"
                "# Tip: use alpha=0.2 for dense plots.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Fit OLS + HC3')}\"></a>\n"
                "## Fit OLS + HC3\n\n"
                "### Goal\n"
                "Fit an OLS regression and compute HC3 robust SE (common default for cross-sectional data).\n\n"
                "Model:\n"
                "$$\\log(rent_i) = \\beta_0 + \\beta_1 \\log(income_i) + \\varepsilon_i$$\n"
            ),
            md("### Your Turn (1): Fit OLS + HC3"),
            code(
                "from src import econometrics\n\n"
                "# TODO: Fit OLS with HC3 robust SE using the helper.\n"
                "res = econometrics.fit_ols_hc3(df_m, y_col='log_rent', x_cols=['log_income'])\n"
                "print(res.summary())\n"
            ),
            md("### Your Turn (2): Extract the slope and interpret it"),
            code(
                "# TODO: Extract coefficient and CI for log_income.\n"
                "beta = float(res.params['log_income'])\n"
                "ci = res.conf_int().loc['log_income'].tolist()\n"
                "\n"
                "print('beta:', beta)\n"
                "print('95% CI:', ci)\n"
                "\n"
                "# Interpretation prompt:\n"
                "# In a log-log model, beta is approximately an elasticity.\n"
                "# Example: a 1% increase in income is associated with about beta% higher rent.\n"
                "# TODO: Compute the implied change for a 10% income increase.\n"
                "pct_income = 10.0\n"
                "approx_pct_rent = ...\n"
                "approx_pct_rent\n"
            ),
            md(
                f"<a id=\"{slugify('Interpretation')}\"></a>\n"
                "## Interpretation\n\n"
                "Write a short interpretation (5-8 sentences) that answers:\n"
                "- What is the estimated relationship?\n"
                "- Is it statistically distinguishable from 0 (given assumptions)?\n"
                "- Is it economically large?\n"
                "- What would have to be true for a causal interpretation?\n"
            ),
            md("### Your Turn: Write your interpretation"),
            code("# TODO: Write your interpretation as a Python multiline string.\nnotes = \"\"\"\n...\n\"\"\"\nprint(notes)\n"),
        ]

    if spec.path.endswith("01_multifactor_regression_micro_controls.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Fit a multi-factor regression on county data, add controls, and discuss omitted variable bias (OVB).\n\n"
                "Big idea:\n"
                "- The coefficient on income can change when you add controls.\n"
                "- That change is a clue that the simple model was absorbing other effects.\n"
            ),
            md(primer("statsmodels_inference")),
            md(primer("hypothesis_testing")),
            md(
                f"<a id=\"{slugify('Choose controls')}\"></a>\n"
                "## Choose controls\n\n"
                "### Goal\n"
                "Pick a set of plausible controls to include alongside income.\n\n"
                "Starter controls (if you have them):\n"
                "- `poverty_rate`\n"
                "- `unemployment_rate`\n"
                "- `log_population`\n"
                "- `log_home_value` (if available)\n"
            ),
            md("### Your Turn (1): Load data and build baseline variables"),
            code(
                "import numpy as np\n"
                "import pandas as pd\n\n"
                "year = 2022\n"
                "path = PROCESSED_DIR / f'census_county_{year}.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'census_county_sample.csv')\n"
                "\n"
                "# Outcome + main regressor\n"
                "income = pd.to_numeric(df['B19013_001E'], errors='coerce')\n"
                "rent = pd.to_numeric(df['B25064_001E'], errors='coerce')\n"
                "\n"
                "# TODO: Build baseline log variables\n"
                "mask = (income > 0) & (rent > 0)\n"
                "df_m = pd.DataFrame({\n"
                "    'log_income': np.log(income[mask]),\n"
                "    'log_rent': np.log(rent[mask]),\n"
                "}).dropna()\n"
                "\n"
                "# TODO: Merge in control columns you want to use (from df)\n"
                "# Hint: df_m = df_m.join(...)\n"
                "...\n"
            ),
            md("### Your Turn (2): Choose controls list"),
            code(
                "# TODO: Choose at least 2 controls available in df_m.\n"
                "# Example: ['poverty_rate', 'unemployment_rate']\n"
                "controls = [\n"
                "    ...,\n"
                "]\n"
                "\n"
                "controls\n"
            ),
            md(
                f"<a id=\"{slugify('Fit model')}\"></a>\n"
                "## Fit model\n\n"
                "### Goal\n"
                "Fit:\n"
                "1) baseline model: log_rent ~ log_income\n"
                "2) controlled model: log_rent ~ log_income + controls\n"
            ),
            md("### Your Turn (1): Fit baseline and controlled models (HC3)"),
            code(
                "from src import econometrics\n\n"
                "# Baseline\n"
                "res_base = econometrics.fit_ols_hc3(df_m, y_col='log_rent', x_cols=['log_income'])\n"
                "\n"
                "# Controlled\n"
                "x_cols = ['log_income'] + controls\n"
                "res_ctrl = econometrics.fit_ols_hc3(df_m, y_col='log_rent', x_cols=x_cols)\n"
                "\n"
                "print(res_base.summary())\n"
                "print(res_ctrl.summary())\n"
            ),
            md("### Optional: cluster-robust SE by state"),
            code(
                "# Advanced (optional): if you have a 'state' column, research statsmodels cluster SE.\n"
                "# The idea: errors may be correlated within a state.\n"
                "# TODO: Try cov_type='cluster' and compare SE to HC3.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Compare coefficients')}\"></a>\n"
                "## Compare coefficients\n\n"
                "### Goal\n"
                "Compare the income coefficient with and without controls.\n\n"
                "Interpretation prompt:\n"
                "- If the coefficient changes a lot, what omitted factors might income have been proxying for?\n"
            ),
            md("### Your Turn (1): Build a comparison table"),
            code(
                "import pandas as pd\n\n"
                "def coef_row(res, name):\n"
                "    return pd.Series({\n"
                "        'coef': float(res.params[name]),\n"
                "        'se': float(res.bse[name]),\n"
                "        'p': float(res.pvalues[name]),\n"
                "    })\n"
                "\n"
                "comp = pd.DataFrame({\n"
                "    'baseline': coef_row(res_base, 'log_income'),\n"
                "    'controlled': coef_row(res_ctrl, 'log_income'),\n"
                "}).T\n"
                "comp\n"
            ),
            md("### Your Turn (2): Multicollinearity check (VIF)"),
            code(
                "from src.econometrics import vif_table\n\n"
                "# TODO: Compute VIF for the controlled model predictors (excluding the intercept).\n"
                "vif = vif_table(df_m.dropna(), x_cols)\n"
                "vif\n"
            ),
        ]

    if spec.path.endswith("02_single_factor_regression_macro.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Fit a classic single-factor macro regression: GDP growth vs yield curve spread.\n\n"
                "This is a great first macro regression because:\n"
                "- it is easy to visualize,\n"
                "- it has a well-known economic story,\n"
                "- it demonstrates why time-series inference (HAC SE) matters.\n"
            ),
            md(primer("pandas_time_series")),
            md(primer("statsmodels_inference")),
            md(primer("hypothesis_testing")),
            md(
                f"<a id=\"{slugify('Load macro data')}\"></a>\n"
                "## Load macro data\n\n"
                "### Goal\n"
                "Load the quarterly macro table produced earlier (`macro_quarterly.csv`).\n\n"
                "If you haven't built it yet, use the bundled sample.\n"
            ),
            md("### Your Turn (1): Load macro_quarterly.csv (or sample)"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "df.head()\n"
            ),
            md("### Your Turn (2): Choose target and predictor"),
            code(
                "# Target: GDP growth\n"
                "y_col = 'gdp_growth_qoq'\n"
                "\n"
                "# Predictor: yield curve spread (try lagged)\n"
                "# TODO: Try 'T10Y2Y_lag1' first.\n"
                "x_cols = ['T10Y2Y_lag1']\n"
                "\n"
                "# Build modeling table\n"
                "df_m = df[[y_col] + x_cols].dropna().copy()\n"
                "df_m.tail()\n"
            ),
            md("### Checkpoint (time order + no NaNs)"),
            code(
                "assert df_m.index.is_monotonic_increasing\n"
                "assert not df_m.isna().any().any()\n"
                "assert df_m.shape[0] > 30\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Fit OLS')}\"></a>\n"
                "## Fit OLS\n\n"
                "### Goal\n"
                "Fit OLS on a time-based train/test split and evaluate out-of-sample error.\n"
            ),
            md("### Your Turn (1): Time split"),
            code(
                "from src.evaluation import time_train_test_split_index\n\n"
                "# TODO: Create a time split (first 80% train, last 20% test)\n"
                "split = time_train_test_split_index(len(df_m), test_size=0.2)\n"
                "train = df_m.iloc[split.train_slice]\n"
                "test = df_m.iloc[split.test_slice]\n"
                "\n"
                "train.index.max(), test.index.min()\n"
            ),
            md("### Your Turn (2): Fit OLS on train and evaluate on test"),
            code(
                "import statsmodels.api as sm\n"
                "from src.evaluation import regression_metrics\n\n"
                "# Build design matrices\n"
                "X_tr = sm.add_constant(train[x_cols], has_constant='add')\n"
                "y_tr = train[y_col]\n"
                "X_te = sm.add_constant(test[x_cols], has_constant='add')\n"
                "y_te = test[y_col]\n"
                "\n"
                "# Fit\n"
                "res_ols = sm.OLS(y_tr, X_tr).fit()\n"
                "y_hat = res_ols.predict(X_te)\n"
                "\n"
                "metrics = regression_metrics(y_te.to_numpy(), y_hat.to_numpy())\n"
                "metrics\n"
            ),
            md(
                f"<a id=\"{slugify('Fit HAC')}\"></a>\n"
                "## Fit HAC\n\n"
                "### Goal\n"
                "Compare naive OLS standard errors to HAC/Newey-West robust standard errors.\n\n"
                "Key idea:\n"
                "- coefficients can stay the same\n"
                "- p-values and confidence intervals can change (sometimes a lot)\n"
            ),
            md("### Your Turn (1): Fit HAC with different maxlags"),
            code(
                "from src import econometrics\n\n"
                "# TODO: Fit HAC on the FULL sample (inference focus) with different maxlags.\n"
                "res_naive = econometrics.fit_ols(df_m, y_col=y_col, x_cols=x_cols)\n"
                "res_hac1 = econometrics.fit_ols_hac(df_m, y_col=y_col, x_cols=x_cols, maxlags=1)\n"
                "res_hac4 = econometrics.fit_ols_hac(df_m, y_col=y_col, x_cols=x_cols, maxlags=4)\n"
                "\n"
                "print('naive p:', res_naive.pvalues)\n"
                "print('hac1  p:', res_hac1.pvalues)\n"
                "print('hac4  p:', res_hac4.pvalues)\n"
            ),
            md("### Your Turn (2): Compare confidence intervals"),
            code(
                "# TODO: Compare CI for the yield spread coefficient under naive vs HAC.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Interpretation')}\"></a>\n"
                "## Interpretation\n\n"
                "Write a short interpretation (8-12 sentences):\n"
                "- What sign do you expect for the yield spread coefficient, and why?\n"
                "- What does a 1 percentage-point change in spread mean for predicted GDP growth (units!)?\n"
                "- How does your inference change under HAC SE?\n"
                "- What limitations do you see (endogeneity, omitted variables, regime changes)?\n"
            ),
            md("### Your Turn: Write your interpretation"),
            code("notes = \"\"\"\n...\n\"\"\"\nprint(notes)\n"),
        ]

    if spec.path.endswith("03_multifactor_regression_macro.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Fit a multi-factor GDP growth regression and learn how to interpret feature weights *carefully*.\n\n"
                "This notebook is where multicollinearity becomes real:\n"
                "- many macro indicators move together\n"
                "- coefficients can change sign or become unstable when features are correlated\n"
            ),
            md(primer("pandas_time_series")),
            md(primer("statsmodels_inference")),
            md(primer("hypothesis_testing")),
            md(
                f"<a id=\"{slugify('Choose features')}\"></a>\n"
                "## Choose features\n\n"
                "### Goal\n"
                "Pick a feature set to predict GDP growth.\n\n"
                "Recommendations:\n"
                "- Start small (3-6 predictors) before you go wide.\n"
                "- Prefer lagged predictors (information available before the quarter).\n"
                "- Keep a record of your feature list.\n"
            ),
            md("### Your Turn (1): Load macro data"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "df.head()\n"
            ),
            md("### Your Turn (2): Choose a target + feature list"),
            code(
                "# Target\n"
                "y_col = 'gdp_growth_qoq'\n"
                "\n"
                "# TODO: Choose features.\n"
                "# Start with lagged predictors to reduce timing ambiguity.\n"
                "x_cols = [\n"
                "    'T10Y2Y_lag1',\n"
                "    'UNRATE_lag1',\n"
                "    'FEDFUNDS_lag1',\n"
                "    # TODO: add 1-3 more\n"
                "]\n"
                "\n"
                "df_m = df[[y_col] + x_cols].dropna().copy()\n"
                "df_m.tail()\n"
            ),
            md("### Checkpoint (feature table)"),
            code(
                "assert df_m.index.is_monotonic_increasing\n"
                "assert not df_m.isna().any().any()\n"
                "assert df_m.shape[0] > 30\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Fit model')}\"></a>\n"
                "## Fit model\n\n"
                "### Goal\n"
                "Fit a multi-factor regression and compare:\n"
                "- raw coefficients (units matter)\n"
                "- standardized coefficients (compare relative importance)\n"
            ),
            md("### Your Turn (1): Time split and fit OLS"),
            code(
                "import statsmodels.api as sm\n"
                "from src.evaluation import time_train_test_split_index, regression_metrics\n\n"
                "split = time_train_test_split_index(len(df_m), test_size=0.2)\n"
                "train = df_m.iloc[split.train_slice]\n"
                "test = df_m.iloc[split.test_slice]\n"
                "\n"
                "X_tr = sm.add_constant(train[x_cols], has_constant='add')\n"
                "y_tr = train[y_col]\n"
                "X_te = sm.add_constant(test[x_cols], has_constant='add')\n"
                "y_te = test[y_col]\n"
                "\n"
                "res = sm.OLS(y_tr, X_tr).fit()\n"
                "y_hat = res.predict(X_te)\n"
                "\n"
                "regression_metrics(y_te.to_numpy(), y_hat.to_numpy())\n"
            ),
            md("### Your Turn (2): Standardize predictors and compare standardized coefficients"),
            code(
                "from sklearn.preprocessing import StandardScaler\n\n"
                "# Standardize X (train-fitted scaler!)\n"
                "sc = StandardScaler().fit(train[x_cols])\n"
                "X_tr_s = sc.transform(train[x_cols])\n"
                "X_te_s = sc.transform(test[x_cols])\n"
                "\n"
                "# Refit on standardized features\n"
                "X_tr_s = sm.add_constant(X_tr_s, has_constant='add')\n"
                "X_te_s = sm.add_constant(X_te_s, has_constant='add')\n"
                "res_s = sm.OLS(y_tr, X_tr_s).fit()\n"
                "\n"
                "# TODO: Map standardized coefficients back to feature names (excluding intercept)\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('VIF + stability')}\"></a>\n"
                "## VIF + stability\n\n"
                "### Goal\n"
                "Measure multicollinearity and see whether coefficients are stable.\n\n"
                "Two simple stability checks:\n"
                "- VIF (collinearity)\n"
                "- fit on different eras and compare coefficients\n"
            ),
            md("### Your Turn (1): VIF table"),
            code(
                "from src.econometrics import vif_table\n\n"
                "# TODO: Compute VIF on the full feature matrix (no intercept).\n"
                "vif = vif_table(df_m, x_cols)\n"
                "vif\n"
            ),
            md("### Your Turn (2): Era split coefficient stability"),
            code(
                "import pandas as pd\n"
                "import statsmodels.api as sm\n\n"
                "# TODO: Fit the same model on an early era vs a late era and compare coefficients.\n"
                "mid = int(len(df_m) * 0.5)\n"
                "early = df_m.iloc[:mid]\n"
                "late = df_m.iloc[mid:]\n"
                "\n"
                "res_early = sm.OLS(early[y_col], sm.add_constant(early[x_cols], has_constant='add')).fit()\n"
                "res_late = sm.OLS(late[y_col], sm.add_constant(late[x_cols], has_constant='add')).fit()\n"
                "\n"
                "comp = pd.DataFrame({'early': res_early.params, 'late': res_late.params})\n"
                "comp\n"
            ),
        ]

    if spec.path.endswith("04_inference_time_series_hac.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Learn why naive OLS standard errors often break in macro time series, and how HAC/Newey-West helps.\n\n"
                "Big idea:\n"
                "- coefficients answer \"what best fits the line\"\n"
                "- standard errors answer \"how uncertain are we about the coefficients\"\n\n"
                "Time series often violate the assumptions behind naive SE.\n"
            ),
            md(primer("statsmodels_inference")),
            md(primer("hypothesis_testing")),
            md(
                f"<a id=\"{slugify('Assumptions')}\"></a>\n"
                "## Assumptions\n\n"
                "### Goal\n"
                "Fit a baseline regression and inspect residuals.\n\n"
                "Reminder:\n"
                "- OLS coefficients can be computed even when assumptions fail.\n"
                "- Inference (SE / p-values / CI) is what becomes unreliable.\n"
            ),
            md("### Your Turn (1): Load macro data and pick a simple regression"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "y_col = 'gdp_growth_qoq'\n"
                "x_cols = ['T10Y2Y_lag1']\n"
                "\n"
                "df_m = df[[y_col] + x_cols].dropna().copy()\n"
                "df_m.tail()\n"
            ),
            md("### Your Turn (2): Fit OLS and compute residuals"),
            code(
                "import statsmodels.api as sm\n\n"
                "X = sm.add_constant(df_m[x_cols], has_constant='add')\n"
                "y = df_m[y_col]\n"
                "res = sm.OLS(y, X).fit()\n"
                "\n"
                "# Residuals = y - y_hat\n"
                "resid = res.resid\n"
                "fitted = res.fittedvalues\n"
                "\n"
                "print(res.summary())\n"
                "resid.head()\n"
            ),
            md("### Your Turn (3): Residual diagnostics plots (simple but useful)"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Plot residuals over time\n"
                "# TODO: Plot residuals vs fitted values (heteroskedasticity visual check)\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Autocorrelation')}\"></a>\n"
                "## Autocorrelation\n\n"
                "### Goal\n"
                "Measure whether residuals are correlated over time.\n\n"
                "If residuals are autocorrelated, naive SE can be too small.\n"
            ),
            md("### Your Turn (1): Autocorrelation by lag"),
            code(
                "# TODO: Compute residual autocorrelation at lags 1..8.\n"
                "# Hint: resid.autocorr(lag=k)\n"
                "ac = {k: float(resid.autocorr(lag=k)) for k in range(1, 9)}\n"
                "ac\n"
            ),
            md("### Your Turn (2): A simple ACF plot"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Plot the autocorrelation values as a bar chart.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('HAC SE')}\"></a>\n"
                "## HAC SE\n\n"
                "### Goal\n"
                "Fit the same model but compute HAC/Newey-West robust SE.\n\n"
                "Your task:\n"
                "- compare coefficient (should be the same)\n"
                "- compare SE/p-values (can differ)\n"
            ),
            md("### Your Turn (1): Compare naive vs HAC with different maxlags"),
            code(
                "from src import econometrics\n\n"
                "res_naive = econometrics.fit_ols(df_m, y_col=y_col, x_cols=x_cols)\n"
                "res_hac1 = econometrics.fit_ols_hac(df_m, y_col=y_col, x_cols=x_cols, maxlags=1)\n"
                "res_hac4 = econometrics.fit_ols_hac(df_m, y_col=y_col, x_cols=x_cols, maxlags=4)\n"
                "\n"
                "# TODO: Print SE/p-values side-by-side for the slope coefficient.\n"
                "...\n"
            ),
            md("### Your Turn (2): Write a careful interpretation"),
            code(
                "# TODO: Write 6-10 sentences answering:\n"
                "# - Did HAC increase or decrease your SE?\n"
                "# - How did that affect your p-value/CI?\n"
                "# - What assumptions are still required even with HAC?\n"
                "notes = \"\"\"\n...\n\"\"\"\nprint(notes)\n"
            ),
        ]

    if spec.path.endswith("05_regularization_ridge_lasso.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Use ridge and lasso regression to handle correlated macro predictors.\n\n"
                "Why this notebook exists:\n"
                "- OLS coefficients can be unstable when predictors are correlated.\n"
                "- Ridge shrinks coefficients smoothly.\n"
                "- Lasso can set some coefficients exactly to 0 (feature selection-ish).\n"
            ),
            md(primer("sklearn_pipelines")),
            md(
                f"<a id=\"{slugify('Build feature matrix')}\"></a>\n"
                "## Build feature matrix\n\n"
                "### Goal\n"
                "Choose a target and feature set from the macro quarterly table.\n"
            ),
            md("### Your Turn (1): Load data and pick columns"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "y_col = 'gdp_growth_qoq'\n"
                "\n"
                "# TODO: Choose a feature list.\n"
                "# Tip: start with lagged features to avoid timing ambiguity.\n"
                "x_cols = [\n"
                "    'T10Y2Y_lag1',\n"
                "    'UNRATE_lag1',\n"
                "    'FEDFUNDS_lag1',\n"
                "    'INDPRO_lag1',\n"
                "    'RSAFS_lag1',\n"
                "    # TODO: add more lags/features if you want\n"
                "]\n"
                "\n"
                "df_m = df[[y_col] + x_cols].dropna().copy()\n"
                "df_m.tail()\n"
            ),
            md("### Your Turn (2): Time split"),
            code(
                "from src.evaluation import time_train_test_split_index\n\n"
                "split = time_train_test_split_index(len(df_m), test_size=0.2)\n"
                "train = df_m.iloc[split.train_slice]\n"
                "test = df_m.iloc[split.test_slice]\n"
                "\n"
                "X_train = train[x_cols]\n"
                "y_train = train[y_col]\n"
                "X_test = test[x_cols]\n"
                "y_test = test[y_col]\n"
                "\n"
                "X_train.shape, X_test.shape\n"
            ),
            md(
                f"<a id=\"{slugify('Fit ridge/lasso')}\"></a>\n"
                "## Fit ridge/lasso\n\n"
                "### Goal\n"
                "Fit ridge and lasso over a range of regularization strengths and compare out-of-sample error.\n"
            ),
            md("### Your Turn (1): Fit ridge and lasso across alpha grid"),
            code(
                "import numpy as np\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import Ridge, Lasso\n"
                "from sklearn.metrics import mean_squared_error\n\n"
                "alphas = np.logspace(-3, 2, 20)\n"
                "\n"
                "ridge_rmse = []\n"
                "lasso_rmse = []\n"
                "\n"
                "for a in alphas:\n"
                "    ridge = Pipeline([\n"
                "        ('scaler', StandardScaler()),\n"
                "        ('model', Ridge(alpha=float(a))),\n"
                "    ])\n"
                "    lasso = Pipeline([\n"
                "        ('scaler', StandardScaler()),\n"
                "        ('model', Lasso(alpha=float(a), max_iter=20000)),\n"
                "    ])\n"
                "\n"
                "    ridge.fit(X_train, y_train)\n"
                "    lasso.fit(X_train, y_train)\n"
                "\n"
                "    ridge_pred = ridge.predict(X_test)\n"
                "    lasso_pred = lasso.predict(X_test)\n"
                "\n"
                "    ridge_rmse.append(mean_squared_error(y_test, ridge_pred, squared=False))\n"
                "    lasso_rmse.append(mean_squared_error(y_test, lasso_pred, squared=False))\n"
                "\n"
                "best_ridge = float(alphas[int(np.argmin(ridge_rmse))])\n"
                "best_lasso = float(alphas[int(np.argmin(lasso_rmse))])\n"
                "best_ridge, best_lasso\n"
            ),
            md("### Your Turn (2): Plot RMSE vs alpha"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Plot ridge_rmse and lasso_rmse vs alphas (log scale).\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Coefficient paths')}\"></a>\n"
                "## Coefficient paths\n\n"
                "### Goal\n"
                "Visualize how coefficients shrink as regularization increases.\n\n"
                "This is one of the best ways to build intuition for what ridge/lasso are doing.\n"
            ),
            md("### Your Turn (1): Fit models and record coefficients across alphas"),
            code(
                "import pandas as pd\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import Ridge, Lasso\n\n"
                "ridge_coefs = []\n"
                "lasso_coefs = []\n"
                "\n"
                "for a in alphas:\n"
                "    ridge = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=float(a)))])\n"
                "    lasso = Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=float(a), max_iter=20000))])\n"
                "    ridge.fit(X_train, y_train)\n"
                "    lasso.fit(X_train, y_train)\n"
                "\n"
                "    ridge_coefs.append(ridge.named_steps['model'].coef_)\n"
                "    lasso_coefs.append(lasso.named_steps['model'].coef_)\n"
                "\n"
                "ridge_coefs = pd.DataFrame(ridge_coefs, columns=x_cols, index=alphas)\n"
                "lasso_coefs = pd.DataFrame(lasso_coefs, columns=x_cols, index=alphas)\n"
                "\n"
                "ridge_coefs.head()\n"
            ),
            md("### Your Turn (2): Plot coefficient paths"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Plot coefficient paths for ridge and lasso.\n"
                "# Hint: loop over columns and plot series on same axes.\n"
                "...\n"
            ),
        ]

    if spec.path.endswith("06_rolling_regressions_stability.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Use rolling regressions to see how relationships change over time.\n\n"
                "This is a realism check:\n"
                "- A coefficient that is stable across decades is rare in macro.\n"
                "- If coefficients drift, you should be cautious about \"the\" relationship.\n"
            ),
            md(primer("pandas_time_series")),
            md(primer("statsmodels_inference")),
            md(
                f"<a id=\"{slugify('Rolling regression')}\"></a>\n"
                "## Rolling regression\n\n"
                "### Goal\n"
                "Fit the same regression repeatedly on a moving window.\n\n"
                "We will start with a simple model:\n"
                "- GDP growth ~ yield curve spread (lagged)\n"
            ),
            md("### Your Turn (1): Load data and set up the window"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "y_col = 'gdp_growth_qoq'\n"
                "x_col = 'T10Y2Y_lag1'\n"
                "\n"
                "df_m = df[[y_col, x_col, 'recession']].dropna().copy()\n"
                "\n"
                "# Rolling window length in quarters\n"
                "window = 40  # ~10 years\n"
                "df_m.head()\n"
            ),
            md("### Your Turn (2): Fit rolling windows and collect coefficients"),
            code(
                "import numpy as np\n"
                "import statsmodels.api as sm\n\n"
                "rows = []\n"
                "for end in range(window, len(df_m) + 1):\n"
                "    chunk = df_m.iloc[end - window : end]\n"
                "    X = sm.add_constant(chunk[[x_col]], has_constant='add')\n"
                "    y = chunk[y_col]\n"
                "    res = sm.OLS(y, X).fit()\n"
                "\n"
                "    # Record the coefficient on x_col and a simple CI\n"
                "    beta = float(res.params[x_col])\n"
                "    ci_low, ci_high = res.conf_int().loc[x_col].tolist()\n"
                "    rows.append({\n"
                "        'date': chunk.index.max(),\n"
                "        'beta': beta,\n"
                "        'ci_low': float(ci_low),\n"
                "        'ci_high': float(ci_high),\n"
                "    })\n"
                "\n"
                "roll = pd.DataFrame(rows).set_index('date')\n"
                "roll.head()\n"
            ),
            md(
                f"<a id=\"{slugify('Coefficient drift')}\"></a>\n"
                "## Coefficient drift\n\n"
                "### Goal\n"
                "Visualize coefficient stability over time.\n"
            ),
            md("### Your Turn (1): Plot coefficient + CI over time"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Plot roll['beta'] and a shaded CI band.\n"
                "...\n"
            ),
            md("### Your Turn (2): Summarize coefficient distribution"),
            code(
                "# TODO: Compute summary stats for beta.\n"
                "# Identify periods where the sign changed.\n"
                "roll['beta'].describe()\n"
            ),
            md(
                f"<a id=\"{slugify('Regime interpretation')}\"></a>\n"
                "## Regime interpretation\n\n"
                "### Goal\n"
                "Compare coefficient drift to recession periods.\n\n"
                "This is not proof of causality.\n"
                "It is a structured way to ask: \"does the relationship change during recessions or different eras?\"\n"
            ),
            md("### Your Turn (1): Overlay recession shading (simple)"),
            code(
                "# TODO: Create a recession indicator aligned to roll index.\n"
                "# Hint: use df_m['recession'] reindexed to roll.index\n"
                "...\n"
            ),
            md("### Reflection"),
            md(
                "- When does the sign or magnitude change?\n"
                "- What macro regimes might explain it?\n"
                "- If you were building a model, would you trust one fixed coefficient?\n"
            ),
        ]

    # Causal inference notebooks
    if spec.path.endswith("00_build_census_county_panel.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Build a multi-year county dataset suitable for panel methods (FE/DiD).\n\n"
                "Important framing:\n"
                "- This is **not** a panel of the same individuals.\n"
                "- It is repeated cross-sections summarized at the county level.\n"
                "- Panel methods can still be useful, but interpretation must be careful.\n"
            ),
            md(primer("paths_and_env")),
            md(
                f"<a id=\"{slugify('Choose years + variables')}\"></a>\n"
                "## Choose years + variables\n\n"
                "### Background\n"
                "This project treats a dataset config as a **contract**:\n"
                "- which years are included,\n"
                "- which variables are fetched,\n"
                "- and which geography level the rows represent.\n\n"
                "ACS variable names look like codes (e.g., `B19013_001E`). That is normal.\n"
                "Your job is to keep a small “data dictionary” as you go: what each code measures and what the units are.\n\n"
                "### What you should see\n"
                "- `years` is a list of years (default: 2014–2022).\n"
                "- `acs_vars` is a list of ACS variable codes.\n"
                "- `geo_for`/`geo_in` describe a county-within-state query.\n\n"
                "### Interpretation prompts\n"
                "- Pick 2 ACS variables and write (in words) what they measure.\n"
                "- Which variables will be numerators vs denominators for rates?\n\n"
                "### Goal\n"
                "Load a default panel config (`configs/census_panel.yaml`) and inspect:\n"
                "- years\n"
                "- ACS variables\n"
                "- geography\n"
            ),
            md("### Your Turn: Load the panel config"),
            code(
                "import yaml\n\n"
                "cfg_path = PROJECT_ROOT / 'configs' / 'census_panel.yaml'\n"
                "cfg = yaml.safe_load(cfg_path.read_text())\n"
                "\n"
                "acs = cfg['acs_panel']\n"
                "years = list(acs['years'])\n"
                "dataset = acs.get('dataset', 'acs/acs5')\n"
                "acs_vars = list(acs['get'])\n"
                "geo_for = acs['geography']['for']\n"
                "geo_in = acs['geography'].get('in')\n"
                "\n"
                "years[:5], acs_vars\n"
            ),
            md(
                f"<a id=\"{slugify('Fetch/cache ACS tables')}\"></a>\n"
                "## Fetch/cache ACS tables\n\n"
                "### Background\n"
                "In applied work, you almost never want to hit an API repeatedly during experiments.\n"
                "So we cache raw pulls under `data/raw/` and build a clean panel under `data/processed/`.\n\n"
                "This notebook is offline-first:\n"
                "- if cached raw CSVs exist, we load them,\n"
                "- otherwise we fall back to the bundled sample panel.\n\n"
                "### What you should see\n"
                "- Either `frames` is non-empty (cached raw CSVs found), or you see a message that the sample panel is used.\n"
                "- `panel_raw` contains county rows with columns like `state`, `county`, and ACS variables.\n\n"
                "### Interpretation prompts\n"
                "- Where on disk is the cache for a given year stored?\n"
                "- What would you change in the config to add/remove variables?\n\n"
                "### Goal\n"
                "For each year, load a cached raw CSV if available; otherwise fetch from the Census API.\n\n"
                "Offline default:\n"
                "- If nothing is cached, use `data/sample/census_county_panel_sample.csv`.\n"
            ),
            md("### Your Turn: Load cached tables or fall back to sample"),
            code(
                "import pandas as pd\n"
                "from src import census_api\n\n"
                "raw_dir = RAW_DIR / 'census'\n"
                "raw_dir.mkdir(parents=True, exist_ok=True)\n"
                "\n"
                "frames = []\n"
                "for year in years:\n"
                "    p = raw_dir / f'acs_county_{int(year)}.csv'\n"
                "    if p.exists():\n"
                "        df_y = pd.read_csv(p)\n"
                "        frames.append((int(year), df_y))\n"
                "    else:\n"
                "        # TODO (optional): fetch and cache.\n"
                "        # df_y = census_api.fetch_acs(year=int(year), dataset=dataset, get=acs_vars, for_geo=geo_for, in_geo=geo_in)\n"
                "        # df_y.to_csv(p, index=False)\n"
                "        # frames.append((int(year), df_y))\n"
                "        pass\n"
                "\n"
                "if not frames:\n"
                "    print('No cached raw CSVs found. Using bundled sample panel.')\n"
                "    panel_raw = pd.read_csv(SAMPLE_DIR / 'census_county_panel_sample.csv')\n"
                "else:\n"
                "    # Attach year and concatenate\n"
                "    tmp = []\n"
                "    for year, df_y in frames:\n"
                "        df_y = df_y.copy()\n"
                "        df_y['year'] = year\n"
                "        tmp.append(df_y)\n"
                "    panel_raw = pd.concat(tmp, ignore_index=True)\n"
                "\n"
                "panel_raw.head()\n"
            ),
            md(
                f"<a id=\"{slugify('Build panel + FIPS')}\"></a>\n"
                "## Build panel + FIPS\n\n"
                "### Background\n"
                "Panel methods require stable unit identifiers.\n"
                "For U.S. counties, a standard identifier is **FIPS**:\n"
                "- 2-digit state code + 3-digit county code.\n\n"
                "We also build key derived outcomes as rates so later regressions have consistent units.\n\n"
                "### What you should see\n"
                "- `fips` is a 5-character string.\n"
                "- `year` is an integer.\n"
                "- `poverty_rate` and `unemployment_rate` are usually between 0 and 1.\n"
                "- the DataFrame has a MultiIndex `('fips','year')` and is sorted.\n\n"
                "### Interpretation prompts\n"
                "- Why do we `zfill` the state/county codes?\n"
                "- If a rate is outside [0, 1], what data issues could cause it?\n\n"
                "### Goal\n"
                "Create stable identifiers and derived rates:\n"
                "- `fips` = state (2-digit) + county (3-digit)\n"
                "- `unemployment_rate`, `poverty_rate`\n"
            ),
            md("### Your Turn: Clean geo ids, build fips, derived rates"),
            code(
                "import pandas as pd\n\n"
                "df = panel_raw.copy()\n"
                "\n"
                "# Geo ids\n"
                "df['state'] = df['state'].astype(str).str.zfill(2)\n"
                "df['county'] = df['county'].astype(str).str.zfill(3)\n"
                "df['fips'] = df['state'] + df['county']\n"
                "df['year'] = df['year'].astype(int)\n"
                "\n"
                "# Derived rates (safe guards)\n"
                "df['unemployment_rate'] = (\n"
                "    df['B23025_005E'].astype(float) / df['B23025_002E'].replace({0: pd.NA}).astype(float)\n"
                ").astype(float)\n"
                "df['poverty_rate'] = (\n"
                "    df['B17001_002E'].astype(float) / df['B01003_001E'].replace({0: pd.NA}).astype(float)\n"
                ").astype(float)\n"
                "\n"
                "# Panel index (PanelOLS-ready)\n"
                "panel = df.set_index(['fips', 'year'], drop=False).sort_index()\n"
                "\n"
                "panel[['state', 'county', 'fips', 'year', 'unemployment_rate', 'poverty_rate']].head()\n"
            ),
            md(
                f"<a id=\"{slugify('Save processed panel')}\"></a>\n"
                "## Save processed panel\n\n"
                "### Background\n"
                "This file is the handoff between the data pipeline and the causal notebooks.\n"
                "Once you write `data/processed/census_county_panel.csv`, later notebooks can run without rebuilding the panel.\n\n"
                "### What you should see\n"
                "- a new file at `data/processed/census_county_panel.csv`.\n"
                "- reloading the file produces a non-empty DataFrame.\n\n"
                "### Interpretation prompts\n"
                "- What columns are essential for later FE/DiD notebooks?\n"
                "- What would you add to the panel if you wanted a richer causal story?\n\n"
                "### Goal\n"
                "Write a panel dataset to `data/processed/census_county_panel.csv`.\n"
            ),
            md("### Your Turn: Save + reload"),
            code(
                "out_path = PROCESSED_DIR / 'census_county_panel.csv'\n"
                "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "panel.to_csv(out_path, index=True)\n"
                "\n"
                "print('wrote', out_path)\n"
                "\n"
                "# Quick reload\n"
                "check = pd.read_csv(out_path)\n"
                "check.head()\n"
            ),
        ]

    if spec.path.endswith("01_panel_fixed_effects_clustered_se.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Compare:\n"
                "- pooled OLS (ignores panel structure)\n"
                "- two-way fixed effects (county FE + year FE)\n"
                "- robust vs clustered standard errors\n\n"
                "This is still not causal by default. FE helps control time-invariant confounding, not everything.\n"
            ),
            md(primer("linearmodels_panel_iv")),
            md(
                f"<a id=\"{slugify('Load panel and define variables')}\"></a>\n"
                "## Load panel and define variables\n\n"
                "### Background\n"
                "Panel regressions expect a clear unit index (county) and time index (year).\n"
                "Before modeling, we build a **small, typed modeling table**:\n"
                "- confirm `fips` and `year`,\n"
                "- set a MultiIndex,\n"
                "- create a few interpretable transforms (like logs).\n\n"
                "Log transforms are common for heavy-tailed variables (income, rent) because they reduce scale and make multiplicative differences more linear.\n\n"
                "### What you should see\n"
                "- the DataFrame is indexed by `('fips','year')`.\n"
                "- `log_income` and `log_rent` are finite (no -inf/inf).\n"
                "- summary stats look plausible (rates roughly in [0,1]).\n\n"
                "### Interpretation prompts\n"
                "- Why might `log_income` be easier to interpret than raw income?\n"
                "- What does a 0.01 change in `poverty_rate` represent?\n\n"
                "### Goal\n"
                "Load the county-year panel and build a small modeling table.\n"
            ),
            md("### Your Turn: Load panel (processed or sample)"),
            code(
                "import numpy as np\n"
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'census_county_panel.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'census_county_panel_sample.csv')\n"
                "\n"
                "# TODO: Ensure fips/year exist and build a MultiIndex\n"
                "df['fips'] = df['fips'].astype(str)\n"
                "df['year'] = df['year'].astype(int)\n"
                "df = df.set_index(['fips', 'year'], drop=False).sort_index()\n"
                "\n"
                "# Starter transforms\n"
                "df['log_income'] = np.log(df['B19013_001E'].astype(float))\n"
                "df['log_rent'] = np.log(df['B25064_001E'].astype(float))\n"
                "\n"
                "df[['poverty_rate', 'unemployment_rate', 'log_income', 'log_rent']].describe()\n"
            ),
            md(
                f"<a id=\"{slugify('Pooled OLS baseline')}\"></a>\n"
                "## Pooled OLS baseline\n\n"
                "### Background\n"
                "Pooled OLS treats each row as an independent observation and ignores that rows come from the same county over time.\n"
                "It is a useful baseline, but it can be misleading when counties differ in unobserved, time-invariant ways (baseline poverty, institutions, geography).\n\n"
                "### What you should see\n"
                "- a `statsmodels` summary table.\n"
                "- coefficients with HC3 robust SE.\n\n"
                "### Interpretation prompts\n"
                "- Interpret the sign and units of one coefficient in 2–4 sentences.\n"
                "- List one plausible omitted variable that differs across counties and could confound this pooled relationship.\n\n"
                "### Goal\n"
                "Fit a pooled model that ignores FE.\n"
            ),
            md("### Your Turn: Fit pooled OLS"),
            code(
                "import statsmodels.api as sm\n\n"
                "y_col = 'poverty_rate'\n"
                "x_cols = ['log_income', 'unemployment_rate']\n"
                "\n"
                "tmp = df[[y_col] + x_cols].dropna().copy()\n"
                "y = tmp[y_col].astype(float)\n"
                "X = sm.add_constant(tmp[x_cols].astype(float), has_constant='add')\n"
                "\n"
                "# TODO: Fit and print a summary (HC3 as a baseline)\n"
                "res_pool = sm.OLS(y, X).fit(cov_type='HC3')\n"
                "print(res_pool.summary())\n"
            ),
            md(
                f"<a id=\"{slugify('Two-way fixed effects')}\"></a>\n"
                "## Two-way fixed effects\n\n"
                "### Background\n"
                "Two-way fixed effects (TWFE) compares counties to themselves over time (county FE) while removing year-wide shocks (year FE).\n"
                "This can reduce bias from time-invariant county differences.\n\n"
                "### What you should see\n"
                "- a `PanelOLS` summary.\n"
                "- coefficients that may differ from pooled OLS (because identification uses within-county changes).\n\n"
                "### Interpretation prompts\n"
                "- Compare pooled vs TWFE: did the coefficient move? What story could explain the change?\n"
                "- What variation identifies $\\beta$ in TWFE (within county, across time)?\n\n"
                "### Goal\n"
                "Estimate a TWFE model:\n"
                "- county FE (entity)\n"
                "- year FE (time)\n"
            ),
            md("### Your Turn: Fit TWFE with PanelOLS"),
            code(
                "from src.causal import fit_twfe_panel_ols\n\n"
                "# TODO: Fit TWFE (robust SE)\n"
                "res_twfe = fit_twfe_panel_ols(\n"
                "    df,\n"
                "    y_col=y_col,\n"
                "    x_cols=x_cols,\n"
                "    entity_effects=True,\n"
                "    time_effects=True,\n"
                ")\n"
                "print(res_twfe.summary)\n"
            ),
            md(
                f"<a id=\"{slugify('Clustered standard errors')}\"></a>\n"
                "## Clustered standard errors\n\n"
                "### Background\n"
                "Even with TWFE, inference can be too optimistic if errors are correlated within groups.\n"
                "A common choice here is clustering by state because counties in the same state share policies and shocks.\n\n"
                "### What you should see\n"
                "- a table comparing robust vs clustered standard errors.\n"
                "- clustered SE are often larger (not guaranteed, but common).\n\n"
                "### Interpretation prompts\n"
                "- Which SE would you report for a state-level policy story and why?\n"
                "- How many clusters do you have (unique states), and why does that matter?\n\n"
                "### Goal\n"
                "Re-fit TWFE with clustered SE.\n\n"
                "Typical clustering choice here:\n"
                "- by state (shared shocks/policies)\n"
            ),
            md("### Your Turn: Cluster by state and compare SE"),
            code(
                "import pandas as pd\n"
                "from src.causal import fit_twfe_panel_ols\n\n"
                "# TODO: Compare robust vs clustered SE\n"
                "res_cluster = fit_twfe_panel_ols(\n"
                "    df,\n"
                "    y_col=y_col,\n"
                "    x_cols=x_cols,\n"
                "    entity_effects=True,\n"
                "    time_effects=True,\n"
                "    cluster_col='state',\n"
                ")\n"
                "\n"
                "pd.DataFrame({'robust_se': res_twfe.std_errors, 'cluster_se': res_cluster.std_errors})\n"
            ),
        ]

    if spec.path.endswith("02_difference_in_differences_event_study.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Practice DiD and event studies using:\n"
                "- a real county-year outcome (poverty rate)\n"
                "- a **synthetic**, deterministic adoption schedule by state\n"
                "- a **semi-synthetic** outcome with a known injected treatment effect\n\n"
                "This is a method exercise, not a real policy evaluation.\n"
            ),
            md(primer("linearmodels_panel_iv")),
            md(
                f"<a id=\"{slugify('Synthetic adoption + treatment')}\"></a>\n"
                "## Synthetic adoption + treatment\n\n"
                "### Background\n"
                "A real DiD design needs a real policy change and careful context.\n"
                "Here we use a **synthetic adoption schedule** so you can focus on mechanics:\n"
                "- how to build treatment indicators,\n"
                "- how to think about identification (parallel trends),\n"
                "- and how to diagnose pre-trends.\n\n"
                "We also create a **semi-synthetic outcome** by injecting a known post-treatment effect into a real outcome.\n"
                "That gives you a ground truth target for checking the estimator.\n\n"
                "### What you should see\n"
                "- `treated` equals 1 only for treated states in post-adoption years.\n"
                "- `poverty_rate_semi` differs from `poverty_rate_real` by about `true_effect` when treated.\n\n"
                "### Interpretation prompts\n"
                "- In one sentence, define the causal question this notebook is pretending to answer.\n"
                "- What assumption would be needed for the TWFE DiD coefficient to be causal on the real outcome?\n\n"
                "### Goal\n"
                "Define a deterministic adoption year by state and build:\n"
                "- `treated_it`\n"
                "- `poverty_rate_semi` (known post-treatment effect)\n"
            ),
            md("### Your Turn: Load panel and create synthetic adoption"),
            code(
                "import numpy as np\n"
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'census_county_panel.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'census_county_panel_sample.csv')\n"
                "\n"
                "df['fips'] = df['fips'].astype(str)\n"
                "df['year'] = df['year'].astype(int)\n"
                "df['state'] = df['state'].astype(str).str.zfill(2)\n"
                "\n"
                "states = sorted(df['state'].unique())\n"
                "# Deterministic adoption schedule (edit if you want):\n"
                "adopt = {states[0]: 2018, states[1]: 2020}  # remaining states are never-treated\n"
                "\n"
                "df['adopt_year'] = df['state'].map(adopt)\n"
                "df['ever_treated'] = df['adopt_year'].notna().astype(int)\n"
                "df['post'] = ((df['year'] >= df['adopt_year']).fillna(False)).astype(int)\n"
                "df['treated'] = df['ever_treated'] * df['post']\n"
                "\n"
                "true_effect = -0.02\n"
                "df['poverty_rate_real'] = df['poverty_rate'].astype(float)\n"
                "df['poverty_rate_semi'] = (df['poverty_rate_real'] + true_effect * df['treated']).clip(0, 1)\n"
                "\n"
                "df[['state', 'year', 'treated', 'poverty_rate_real', 'poverty_rate_semi']].head()\n"
            ),
            md(
                f"<a id=\"{slugify('TWFE DiD')}\"></a>\n"
                "## TWFE DiD\n\n"
                "### Background\n"
                "The simplest multi-period DiD estimator is a TWFE regression with a treatment indicator.\n"
                "Under parallel trends (and related assumptions), the coefficient on `treated` can be interpreted as an average treatment effect.\n\n"
                "### What you should see\n"
                "- On the semi-synthetic outcome, the estimated `treated` coefficient should be in the neighborhood of `true_effect`.\n"
                "- Standard errors should be clustered by state (treatment assignment/shocks).\n\n"
                "### Interpretation prompts\n"
                "- Compare the estimate to `true_effect`. Is it close? If not, why might it differ (small sample, noise, design)?\n"
                "- Write the parallel trends assumption in words for this setting.\n\n"
                "### Goal\n"
                "Estimate the effect of treatment with TWFE DiD:\n"
                "- county FE\n"
                "- year FE\n"
                "- clustered SE by state (common)\n"
            ),
            md("### Your Turn: Fit TWFE DiD"),
            code(
                "from src.causal import fit_twfe_panel_ols\n\n"
                "# Panel index\n"
                "df = df.set_index(['fips', 'year'], drop=False).sort_index()\n"
                "\n"
                "# TODO: Fit DiD on semi-synthetic outcome\n"
                "res_did = fit_twfe_panel_ols(\n"
                "    df,\n"
                "    y_col='poverty_rate_semi',\n"
                "    x_cols=['treated'],\n"
                "    entity_effects=True,\n"
                "    time_effects=True,\n"
                "    cluster_col='state',\n"
                ")\n"
                "\n"
                "res_did.params\n"
            ),
            md(
                f"<a id=\"{slugify('Event study (leads/lags)')}\"></a>\n"
                "## Event study (leads/lags)\n\n"
                "### Background\n"
                "An event study replaces a single post indicator with a set of lead/lag indicators.\n"
                "This lets you:\n"
                "- visualize dynamics after adoption, and\n"
                "- test for pre-trends using lead coefficients.\n\n"
                "### What you should see\n"
                "- lead coefficients (k<0) near 0 on the semi-synthetic outcome.\n"
                "- post coefficients (k>=0) around the injected effect.\n\n"
                "### Interpretation prompts\n"
                "- Which lead coefficients would worry you most, and why?\n"
                "- Explain what the base period means (why one event-time dummy is omitted).\n\n"
                "### Goal\n"
                "Estimate dynamic effects around adoption and inspect pre-trends.\n"
            ),
            md("### Your Turn: Build leads/lags and fit"),
            code(
                "import pandas as pd\n"
                "import matplotlib.pyplot as plt\n"
                "\n"
                "df_es = df.reset_index(drop=True).copy()\n"
                "df_es['event_time'] = df_es['year'] - df_es['adopt_year']\n"
                "\n"
                "window = list(range(-3, 4))\n"
                "base = -1\n"
                "event_cols = []\n"
                "for k in window:\n"
                "    if k == base:\n"
                "        continue\n"
                "    col = f'event_{k}'\n"
                "    df_es[col] = ((df_es['ever_treated'] == 1) & (df_es['event_time'] == k)).astype(int)\n"
                "    event_cols.append(col)\n"
                "\n"
                "df_es = df_es.set_index(['fips', 'year'], drop=False).sort_index()\n"
                "\n"
                "res_es = fit_twfe_panel_ols(\n"
                "    df_es,\n"
                "    y_col='poverty_rate_semi',\n"
                "    x_cols=event_cols,\n"
                "    entity_effects=True,\n"
                "    time_effects=True,\n"
                "    cluster_col='state',\n"
                ")\n"
                "\n"
                "coefs = res_es.params.filter(like='event_')\n"
                "ses = res_es.std_errors.filter(like='event_')\n"
                "out = coefs.to_frame('coef').join(ses.to_frame('se'))\n"
                "out['k'] = out.index.str.replace('event_', '').astype(int)\n"
                "out = out.sort_values('k')\n"
                "\n"
                "# TODO: Plot coefficient path with 95% CI\n"
                "plt.errorbar(out['k'], out['coef'], yerr=1.96*out['se'], fmt='o-')\n"
                "plt.axhline(0, color='gray', linestyle='--')\n"
                "plt.axvline(base, color='gray', linestyle=':')\n"
                "plt.xlabel('Event time (years relative to adoption)')\n"
                "plt.ylabel('Effect')\n"
                "plt.title('Event study (semi-synthetic)')\n"
                "plt.show()\n"
            ),
            md(
                f"<a id=\"{slugify('Diagnostics: pre-trends + placebo')}\"></a>\n"
                "## Diagnostics: pre-trends + placebo\n\n"
                "### Background\n"
                "DiD is only as credible as its diagnostics.\n"
                "In real research, this is where most of the work lives:\n"
                "- are treated and control trending similarly before treatment?\n"
                "- are results robust to reasonable specification changes?\n"
                "- do placebo tests behave as expected?\n\n"
                "### What you should see\n"
                "- a short diagnostic result (table/plot) and a written interpretation.\n\n"
                "### Interpretation prompts\n"
                "- If the placebo finds a large effect, what does that suggest about the design?\n"
                "- Why is the real outcome analysis explicitly **not** a real policy evaluation here?\n\n"
                "### Goal\n"
                "Run at least one falsification / diagnostic.\n\n"
                "Suggestions:\n"
                "- Pre-trends: are lead coefficients near 0?\n"
                "- Placebo: shift adoption years earlier for treated states.\n"
                "- Re-run on the real outcome (`poverty_rate_real`) and reflect on why it is not causal.\n"
            ),
            md("### Your Turn: One diagnostic"),
            code(
                "# TODO: Implement one diagnostic and summarize what you found.\n"
                "...\n"
            ),
        ]

    if spec.path.endswith("03_instrumental_variables_2sls.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Practice IV/2SLS by simulating a classic endogeneity problem.\n\n"
                "We do this synthetically so you can see the bias and how IV can fix it under assumptions.\n"
            ),
            md(primer("linearmodels_panel_iv")),
            md(
                f"<a id=\"{slugify('Simulate endogeneity')}\"></a>\n"
                "## Simulate endogeneity\n\n"
                "### Background\n"
                "Endogeneity means your regressor $x$ is correlated with the error term.\n"
                "That breaks the core OLS condition $E[u\\mid X]=0$ and typically biases OLS.\n\n"
                "We simulate endogeneity by constructing a hidden confounder $u$ that affects both $x$ and $y$.\n"
                "We then construct an instrument $z$ that shifts $x$ but (by design) does not directly shift $y$.\n\n"
                "### What you should see\n"
                "- `x` is correlated with the confounder-driven error component.\n"
                "- `z` is correlated with `x` (relevance).\n"
                "- `z` is not directly in the structural equation for `y` (exclusion in this synthetic setup).\n\n"
                "### Interpretation prompts\n"
                "- In one sentence, explain why OLS is biased here.\n"
                "- Write the relevance and exclusion conditions in words for this simulation.\n\n"
                "### Goal\n"
                "Create data where:\n"
                "- x is correlated with the error term (endogenous)\n"
                "- z shifts x but not y directly (instrument)\n"
            ),
            md("### Your Turn: Simulate (y, x, z)"),
            code(
                "import numpy as np\n"
                "import pandas as pd\n\n"
                "rng = np.random.default_rng(0)\n"
                "n = 2000\n"
                "\n"
                "# Instrument\n"
                "z = rng.normal(size=n)\n"
                "\n"
                "# Hidden confounder\n"
                "u = rng.normal(size=n)\n"
                "\n"
                "# Endogenous regressor: depends on z and u\n"
                "x = 0.8*z + 0.8*u + rng.normal(size=n)\n"
                "\n"
                "# Error term correlated with u\n"
                "eps = 0.8*u + rng.normal(size=n)\n"
                "\n"
                "beta_true = 1.5\n"
                "y = beta_true * x + eps\n"
                "\n"
                "df = pd.DataFrame({'y': y, 'x': x, 'z': z})\n"
                "df.head()\n"
            ),
            md(
                f"<a id=\"{slugify('OLS vs 2SLS')}\"></a>\n"
                "## OLS vs 2SLS\n\n"
                "### Background\n"
                "OLS uses all variation in $x$, including the endogenous part correlated with the error.\n"
                "2SLS replaces $x$ with the part predicted by $z$ (instrumented variation).\n\n"
                "### What you should see\n"
                "- OLS estimate differs from `beta_true` (bias).\n"
                "- IV/2SLS estimate is closer to `beta_true` (in this synthetic world).\n\n"
                "### Interpretation prompts\n"
                "- Which direction is the OLS bias and why (link it to how you constructed the confounder)?\n"
                "- Why does IV move the estimate toward the truth in this setup?\n\n"
                "### Goal\n"
                "Compare naive OLS (biased) to IV/2SLS.\n"
            ),
            md("### Your Turn: Fit OLS and 2SLS"),
            code(
                "import statsmodels.api as sm\n"
                "from src.causal import fit_iv_2sls\n\n"
                "# OLS\n"
                "ols = sm.OLS(df['y'], sm.add_constant(df[['x']], has_constant='add')).fit()\n"
                "print('OLS beta:', float(ols.params['x']))\n"
                "\n"
                "# 2SLS\n"
                "iv = fit_iv_2sls(df, y_col='y', x_endog='x', x_exog=[], z_cols=['z'])\n"
                "print('IV beta :', float(iv.params['x']))\n"
                "\n"
                "iv.summary\n"
            ),
            md(
                f"<a id=\"{slugify('First-stage + weak IV checks')}\"></a>\n"
                "## First-stage + weak IV checks\n\n"
                "### Background\n"
                "A valid instrument must be relevant.\n"
                "If $z$ barely predicts $x$, 2SLS can be unstable and misleading (weak instruments).\n\n"
                "### What you should see\n"
                "- a first-stage relationship where `z` helps explain `x`.\n"
                "- a discussion of instrument strength (even informally).\n\n"
                "### Interpretation prompts\n"
                "- What would happen to 2SLS if `z` were only weakly related to `x`?\n"
                "- Which parts of IV validity are testable from the data, and which are not?\n\n"
                "### Goal\n"
                "Inspect the first stage and discuss instrument strength.\n"
            ),
            md("### Your Turn: Inspect first stage"),
            code(
                "# TODO: Explore first-stage outputs.\n"
                "# Hint: `iv.first_stage` is usually informative.\n"
                "iv.first_stage\n"
            ),
            md(
                f"<a id=\"{slugify('Interpretation + limitations')}\"></a>\n"
                "## Interpretation + limitations\n\n"
                "Write 5-8 sentences on:\n"
                "- relevance and exclusion in this synthetic setup\n"
                "- what would break IV in real data\n"
                "- why IV identifies a local effect when effects are heterogeneous (LATE intuition)\n"
            ),
        ]

    # Time-series econometrics notebooks
    if spec.path.endswith("00_stationarity_unit_roots.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Learn the stationarity toolkit that prevents common macro mistakes:\n"
                "- spurious regression\n"
                "- over-trusting p-values on trending series\n"
                "- misinterpreting dynamics\n"
            ),
            md(primer("pandas_time_series")),
            md(primer("statsmodels_tsa_var")),
            md(
                f"<a id=\"{slugify('Load macro series')}\"></a>\n"
                "## Load macro series\n\n"
                "### Background\n"
                "Stationarity analysis is only meaningful if your time index is correct.\n"
                "So the first task is: load a clean monthly panel with a proper `DatetimeIndex`.\n\n"
                "### What you should see\n"
                "- a DataFrame indexed by dates (monthly).\n"
                "- key macro columns like CPI, unemployment, production.\n\n"
                "### Interpretation prompts\n"
                "- Which of these series looks trending in levels?\n"
                "- Which series might be closer to stationary already (in levels)?\n\n"
                "### Goal\n"
                "Load the macro monthly panel.\n"
            ),
            md("### Your Turn: Load panel_monthly.csv (or sample)"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'panel_monthly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "df = df.dropna().copy()\n"
                "df.head()\n"
            ),
            md(
                f"<a id=\"{slugify('Transformations')}\"></a>\n"
                "## Transformations\n\n"
                "### Background\n"
                "Many macro series are nonstationary in levels.\n"
                "Common fixes are:\n"
                "- differences (change),\n"
                "- percent changes (growth rates),\n"
                "- log-differences (approx growth rates for positive series).\n\n"
                "### What you should see\n"
                "- transformed columns with fewer trends.\n"
                "- a smaller DataFrame after `dropna()` (because differencing creates missing first row).\n\n"
                "### Interpretation prompts\n"
                "- Why might log-differences be preferred for production indexes?\n"
                "- What information do you lose when you difference?\n\n"
                "### Goal\n"
                "Create stationary-ish transformations (diff, pct change, log diff).\n"
            ),
            md("### Your Turn: Differences and growth rates"),
            code(
                "import numpy as np\n\n"
                "# TODO: Pick a few series and create transformations\n"
                "tmp = df[['CPIAUCSL', 'UNRATE', 'INDPRO']].astype(float).copy()\n"
                "tmp['dCPI'] = tmp['CPIAUCSL'].diff()\n"
                "tmp['dUNRATE'] = tmp['UNRATE'].diff()\n"
                "\n"
                "# log-diff for industrial production (example)\n"
                "x = tmp['INDPRO'].where(tmp['INDPRO'] > 0)\n"
                "tmp['dlog_INDPRO'] = np.log(x).diff()\n"
                "\n"
                "tmp = tmp.dropna()\n"
                "tmp.head()\n"
            ),
            md(
                f"<a id=\"{slugify('ADF/KPSS tests')}\"></a>\n"
                "## ADF/KPSS tests\n\n"
                "### Background\n"
                "ADF and KPSS are complementary diagnostics:\n"
                "- ADF null: unit root (nonstationary)\n"
                "- KPSS null: stationary\n\n"
                "No single p-value is a proof. Use tests alongside plots and economic context.\n\n"
                "### What you should see\n"
                "- different p-values for level vs differenced series.\n"
                "- clearer stationarity evidence after transformation.\n\n"
                "### Interpretation prompts\n"
                "- In words: what does a small ADF p-value suggest?\n"
                "- In words: what does a small KPSS p-value suggest?\n\n"
                "### Goal\n"
                "Run stationarity diagnostics on levels vs transformed series.\n"
            ),
            md("### Your Turn: ADF and KPSS"),
            code(
                "from statsmodels.tsa.stattools import adfuller, kpss\n\n"
                "# TODO: Choose one series and compare levels vs diff\n"
                "x = df['CPIAUCSL'].astype(float).dropna()\n"
                "dx = x.diff().dropna()\n"
                "\n"
                "out = {\n"
                "    'adf_p_level': adfuller(x)[1],\n"
                "    'adf_p_diff': adfuller(dx)[1],\n"
                "    'kpss_p_level': kpss(x, regression='c', nlags='auto')[1],\n"
                "    'kpss_p_diff': kpss(dx, regression='c', nlags='auto')[1],\n"
                "}\n"
                "out\n"
            ),
            md(
                f"<a id=\"{slugify('Spurious regression demo')}\"></a>\n"
                "## Spurious regression demo\n\n"
                "### Background\n"
                "A classic macro trap is regressing one trending series on another.\n"
                "You can get a high $R^2$ and significant coefficients even when the relationship is meaningless.\n\n"
                "### What you should see\n"
                "- the levels regression often has a higher $R^2$ than the differences regression.\n"
                "- this demonstrates why stationarity checks are a prerequisite for inference.\n\n"
                "### Interpretation prompts\n"
                "- Why can $R^2$ be high in a spurious regression?\n"
                "- What would you do next if you *needed* a meaningful long-run relationship? (hint: cointegration)\n\n"
                "### Goal\n"
                "Show how levels-on-levels regressions can look good for the wrong reasons.\n"
            ),
            md("### Your Turn: Levels vs differences"),
            code(
                "import statsmodels.api as sm\n\n"
                "tmp2 = df[['CPIAUCSL', 'INDPRO']].astype(float).dropna()\n"
                "\n"
                "# Levels regression\n"
                "res_lvl = sm.OLS(tmp2['CPIAUCSL'], sm.add_constant(tmp2[['INDPRO']], has_constant='add')).fit()\n"
                "\n"
                "# Differences regression\n"
                "d = tmp2.diff().dropna()\n"
                "res_diff = sm.OLS(d['CPIAUCSL'], sm.add_constant(d[['INDPRO']], has_constant='add')).fit()\n"
                "\n"
                "print('R2 levels:', res_lvl.rsquared)\n"
                "print('R2 diffs :', res_diff.rsquared)\n"
            ),
        ]

    if spec.path.endswith("01_cointegration_error_correction.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Learn cointegration and error correction models (ECM):\n"
                "- long-run equilibrium relationship\n"
                "- short-run dynamics that correct deviations\n"
            ),
            md(primer("statsmodels_tsa_var")),
            md(
                f"<a id=\"{slugify('Construct cointegrated pair')}\"></a>\n"
                "## Construct cointegrated pair\n\n"
                "### Background\n"
                "Cointegration is the key exception to the “levels regressions are spurious” warning.\n"
                "Two series can be nonstationary individually but have a stable long-run relationship.\n\n"
                "### What you should see\n"
                "- `x` and `y` trend over time.\n"
                "- `y - x` should look roughly stationary (because we simulated cointegration).\n\n"
                "### Interpretation prompts\n"
                "- In one sentence: what does it mean for two series to be cointegrated?\n"
                "- Why do we simulate first before applying to real macro series?\n\n"
                "### Goal\n"
                "Construct a pair of series that are individually nonstationary but cointegrated.\n"
            ),
            md("### Your Turn: Simulate a cointegrated pair"),
            code(
                "import numpy as np\n"
                "import pandas as pd\n\n"
                "rng = np.random.default_rng(0)\n"
                "n = 240\n"
                "idx = pd.date_range('2000-01-31', periods=n, freq='ME')\n"
                "\n"
                "x = rng.normal(size=n).cumsum()\n"
                "y = 1.0 * x + rng.normal(scale=0.5, size=n)\n"
                "\n"
                "df = pd.DataFrame({'x': x, 'y': y}, index=idx)\n"
                "df.head()\n"
            ),
            md(
                f"<a id=\"{slugify('Engle-Granger test')}\"></a>\n"
                "## Engle-Granger test\n\n"
                "### Background\n"
                "The Engle–Granger approach:\n"
                "1) regress $y$ on $x$ in levels,\n"
                "2) test whether the residual is stationary.\n\n"
                "### What you should see\n"
                "- a cointegration test p-value (often small in this simulated example).\n\n"
                "### Interpretation prompts\n"
                "- What is the null hypothesis in the cointegration test?\n"
                "- If the p-value were large, what would that suggest?\n\n"
                "### Goal\n"
                "Run a cointegration test and interpret the p-value carefully.\n"
            ),
            md("### Your Turn: Cointegration test"),
            code(
                "from statsmodels.tsa.stattools import coint\n\n"
                "t_stat, p_val, _ = coint(df['y'], df['x'])\n"
                "{'t': t_stat, 'p': p_val}\n"
            ),
            md(
                f"<a id=\"{slugify('Error correction model')}\"></a>\n"
                "## Error correction model\n\n"
                "### Background\n"
                "An ECM links:\n"
                "- short-run changes ($\\Delta y_t$)\n"
                "- to long-run disequilibrium (lagged residual from the levels relationship).\n\n"
                "### What you should see\n"
                "- an estimated coefficient on `u_lag1` (often negative in a stable cointegrated system).\n"
                "- interpretation as “speed of adjustment.”\n\n"
                "### Interpretation prompts\n"
                "- What does the sign of the error-correction coefficient mean?\n"
                "- Why do we use the lagged residual rather than the current residual?\n\n"
                "### Goal\n"
                "Fit an ECM:\n"
                "- short-run changes depend on long-run disequilibrium (lagged residual)\n"
            ),
            md("### Your Turn: Fit ECM"),
            code(
                "import statsmodels.api as sm\n\n"
                "# Long-run regression\n"
                "lr = sm.OLS(df['y'], sm.add_constant(df[['x']], has_constant='add')).fit()\n"
                "df['u'] = lr.resid\n"
                "\n"
                "# ECM regression\n"
                "ecm = pd.DataFrame({\n"
                "    'dy': df['y'].diff(),\n"
                "    'dx': df['x'].diff(),\n"
                "    'u_lag1': df['u'].shift(1),\n"
                "}).dropna()\n"
                "\n"
                "res = sm.OLS(ecm['dy'], sm.add_constant(ecm[['dx', 'u_lag1']], has_constant='add')).fit()\n"
                "res.params\n"
            ),
            md(
                f"<a id=\"{slugify('Interpretation')}\"></a>\n"
                "## Interpretation\n\n"
                "Write 5-8 sentences:\n"
                "- What does the error-correction coefficient mean?\n"
                "- What would you expect if there were no cointegration?\n"
            ),
        ]

    if spec.path.endswith("02_var_impulse_responses.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Fit a VAR on transformed macro series and interpret:\n"
                "- lag selection\n"
                "- Granger causality\n"
                "- impulse response functions (IRFs)\n"
            ),
            md(primer("statsmodels_tsa_var")),
            md(
                f"<a id=\"{slugify('Build stationary dataset')}\"></a>\n"
                "## Build stationary dataset\n\n"
                "### Background\n"
                "VARs generally require stable (stationary-ish) inputs.\n"
                "A common first pass is to difference level series.\n\n"
                "### What you should see\n"
                "- a DataFrame of transformed series (no missing values).\n"
                "- columns are numeric floats.\n\n"
                "### Interpretation prompts\n"
                "- What does differencing do to trends and to noise?\n"
                "- Which series might require log-differencing rather than differencing?\n\n"
                "### Goal\n"
                "Build a small stationary-ish dataset to fit a VAR.\n"
            ),
            md("### Your Turn: Load and transform"),
            code(
                "import pandas as pd\n\n"
                "panel = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True).dropna()\n"
                "\n"
                "# TODO: Choose a few columns and difference them\n"
                "df = panel[['UNRATE', 'FEDFUNDS', 'INDPRO']].astype(float).diff().dropna()\n"
                "df.head()\n"
            ),
            md(
                f"<a id=\"{slugify('Fit VAR + choose lags')}\"></a>\n"
                "## Fit VAR + choose lags\n\n"
                "### Background\n"
                "VAR lag length is a bias–variance decision:\n"
                "- too few lags → leftover autocorrelation and misspecification,\n"
                "- too many lags → unstable estimates and low degrees of freedom.\n\n"
                "### What you should see\n"
                "- a chosen lag order (`res.k_ar`).\n"
                "- a model summary with coefficients for lagged terms.\n\n"
                "### Interpretation prompts\n"
                "- Why might AIC choose more lags than BIC?\n"
                "- What diagnostic would you check if you suspect too few lags?\n\n"
                "### Goal\n"
                "Fit a VAR and choose lags using an information criterion.\n"
            ),
            md("### Your Turn: Fit VAR"),
            code(
                "from statsmodels.tsa.api import VAR\n\n"
                "# TODO: Fit and inspect chosen lag order\n"
                "res = VAR(df).fit(maxlags=8, ic='aic')\n"
                "res.k_ar\n"
            ),
            md(
                f"<a id=\"{slugify('Granger causality')}\"></a>\n"
                "## Granger causality\n\n"
                "### Background\n"
                "Granger causality asks a forecasting question:\n"
                "- do lagged values of $x$ help predict $y$ beyond lagged $y$?\n\n"
                "It is not structural causality.\n\n"
                "### What you should see\n"
                "- a test output summary for one direction (e.g., FEDFUNDS → UNRATE).\n\n"
                "### Interpretation prompts\n"
                "- Rewrite the Granger test result as a forecasting statement (not a causal one).\n"
                "- Why can a third variable create apparent Granger relationships?\n\n"
                "### Goal\n"
                "Run at least one Granger causality test.\n\n"
                "Reminder: this is predictive causality, not structural causality.\n"
            ),
            md("### Your Turn: Test causality"),
            code(
                "# Example: do lagged FEDFUNDS help predict UNRATE?\n"
                "res.test_causality('UNRATE', ['FEDFUNDS']).summary()\n"
            ),
            md(
                f"<a id=\"{slugify('IRFs + forecasting')}\"></a>\n"
                "## IRFs + forecasting\n\n"
                "### Background\n"
                "Impulse responses trace how a one-time shock propagates through the VAR dynamics.\n"
                "Orthogonalized IRFs (Cholesky) impose an identification choice via variable ordering.\n\n"
                "### What you should see\n"
                "- an IRF plot over the chosen horizon.\n"
                "- qualitative responses that decay if the VAR is stable.\n\n"
                "### Interpretation prompts\n"
                "- How does changing the variable ordering change the meaning of the shock?\n"
                "- Which IRF responses would you view as economically plausible vs suspicious?\n\n"
                "### Goal\n"
                "Compute and plot impulse responses.\n\n"
                "Caution:\n"
                "- orthogonalized IRFs depend on variable ordering.\n"
            ),
            md("### Your Turn: IRFs"),
            code(
                "irf = res.irf(12)\n"
                "irf.plot(orth=True)\n"
            ),
        ]

    # Classification notebooks
    if spec.path.endswith("00_recession_classifier_baselines.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Establish baselines for predicting **next-quarter technical recession**.\n\n"
                "Baselines matter because:\n"
                "- recession is rare (class imbalance)\n"
                "- a model that beats chance may still be useless\n"
                "- you need a reference point before tuning anything\n"
            ),
            md(primer("sklearn_pipelines")),
            md(
                f"<a id=\"{slugify('Load data')}\"></a>\n"
                "## Load data\n\n"
                "### Goal\n"
                "Load the macro quarterly modeling table and select:\n"
                "- `y = target_recession_next_q`\n"
                "- a minimal set of features\n"
            ),
            md("### Your Turn (1): Load macro_quarterly.csv (or sample)"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "df.head()\n"
            ),
            md("### Your Turn (2): Define target and a starter feature set"),
            code(
                "# Target (0/1)\n"
                "y_col = 'target_recession_next_q'\n"
                "\n"
                "# TODO: Pick a small feature set.\n"
                "# Tip: use lagged predictors.\n"
                "x_cols = [\n"
                "    'T10Y2Y_lag1',\n"
                "    'UNRATE_lag1',\n"
                "    'FEDFUNDS_lag1',\n"
                "]\n"
                "\n"
                "df_m = df[[y_col] + x_cols + ['recession']].dropna().copy()\n"
                "df_m[y_col].value_counts(dropna=False)\n"
            ),
            md("### Checkpoint (class imbalance awareness)"),
            code(
                "# TODO: Compute the base rate of recession in the target.\n"
                "base_rate = df_m[y_col].mean()\n"
                "base_rate\n"
            ),
            md(
                f"<a id=\"{slugify('Define baselines')}\"></a>\n"
                "## Define baselines\n\n"
                "You will implement 3 baselines:\n"
                "1) **Majority class**: always predict 0\n"
                "2) **Persistence**: predict next recession equals current recession label\n"
                "3) **Simple rule**: yield spread negative => recession (choose threshold)\n"
            ),
            md("### Your Turn (1): Build baseline probability scores"),
            code(
                "import numpy as np\n\n"
                "y_true = df_m[y_col].astype(int).to_numpy()\n"
                "\n"
                "# Baseline 1: always 0 probability\n"
                "p_majority = np.zeros_like(y_true, dtype=float)\n"
                "\n"
                "# Baseline 2: persistence (use current recession label as probability)\n"
                "p_persist = df_m['recession'].astype(float).to_numpy()\n"
                "\n"
                "# Baseline 3: simple rule on yield spread\n"
                "# TODO: Choose a threshold (e.g., 0.0 means inverted curve)\n"
                "thr = 0.0\n"
                "p_rule = (df_m['T10Y2Y_lag1'].to_numpy() < thr).astype(float)\n"
                "\n"
                "p_majority[:5], p_persist[:5], p_rule[:5]\n"
            ),
            md(
                f"<a id=\"{slugify('Evaluate metrics')}\"></a>\n"
                "## Evaluate metrics\n\n"
                "### Goal\n"
                "Evaluate baselines with metrics that make sense for imbalanced classification:\n"
                "- ROC-AUC\n"
                "- PR-AUC\n"
                "- Brier score\n"
                "- precision/recall at a chosen threshold\n"
            ),
            md("### Your Turn (1): Evaluate metrics"),
            code(
                "from src.evaluation import classification_metrics\n\n"
                "metrics = {\n"
                "    'majority': classification_metrics(y_true, p_majority, threshold=0.5),\n"
                "    'persistence': classification_metrics(y_true, p_persist, threshold=0.5),\n"
                "    'rule': classification_metrics(y_true, p_rule, threshold=0.5),\n"
                "}\n"
                "\n"
                "metrics\n"
            ),
            md("### Your Turn (2): Time-aware evaluation preview (optional)"),
            code(
                "# Optional: do the same baseline evaluation on a time-based train/test split.\n"
                "# Why: baselines can look better in-sample than out-of-sample.\n"
                "...\n"
            ),
        ]

    if spec.path.endswith("01_logistic_recession_classifier.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Train a logistic regression classifier for next-quarter technical recession.\n\n"
                "You will learn:\n"
                "- how to do a time-based split\n"
                "- how to fit a probabilistic classifier (outputs probabilities)\n"
                "- how to interpret coefficients as log-odds / odds ratios\n"
                "- why threshold selection is a decision problem, not a default\n"
            ),
            md(primer("sklearn_pipelines")),
            md(
                f"<a id=\"{slugify('Train/test split')}\"></a>\n"
                "## Train/test split\n\n"
                "### Goal\n"
                "Split chronologically so the model trains on the past and is evaluated on the future.\n"
            ),
            md("### Your Turn (1): Load data and select columns"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "y_col = 'target_recession_next_q'\n"
                "x_cols = ['T10Y2Y_lag1', 'UNRATE_lag1', 'FEDFUNDS_lag1', 'INDPRO_lag1']\n"
                "\n"
                "df_m = df[[y_col] + x_cols].dropna().copy()\n"
                "df_m.tail()\n"
            ),
            md("### Your Turn (2): Create a time split"),
            code(
                "from src.evaluation import time_train_test_split_index\n\n"
                "split = time_train_test_split_index(len(df_m), test_size=0.2)\n"
                "train = df_m.iloc[split.train_slice]\n"
                "test = df_m.iloc[split.test_slice]\n"
                "\n"
                "X_train = train[x_cols]\n"
                "y_train = train[y_col].astype(int)\n"
                "X_test = test[x_cols]\n"
                "y_test = test[y_col].astype(int)\n"
                "\n"
                "train.index.max(), test.index.min()\n"
            ),
            md(
                f"<a id=\"{slugify('Fit logistic')}\"></a>\n"
                "## Fit logistic\n\n"
                "### Goal\n"
                "Fit a logistic regression model inside a Pipeline (to avoid preprocessing leakage).\n"
            ),
            md("### Your Turn (1): Fit the pipeline"),
            code(
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import LogisticRegression\n\n"
                "clf = Pipeline([\n"
                "    ('scaler', StandardScaler()),\n"
                "    ('model', LogisticRegression(max_iter=5000)),\n"
                "])\n"
                "\n"
                "# TODO: Fit on training data\n"
                "clf.fit(X_train, y_train)\n"
                "\n"
                "# Predicted probabilities for class 1\n"
                "p_test = clf.predict_proba(X_test)[:, 1]\n"
                "p_test[:5]\n"
            ),
            md("### Your Turn (2): Evaluate metrics"),
            code(
                "from src.evaluation import classification_metrics\n\n"
                "classification_metrics(y_test.to_numpy(), p_test, threshold=0.5)\n"
            ),
            md("### Your Turn (3): Interpret coefficients (odds ratios)"),
            code(
                "import numpy as np\n"
                "import pandas as pd\n\n"
                "# Coefficients live in the underlying model\n"
                "coefs = clf.named_steps['model'].coef_[0]\n"
                "\n"
                "# TODO: Build a coefficient table.\n"
                "coef_df = pd.DataFrame({'feature': x_cols, 'coef': coefs})\n"
                "\n"
                "# Odds ratio for a 1-unit increase in standardized feature:\n"
                "# OR = exp(coef)\n"
                "coef_df['odds_ratio'] = np.exp(coef_df['coef'])\n"
                "coef_df.sort_values('coef')\n"
            ),
            md(
                f"<a id=\"{slugify('ROC/PR')}\"></a>\n"
                "## ROC/PR\n\n"
                "### Goal\n"
                "Plot ROC and precision-recall curves.\n\n"
                "Why both?\n"
                "- ROC can look optimistic under heavy class imbalance\n"
                "- PR focuses on the positive class (recessions)\n"
            ),
            md("### Your Turn: Plot ROC and PR curves"),
            code(
                "import matplotlib.pyplot as plt\n"
                "from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay\n\n"
                "# TODO: Create ROC and PR plots.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Threshold tuning')}\"></a>\n"
                "## Threshold tuning\n\n"
                "### Goal\n"
                "Compare metrics at different probability thresholds.\n\n"
                "A lower threshold catches more recessions (higher recall) but raises false positives.\n"
            ),
            md("### Your Turn: Evaluate multiple thresholds"),
            code(
                "from src.evaluation import classification_metrics\n\n"
                "for thr in [0.3, 0.5, 0.7]:\n"
                "    m = classification_metrics(y_test.to_numpy(), p_test, threshold=thr)\n"
                "    print(thr, m)\n"
            ),
        ]

    if spec.path.endswith("02_calibration_and_costs.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Go beyond accuracy: evaluate probability quality (calibration) and choose thresholds based on decision costs.\n\n"
                "Why this matters:\n"
                "- A recession probability model is only useful if probabilities mean something.\n"
                "- Threshold selection depends on the cost of false positives vs false negatives.\n"
            ),
            md(primer("sklearn_pipelines")),
            md(
                f"<a id=\"{slugify('Calibration')}\"></a>\n"
                "## Calibration\n\n"
                "### Goal\n"
                "Check whether predicted probabilities match observed frequencies.\n\n"
                "A calibrated model:\n"
                "- among events predicted at 30%, about 30% should occur (in the long run)\n"
            ),
            md("### Your Turn (1): Fit a base classifier and get probabilities"),
            code(
                "import pandas as pd\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from src.evaluation import time_train_test_split_index\n\n"
                "# Load data\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "y_col = 'target_recession_next_q'\n"
                "x_cols = ['T10Y2Y_lag1', 'UNRATE_lag1', 'FEDFUNDS_lag1', 'INDPRO_lag1']\n"
                "df_m = df[[y_col] + x_cols].dropna().copy()\n"
                "\n"
                "split = time_train_test_split_index(len(df_m), test_size=0.2)\n"
                "train = df_m.iloc[split.train_slice]\n"
                "test = df_m.iloc[split.test_slice]\n"
                "\n"
                "X_train = train[x_cols]\n"
                "y_train = train[y_col].astype(int)\n"
                "X_test = test[x_cols]\n"
                "y_test = test[y_col].astype(int)\n"
                "\n"
                "clf = Pipeline([\n"
                "    ('scaler', StandardScaler()),\n"
                "    ('model', LogisticRegression(max_iter=5000)),\n"
                "])\n"
                "clf.fit(X_train, y_train)\n"
                "p_test = clf.predict_proba(X_test)[:, 1]\n"
                "\n"
                "p_test[:5]\n"
            ),
            md("### Your Turn (2): Calibration curve and reliability plot"),
            code(
                "import matplotlib.pyplot as plt\n"
                "from sklearn.calibration import calibration_curve\n\n"
                "# TODO: Compute calibration curve\n"
                "# Hint: calibration_curve(y_true, y_prob, n_bins=...)\n"
                "prob_true, prob_pred = ...\n"
                "\n"
                "# TODO: Plot prob_pred vs prob_true with a y=x reference line\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Brier score')}\"></a>\n"
                "## Brier score\n\n"
                "### Goal\n"
                "Compute the Brier score (mean squared error of probabilities).\n\n"
                "Lower is better.\n"
            ),
            md("### Your Turn: Compute Brier score and interpret"),
            code(
                "from sklearn.metrics import brier_score_loss\n\n"
                "brier = brier_score_loss(y_test.to_numpy(), p_test)\n"
                "brier\n"
            ),
            md(
                f"<a id=\"{slugify('Decision costs')}\"></a>\n"
                "## Decision costs\n\n"
                "### Goal\n"
                "Choose a probability threshold using a simple cost model.\n\n"
                "Example framing:\n"
                "- False negative (miss a recession) might be more costly than a false positive.\n"
                "- You will encode that as a cost ratio and pick the threshold that minimizes expected cost.\n"
            ),
            md("### Your Turn (1): Define costs and compute expected cost across thresholds"),
            code(
                "import numpy as np\n\n"
                "# Cost of false positives vs false negatives\n"
                "cost_fp = 1.0\n"
                "cost_fn = 5.0\n"
                "\n"
                "thresholds = np.linspace(0.05, 0.95, 19)\n"
                "costs = []\n"
                "for thr in thresholds:\n"
                "    y_pred = (p_test >= thr).astype(int)\n"
                "    fp = ((y_pred == 1) & (y_test.to_numpy() == 0)).sum()\n"
                "    fn = ((y_pred == 0) & (y_test.to_numpy() == 1)).sum()\n"
                "    costs.append(cost_fp * fp + cost_fn * fn)\n"
                "\n"
                "best_thr = float(thresholds[int(np.argmin(costs))])\n"
                "best_thr\n"
            ),
            md("### Your Turn (2): Plot cost vs threshold"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Plot thresholds vs costs and mark the best_thr.\n"
                "...\n"
            ),
        ]

    if spec.path.endswith("03_tree_models_and_importance.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Compare a tree-based classifier to logistic regression and interpret feature importance.\n\n"
                "Trees can capture non-linearities and interactions, but are easier to overfit.\n"
            ),
            md(primer("sklearn_pipelines")),
            md(
                f"<a id=\"{slugify('Fit tree model')}\"></a>\n"
                "## Fit tree model\n\n"
                "### Goal\n"
                "Fit a tree-based classifier on the recession prediction task.\n"
            ),
            md("### Your Turn (1): Load data and split"),
            code(
                "import pandas as pd\n"
                "from src.evaluation import time_train_test_split_index\n\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "y_col = 'target_recession_next_q'\n"
                "x_cols = ['T10Y2Y_lag1', 'UNRATE_lag1', 'FEDFUNDS_lag1', 'INDPRO_lag1', 'RSAFS_lag1']\n"
                "df_m = df[[y_col] + x_cols].dropna().copy()\n"
                "\n"
                "split = time_train_test_split_index(len(df_m), test_size=0.2)\n"
                "train = df_m.iloc[split.train_slice]\n"
                "test = df_m.iloc[split.test_slice]\n"
                "\n"
                "X_train = train[x_cols]\n"
                "y_train = train[y_col].astype(int)\n"
                "X_test = test[x_cols]\n"
                "y_test = test[y_col].astype(int)\n"
            ),
            md("### Your Turn (2): Fit a RandomForestClassifier"),
            code(
                "from sklearn.ensemble import RandomForestClassifier\n\n"
                "# TODO: Fit a simple random forest.\n"
                "# Start small to avoid overfitting.\n"
                "rf = RandomForestClassifier(\n"
                "    n_estimators=300,\n"
                "    max_depth=3,\n"
                "    random_state=0,\n"
                ")\n"
                "\n"
                "rf.fit(X_train, y_train)\n"
                "p_test = rf.predict_proba(X_test)[:, 1]\n"
                "p_test[:5]\n"
            ),
            md(
                f"<a id=\"{slugify('Compare metrics')}\"></a>\n"
                "## Compare metrics\n\n"
                "### Goal\n"
                "Compare tree performance to a logistic baseline.\n"
            ),
            md("### Your Turn: Evaluate and compare"),
            code(
                "from src.evaluation import classification_metrics\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import LogisticRegression\n\n"
                "# Logistic baseline\n"
                "logit = Pipeline([\n"
                "    ('scaler', StandardScaler()),\n"
                "    ('model', LogisticRegression(max_iter=5000)),\n"
                "])\n"
                "logit.fit(X_train, y_train)\n"
                "p_logit = logit.predict_proba(X_test)[:, 1]\n"
                "\n"
                "m_rf = classification_metrics(y_test.to_numpy(), p_test, threshold=0.5)\n"
                "m_logit = classification_metrics(y_test.to_numpy(), p_logit, threshold=0.5)\n"
                "\n"
                "{'random_forest': m_rf, 'logistic': m_logit}\n"
            ),
            md(
                f"<a id=\"{slugify('Interpret importance')}\"></a>\n"
                "## Interpret importance\n\n"
                "### Goal\n"
                "Compare built-in feature importance to permutation importance.\n\n"
                "Why they can disagree:\n"
                "- built-in importance can be biased toward high-cardinality/noisy features\n"
                "- permutation importance measures impact on a chosen metric on a chosen dataset\n"
            ),
            md("### Your Turn (1): Built-in feature importances"),
            code(
                "import pandas as pd\n\n"
                "imp = pd.Series(rf.feature_importances_, index=x_cols).sort_values(ascending=False)\n"
                "imp\n"
            ),
            md("### Your Turn (2): Permutation importance"),
            code(
                "import pandas as pd\n"
                "from sklearn.inspection import permutation_importance\n\n"
                "# TODO: Compute permutation importance using ROC-AUC or average_precision.\n"
                "r = permutation_importance(\n"
                "    rf,\n"
                "    X_test,\n"
                "    y_test,\n"
                "    n_repeats=30,\n"
                "    random_state=0,\n"
                "    scoring='roc_auc',\n"
                ")\n"
                "\n"
                "perm = pd.Series(r.importances_mean, index=x_cols).sort_values(ascending=False)\n"
                "perm\n"
            ),
        ]

    if spec.path.endswith("04_walk_forward_validation.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Evaluate recession prediction stability over time using walk-forward validation.\n\n"
                "Walk-forward answers the question:\n"
                "- \"Does my model work in *multiple eras*, or only in the era I trained on?\"\n"
            ),
            md(primer("sklearn_pipelines")),
            md(
                f"<a id=\"{slugify('Walk-forward splits')}\"></a>\n"
                "## Walk-forward splits\n\n"
                "### Goal\n"
                "Generate a sequence of train/test splits that move forward through time.\n"
            ),
            md("### Your Turn (1): Load data and define X/y"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "y_col = 'target_recession_next_q'\n"
                "x_cols = ['T10Y2Y_lag1', 'UNRATE_lag1', 'FEDFUNDS_lag1', 'INDPRO_lag1', 'RSAFS_lag1']\n"
                "df_m = df[[y_col] + x_cols].dropna().copy()\n"
                "\n"
                "X = df_m[x_cols]\n"
                "y = df_m[y_col].astype(int)\n"
                "X.shape, y.mean()\n"
            ),
            md("### Your Turn (2): Generate walk-forward splits"),
            code(
                "from src.evaluation import walk_forward_splits\n\n"
                "# TODO: Choose split settings.\n"
                "# initial_train_size: first training window size\n"
                "# test_size: number of quarters per fold\n"
                "splits = list(walk_forward_splits(len(df_m), initial_train_size=40, test_size=8))\n"
                "len(splits), splits[0]\n"
            ),
            md(
                f"<a id=\"{slugify('Metric stability')}\"></a>\n"
                "## Metric stability\n\n"
                "### Goal\n"
                "Fit the same model on each fold and track metrics across time.\n"
            ),
            md("### Your Turn (1): Evaluate a model across folds"),
            code(
                "import pandas as pd\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from src.evaluation import classification_metrics\n\n"
                "rows = []\n"
                "for sp in splits:\n"
                "    X_tr = X.iloc[sp.train_slice]\n"
                "    y_tr = y.iloc[sp.train_slice]\n"
                "    X_te = X.iloc[sp.test_slice]\n"
                "    y_te = y.iloc[sp.test_slice]\n"
                "\n"
                "    clf = Pipeline([\n"
                "        ('scaler', StandardScaler()),\n"
                "        ('model', LogisticRegression(max_iter=5000)),\n"
                "    ])\n"
                "    clf.fit(X_tr, y_tr)\n"
                "    p = clf.predict_proba(X_te)[:, 1]\n"
                "    m = classification_metrics(y_te.to_numpy(), p, threshold=0.5)\n"
                "\n"
                "    rows.append({\n"
                "        'train_end': X_tr.index.max(),\n"
                "        'test_end': X_te.index.max(),\n"
                "        **m,\n"
                "    })\n"
                "\n"
                "wf = pd.DataFrame(rows).set_index('test_end')\n"
                "wf.head()\n"
            ),
            md("### Your Turn (2): Plot metrics over time"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Plot roc_auc, pr_auc, and brier over time.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Failure analysis')}\"></a>\n"
                "## Failure analysis\n\n"
                "### Goal\n"
                "Identify which eras the model struggles with and investigate why.\n"
            ),
            md("### Your Turn (1): Find the worst fold"),
            code(
                "# TODO: Pick a metric and find the worst fold.\n"
                "# Example: lowest PR-AUC\n"
                "worst = wf.sort_values('pr_auc').head(1)\n"
                "worst\n"
            ),
            md("### Your Turn (2): Inspect features in that era"),
            code(
                "# TODO: Look at feature distributions in the worst fold.\n"
                "# Compare to a better-performing fold.\n"
                "...\n"
            ),
        ]

    # Unsupervised
    if spec.path.endswith("01_pca_macro_factors.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Use PCA to extract a small number of macro factors from many indicators.\n\n"
                "Why PCA is useful here:\n"
                "- macro series are correlated\n"
                "- PCA creates orthogonal (uncorrelated) components\n"
                "- components can act like \"macro factors\" (growth, inflation, rates, etc.)\n"
            ),
            md(primer("pandas_time_series")),
            md(
                f"<a id=\"{slugify('Standardize')}\"></a>\n"
                "## Standardize\n\n"
                "### Goal\n"
                "Load a monthly panel and standardize features (mean 0, std 1).\n\n"
                "PCA is sensitive to scale: if one variable has larger units, it can dominate the components.\n"
            ),
            md("### Your Turn (1): Load panel_monthly.csv (or sample)"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'panel_monthly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'panel_monthly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "df.head()\n"
            ),
            md("### Your Turn (2): Build X and standardize"),
            code(
                "from sklearn.preprocessing import StandardScaler\n\n"
                "x_cols = df.columns.tolist()\n"
                "X = df[x_cols].dropna().copy()\n"
                "\n"
                "sc = StandardScaler().fit(X)\n"
                "X_s = sc.transform(X)\n"
                "\n"
                "# TODO: Validate standardization\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Fit PCA')}\"></a>\n"
                "## Fit PCA\n\n"
                "### Goal\n"
                "Fit PCA and inspect explained variance.\n"
            ),
            md("### Your Turn (1): Fit PCA"),
            code(
                "import numpy as np\n"
                "import pandas as pd\n"
                "from sklearn.decomposition import PCA\n\n"
                "pca = PCA(n_components=3).fit(X_s)\n"
                "\n"
                "evr = pd.Series(pca.explained_variance_ratio_, index=[f'PC{i+1}' for i in range(pca.n_components_)])\n"
                "evr\n"
            ),
            md("### Your Turn (2): Scree plot"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Plot explained variance ratio by component.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Interpret loadings')}\"></a>\n"
                "## Interpret loadings\n\n"
                "### Goal\n"
                "Interpret what each component represents in economic terms.\n\n"
                "Loadings tell you which original variables contribute most to each component.\n"
            ),
            md("### Your Turn (1): Build a loadings table"),
            code(
                "import pandas as pd\n\n"
                "loadings = pd.DataFrame(\n"
                "    pca.components_.T,\n"
                "    index=x_cols,\n"
                "    columns=[f'PC{i+1}' for i in range(pca.n_components_)],\n"
                ")\n"
                "\n"
                "# TODO: For each PC, list the top + and top - loadings.\n"
                "...\n"
            ),
            md("### Your Turn (2): Name the components"),
            code(
                "# TODO: Write a name/interpretation for PC1 and PC2.\n"
                "notes = \"\"\"\n"
                "PC1: ...\n"
                "PC2: ...\n"
                "\"\"\"\n"
                "print(notes)\n"
            ),
        ]

    if spec.path.endswith("02_clustering_macro_regimes.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Cluster macro periods into \"regimes\" and relate regimes to your technical recession label.\n\n"
                "Clustering is exploratory:\n"
                "- you are not predicting a target\n"
                "- you are summarizing structure in the data\n"
            ),
            md(primer("pandas_time_series")),
            md(
                f"<a id=\"{slugify('Clustering')}\"></a>\n"
                "## Clustering\n\n"
                "### Goal\n"
                "Choose a feature representation and fit a clustering algorithm.\n"
            ),
            md("### Your Turn (1): Load data and choose features to cluster"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "# TODO: Choose a small set of features (levels or lags).\n"
                "x_cols = ['UNRATE', 'FEDFUNDS', 'CPIAUCSL', 'INDPRO', 'RSAFS', 'T10Y2Y']\n"
                "\n"
                "df_m = df[x_cols + ['recession']].dropna().copy()\n"
                "df_m.head()\n"
            ),
            md("### Your Turn (2): Standardize features"),
            code(
                "from sklearn.preprocessing import StandardScaler\n\n"
                "sc = StandardScaler().fit(df_m[x_cols])\n"
                "X = sc.transform(df_m[x_cols])\n"
                "\n"
                "# TODO: sanity-check scaling (mean ~0, std ~1)\n"
                "...\n"
            ),
            md("### Your Turn (3): Fit a clustering model (KMeans)"),
            code(
                "from sklearn.cluster import KMeans\n\n"
                "# TODO: Choose a k to start (you'll justify it in the next section)\n"
                "k = 4\n"
                "km = KMeans(n_clusters=k, random_state=0, n_init=20)\n"
                "labels = km.fit_predict(X)\n"
                "\n"
                "df_m['cluster'] = labels\n"
                "df_m['cluster'].value_counts()\n"
            ),
            md(
                f"<a id=\"{slugify('Choose k')}\"></a>\n"
                "## Choose k\n\n"
                "### Goal\n"
                "Try multiple k values and use a diagnostic to pick one.\n\n"
                "We'll use the silhouette score as a simple quantitative guide (not a proof).\n"
            ),
            md("### Your Turn: Compute silhouette scores for k=2..6"),
            code(
                "import pandas as pd\n"
                "from sklearn.cluster import KMeans\n"
                "from sklearn.metrics import silhouette_score\n\n"
                "rows = []\n"
                "for k in range(2, 7):\n"
                "    km = KMeans(n_clusters=k, random_state=0, n_init=20)\n"
                "    lab = km.fit_predict(X)\n"
                "    rows.append({'k': k, 'silhouette': float(silhouette_score(X, lab))})\n"
                "\n"
                "pd.DataFrame(rows)\n"
            ),
            md(
                f"<a id=\"{slugify('Relate to recessions')}\"></a>\n"
                "## Relate to recessions\n\n"
                "### Goal\n"
                "Relate cluster assignments to recession periods.\n\n"
                "A useful first question:\n"
                "- Which clusters have the highest recession rate?\n"
            ),
            md("### Your Turn (1): Recession rate by cluster"),
            code(
                "import pandas as pd\n\n"
                "rate = df_m.groupby('cluster')['recession'].mean().sort_values(ascending=False)\n"
                "rate\n"
            ),
            md("### Your Turn (2): Visualize clusters over time"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Plot one feature over time and color by cluster.\n"
                "# Or plot clusters as colored bands over the timeline.\n"
                "...\n"
            ),
        ]

    if spec.path.endswith("03_anomaly_detection.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Detect unusual macro periods (potential crises) using anomaly detection.\n\n"
                "This is not about predicting recessions directly. It's about:\n"
                "- flagging periods that look \"different\" in feature space\n"
                "- interpreting what those periods correspond to historically\n"
            ),
            md(primer("pandas_time_series")),
            md(
                f"<a id=\"{slugify('Fit detector')}\"></a>\n"
                "## Fit detector\n\n"
                "### Goal\n"
                "Fit an anomaly detector on standardized macro features.\n"
            ),
            md("### Your Turn (1): Load data and choose features"),
            code(
                "import pandas as pd\n\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "if path.exists():\n"
                "    df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "else:\n"
                "    df = pd.read_csv(SAMPLE_DIR / 'macro_quarterly_sample.csv', index_col=0, parse_dates=True)\n"
                "\n"
                "x_cols = ['UNRATE', 'FEDFUNDS', 'CPIAUCSL', 'INDPRO', 'RSAFS', 'T10Y2Y']\n"
                "df_m = df[x_cols + ['recession']].dropna().copy()\n"
                "df_m.head()\n"
            ),
            md("### Your Turn (2): Standardize and fit IsolationForest"),
            code(
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.ensemble import IsolationForest\n\n"
                "sc = StandardScaler().fit(df_m[x_cols])\n"
                "X = sc.transform(df_m[x_cols])\n"
                "\n"
                "# TODO: Choose contamination (expected fraction of anomalies)\n"
                "iso = IsolationForest(contamination=0.05, random_state=0)\n"
                "iso.fit(X)\n"
                "\n"
                "# Higher score means more normal in sklearn; we'll invert for \"anomaly score\"\n"
                "score_normal = iso.score_samples(X)\n"
                "df_m['anomaly_score'] = -score_normal\n"
                "df_m[['anomaly_score']].head()\n"
            ),
            md(
                f"<a id=\"{slugify('Inspect anomalies')}\"></a>\n"
                "## Inspect anomalies\n\n"
                "### Goal\n"
                "Identify the most anomalous periods and inspect them.\n"
            ),
            md("### Your Turn (1): List top anomalous dates"),
            code(
                "# TODO: Sort by anomaly_score and print the top 10 most anomalous dates.\n"
                "df_m[['anomaly_score']].sort_values('anomaly_score', ascending=False).head(10)\n"
            ),
            md("### Your Turn (2): Plot anomaly score over time"),
            code(
                "import matplotlib.pyplot as plt\n\n"
                "# TODO: Plot df_m['anomaly_score'] over time.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Compare to recessions')}\"></a>\n"
                "## Compare to recessions\n\n"
                "### Goal\n"
                "Compare anomaly flags to technical recession labels.\n\n"
                "Questions:\n"
                "- Do anomalies cluster around recessions?\n"
                "- Are there anomalies outside recessions (e.g., inflation shocks)?\n"
            ),
            md("### Your Turn: Simple comparison"),
            code(
                "# TODO: Define a binary anomaly flag (top X%) and compute recession rate among anomalies.\n"
                "...\n"
            ),
        ]

    # Model ops
    if spec.path.endswith("01_reproducible_pipeline_design.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Understand how to turn notebooks into reproducible runs with configs + artifacts.\n\n"
                "A model is not \"done\" when you get a good plot.\n"
                "A model is done when you can re-run it and reproduce:\n"
                "- the dataset used\n"
                "- the features used\n"
                "- the metrics\n"
                "- the predictions\n"
            ),
            md(primer("paths_and_env")),
            md(
                f"<a id=\"{slugify('Configs')}\"></a>\n"
                "## Configs\n\n"
                "### Goal\n"
                "Inspect a YAML config and understand what it controls.\n"
            ),
            md("### Your Turn (1): Load and inspect configs/recession.yaml"),
            code(
                "import yaml\n"
                "from pathlib import Path\n\n"
                "cfg_path = PROJECT_ROOT / 'configs' / 'recession.yaml'\n"
                "cfg = yaml.safe_load(cfg_path.read_text())\n"
                "\n"
                "# TODO: Print top-level keys and explain what each one controls.\n"
                "cfg.keys()\n"
            ),
            md("### Your Turn (2): Find where config values are used"),
            code(
                "# TODO: Open scripts/train_recession.py and scripts/build_datasets.py.\n"
                "# Find how 'series', 'feature settings', and 'split rules' are used.\n"
                "# Write a short list of 'hard-coded' vs 'configurable'.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Outputs')}\"></a>\n"
                "## Outputs\n\n"
                "### Goal\n"
                "Run a pipeline and inspect the artifact bundle under `outputs/<run_id>/`.\n"
            ),
            md("### Your Turn: Run the pipeline from your terminal"),
            md(
                "Run these commands in terminal (from repo root):\n"
                "- `python scripts/build_datasets.py --recession-config configs/recession.yaml --census-config configs/census.yaml`\n"
                "- `python scripts/train_recession.py --config configs/recession.yaml`\n\n"
                "Then come back here and inspect the generated `outputs/<run_id>/` folder.\n"
            ),
            md("### Your Turn (2): Inspect outputs/ in Python"),
            code(
                "from pathlib import Path\n\n"
                "# TODO: List run folders under outputs/\n"
                "out_dir = PROJECT_ROOT / 'outputs'\n"
                "runs = sorted([p for p in out_dir.glob('*') if p.is_dir()])\n"
                "runs[-3:]\n"
            ),
            md(
                f"<a id=\"{slugify('Reproducibility')}\"></a>\n"
                "## Reproducibility\n\n"
                "### Goal\n"
                "Verify that a run is self-describing (you can tell what it did).\n\n"
                "Minimum expected artifacts:\n"
                "- `model.joblib`\n"
                "- `metrics.json`\n"
                "- `predictions.csv`\n"
            ),
            md("### Your Turn: Check artifact bundle completeness"),
            code(
                "# TODO: Pick the newest run folder and check expected files exist.\n"
                "if not runs:\n"
                "    raise RuntimeError('No runs found. Did you run the training script?')\n"
                "\n"
                "run = runs[-1]\n"
                "expected = ['model.joblib', 'metrics.json', 'predictions.csv']\n"
                "for name in expected:\n"
                "    print(name, (run / name).exists())\n"
                "...\n"
            ),
        ]

    if spec.path.endswith("02_build_cli_train_predict.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Practice model ops by extending the CLI scripts:\n"
                "- add flags/config controls\n"
                "- generate artifact bundles\n"
                "- run predict to produce new outputs\n\n"
                "This notebook is hands-on *engineering*, not just analysis.\n"
            ),
            md(primer("paths_and_env")),
            md(
                f"<a id=\"{slugify('Training CLI')}\"></a>\n"
                "## Training CLI\n\n"
                "### Goal\n"
                "Extend `scripts/train_recession.py` so you can control key behavior from the command line.\n"
            ),
            md("### Your Turn (1): Inspect the current CLI"),
            code(
                "# TODO: Open scripts/train_recession.py and find:\n"
                "# - how argparse is set up\n"
                "# - what args exist today\n"
                "# - what config fields are read\n"
                "...\n"
            ),
            md("### Your Turn (2): Add a meaningful flag"),
            md(
                "Implement at least one of these:\n"
                "- `--model logistic|random_forest`\n"
                "- `--include-gdp-features true|false`\n"
                "- `--test-size 0.2`\n\n"
                "Constraints:\n"
                "- default behavior should remain unchanged\n"
                "- the selected option must be written to the run folder (as JSON/YAML)\n"
            ),
            code(
                "# TODO: After implementing, re-run training and confirm it works.\n"
                "# In terminal: python scripts/train_recession.py --config configs/recession.yaml --model random_forest\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Prediction CLI')}\"></a>\n"
                "## Prediction CLI\n\n"
                "### Goal\n"
                "Extend `scripts/predict_recession.py` to support useful output controls.\n"
            ),
            md("### Your Turn: Add a filter option"),
            md(
                "Implement one option:\n"
                "- `--last-n 20` (only write last N predictions)\n"
                "- `--from-date 2010-01-01` (filter by date)\n\n"
                "Make sure the filter is applied to the output CSV.\n"
            ),
            code(
                "# TODO: After implementing, run prediction on your latest run.\n"
                "# In terminal: python scripts/predict_recession.py --run-id <run_id> --last-n 20\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Artifacts')}\"></a>\n"
                "## Artifacts\n\n"
                "### Goal\n"
                "Verify your artifact bundle is complete and interpretable.\n"
            ),
            md("### Your Turn: Inspect the newest run artifacts"),
            code(
                "from pathlib import Path\n\n"
                "out_dir = PROJECT_ROOT / 'outputs'\n"
                "runs = sorted([p for p in out_dir.glob('*') if p.is_dir()])\n"
                "if not runs:\n"
                "    raise RuntimeError('No runs found. Run training first.')\n"
                "\n"
                "run = runs[-1]\n"
                "print('run:', run.name)\n"
                "print('files:', [p.name for p in run.iterdir()])\n"
                "...\n"
            ),
        ]

    if spec.path.endswith("03_model_cards_and_reporting.ipynb"):
        cells += [
            md(
                "## Goal\n"
                "Write a \"model card\"-style document for one training run.\n\n"
                "A model card forces you to answer:\n"
                "- What is this model for?\n"
                "- What data did it use?\n"
                "- How was it evaluated?\n"
                "- What are the limitations and risks?\n"
            ),
            md(primer("paths_and_env")),
            md(
                f"<a id=\"{slugify('Model card')}\"></a>\n"
                "## Model card\n\n"
                "### Goal\n"
                "Fill out the provided report template for one run.\n"
            ),
            md("### Your Turn (1): Pick a run folder"),
            code(
                "from pathlib import Path\n\n"
                "out_dir = PROJECT_ROOT / 'outputs'\n"
                "runs = sorted([p for p in out_dir.glob('*') if p.is_dir()])\n"
                "runs[-3:]\n"
            ),
            md("### Your Turn (2): Load metrics.json and predictions.csv"),
            code(
                "import json\n"
                "import pandas as pd\n\n"
                "# TODO: Pick a run folder (e.g., the newest)\n"
                "run = runs[-1]\n"
                "\n"
                "metrics = json.loads((run / 'metrics.json').read_text())\n"
                "preds = pd.read_csv(run / 'predictions.csv')\n"
                "\n"
                "metrics, preds.head()\n"
            ),
            md(
                f"<a id=\"{slugify('Reporting')}\"></a>\n"
                "## Reporting\n\n"
                "### Goal\n"
                "Connect artifacts to a written narrative.\n\n"
                "Write a report that includes:\n"
                "- what you predicted (label definition)\n"
                "- dataset + time range\n"
                "- train/test or walk-forward evaluation summary\n"
                "- key plots/metrics\n"
            ),
            md("### Your Turn: Fill reports/capstone_report.md for this run"),
            code(
                "# TODO: Open reports/capstone_report.md and fill it using the artifacts above.\n"
                "# - insert metrics\n"
                "# - insert a short interpretation narrative\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Limitations')}\"></a>\n"
                "## Limitations\n\n"
                "### Goal\n"
                "Write a high-quality limitations section.\n\n"
                "Prompts:\n"
                "- data limitations (timing, revisions, coverage)\n"
                "- label limitations (technical recession proxy)\n"
                "- evaluation limitations (few recessions, era changes)\n"
                "- deployment limitations (stale model, monitoring)\n"
            ),
            md("### Your Turn: Write limitations as bullet points"),
            code("limitations = [\n    '...'\n]\nlimitations\n"),
        ]

    # Capstone
    if spec.path.endswith("00_capstone_brief.ipynb"):
        cells += [
            md(
                "## Capstone Brief\n"
                "You will produce a final model + report + dashboard.\n\n"
                "Treat this like a mini research + engineering project:\n"
                "- clear question\n"
                "- defensible dataset + label\n"
                "- time-aware evaluation\n"
                "- careful interpretation\n"
                "- reproducible artifacts\n"
            ),
            md(
                f"<a id=\"{slugify('Deliverables')}\"></a>\n"
                "## Deliverables\n"
                "- A reproducible run under `outputs/<run_id>/`\n"
                "- A written report in `reports/capstone_report.md`\n"
                "- A Streamlit dashboard that loads your artifacts (`apps/streamlit_app.py`)\n"
            ),
            md(
                f"<a id=\"{slugify('Rubric')}\"></a>\n"
                "## Rubric\n"
                "- Problem framing and label definition\n"
                "- Data pipeline correctness (no leakage)\n"
                "- Evaluation quality (time-aware)\n"
                "- Interpretation and limitations\n"
                "- Reproducibility\n"
            ),
            md(
                f"<a id=\"{slugify('Scope selection')}\"></a>\n"
                "## Scope selection\n\n"
                "Pick one:\n"
                "- **Option A: Macro-only** recession prediction + deep evaluation/interpretation\n"
                "- **Option B: Macro + Micro** (add a cross-sectional section in your report)\n"
            ),
            md("### Your Turn: Decide and write down your scope"),
            code("scope = \"...\"  # TODO: 'macro' or 'macro+micro'\nprint(scope)\n"),
        ]

    if spec.path.endswith("01_capstone_workspace.ipynb"):
        cells += [
            md(
                "## Capstone Workspace\n"
                "This notebook is your working area. It should end with:\n"
                "- artifacts under `outputs/<run_id>/`\n"
                "- an updated report `reports/capstone_report.md`\n"
                "- a working Streamlit dashboard\n"
            ),
            md(
                f"<a id=\"{slugify('Data')}\"></a>\n"
                "## Data\n\n"
                "### Goal\n"
                "Choose the dataset(s), target, and feature set you will use.\n"
            ),
            md("### Your Turn (1): Load final dataset"),
            code(
                "import pandas as pd\n\n"
                "# TODO: Load macro_quarterly.csv from data/processed/.\n"
                "# If it doesn't exist, build it first in the data notebooks.\n"
                "path = PROCESSED_DIR / 'macro_quarterly.csv'\n"
                "df = pd.read_csv(path, index_col=0, parse_dates=True)\n"
                "df.tail()\n"
            ),
            md("### Your Turn (2): Define target + features"),
            code(
                "# TODO: Choose your target and feature list.\n"
                "# Example target: 'target_recession_next_q'\n"
                "y_col = 'target_recession_next_q'\n"
                "\n"
                "x_cols = [\n"
                "    # TODO: your features\n"
                "]\n"
                "\n"
                "df_m = df[[y_col] + x_cols].dropna().copy()\n"
                "df_m.head()\n"
            ),
            md(
                f"<a id=\"{slugify('Modeling')}\"></a>\n"
                "## Modeling\n\n"
                "### Goal\n"
                "Train at least 2 models and select one based on time-aware evaluation.\n"
            ),
            md("### Your Turn (1): Train/test split + baseline"),
            code(
                "# TODO: Implement a time split and fit a baseline model.\n"
                "# Baselines can include:\n"
                "# - logistic regression\n"
                "# - simple rule model\n"
                "...\n"
            ),
            md("### Your Turn (2): Walk-forward evaluation"),
            code(
                "# TODO: Implement walk-forward evaluation for your chosen models.\n"
                "# Save fold metrics and compare stability.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Interpretation')}\"></a>\n"
                "## Interpretation\n\n"
                "### Goal\n"
                "Explain what your model learned and where it fails.\n"
            ),
            md("### Your Turn (1): Feature interpretation"),
            code(
                "# TODO: Provide at least one of:\n"
                "# - coefficient interpretation (logistic regression odds ratios)\n"
                "# - permutation importance\n"
                "# - partial dependence (optional)\n"
                "...\n"
            ),
            md("### Your Turn (2): Error analysis"),
            code(
                "# TODO: Identify false positives and false negatives in the test period.\n"
                "# Compare indicator levels during those errors.\n"
                "...\n"
            ),
            md(
                f"<a id=\"{slugify('Artifacts')}\"></a>\n"
                "## Artifacts\n\n"
                "### Goal\n"
                "Write a complete artifact bundle and update your report.\n"
            ),
            md("### Your Turn (1): Save artifacts under outputs/<run_id>/"),
            code(
                "# TODO: Save:\n"
                "# - model.joblib\n"
                "# - metrics.json\n"
                "# - predictions.csv\n"
                "# - plots/ (optional)\n"
                "...\n"
            ),
            md("### Your Turn (2): Update report"),
            code(
                "# TODO: Update reports/capstone_report.md with:\n"
                "# - your final metrics\n"
                "# - your interpretation narrative\n"
                "# - limitations and monitoring plan\n"
                "...\n"
            ),
            md("### Your Turn (3): Run dashboard"),
            md("In terminal: `streamlit run apps/streamlit_app.py`"),
        ]

    # A lightweight, TODO-driven checkpoint so learners can self-validate before moving on.
    cells += [
        md(
            "<a id=\"checkpoint-self-check\"></a>\n"
            "## Checkpoint (Self-Check)\n"
            "Run a few asserts and write 2-3 sentences summarizing what you verified.\n"
        ),
        code(notebook_checkpoint_snippet(stem, category)),
    ]

    # Generic end-of-notebook prompts (kept short; guides contain the deep dive).
    cells += [
        md(
            "## Extensions (Optional)\n"
            "- Try one additional variant beyond the main path (different features, different split, different model).\n"
            "- Write down what improved, what got worse, and your hypothesis for why.\n"
        ),
        md(
            "## Reflection\n"
            "- What did you assume implicitly (about timing, availability, stationarity, or costs)?\n"
            "- If you had to ship this model, what would you monitor?\n"
        ),
    ]

    # Reference solutions are intentionally collapsed so the notebook stays hands-on.
    cells.append(md(solutions_markdown(stem, spec.sections)))

    out_path.write_text(json.dumps(nb(cells), indent=2))
 
 
# Guide templates and deep-dive content live under scripts/templates/.

def concept_leakage_deep_dive() -> str:
    return concept("leakage")


def concept_time_split_deep_dive() -> str:
    return concept("time_split")


def concept_multicollinearity_vif_deep_dive() -> str:
    return concept("multicollinearity_vif")


def concept_hac_newey_west_deep_dive() -> str:
    return concept("hac_newey_west")


def concept_logistic_regression_odds_deep_dive() -> str:
    return concept("logistic_odds")


def concept_calibration_brier_deep_dive() -> str:
    return concept("calibration_brier")


def concept_walk_forward_validation_deep_dive() -> str:
    return concept("walk_forward_validation")


def concept_pca_loadings_deep_dive() -> str:
    return concept("pca_loadings")


def concept_api_and_caching_deep_dive() -> str:
    return concept("api_caching")


def concept_resampling_and_alignment_deep_dive() -> str:
    return concept("resampling_alignment")


def concept_gdp_growth_and_recession_label_deep_dive() -> str:
    return concept("gdp_growth_recession_label")


def concept_quarterly_feature_engineering_deep_dive() -> str:
    return concept("quarterly_feature_engineering")


def concept_micro_regression_log_log_deep_dive() -> str:
    return concept("micro_log_log")


def concept_robust_se_hc3_deep_dive() -> str:
    return concept("robust_se_hc3")


def concept_omitted_variable_bias_deep_dive() -> str:
    return concept("omitted_variable_bias")


def concept_regularization_ridge_lasso_deep_dive() -> str:
    return concept("regularization_ridge_lasso")


def concept_rolling_regression_stability_deep_dive() -> str:
    return concept("rolling_regression")


def concept_class_imbalance_and_metrics_deep_dive() -> str:
    return concept("class_imbalance_metrics")


def concept_tree_models_and_importance_deep_dive() -> str:
    return concept("tree_importance")


def concept_clustering_regimes_deep_dive() -> str:
    return concept("clustering_regimes")


def concept_anomaly_detection_deep_dive() -> str:
    return concept("anomaly_detection")


def guide_code_map(category: str, stem: str) -> list[str]:
    """Return a short list of project code references relevant to this guide."""

    common = [
        "`src/data.py`: caching helpers (`load_or_fetch_json`, `load_json`, `save_json`)",
        "`src/features.py`: feature helpers (`to_monthly`, `add_lag_features`, `add_pct_change_features`, `add_rolling_features`)",
        "`src/evaluation.py`: splits + metrics (`time_train_test_split_index`, `walk_forward_splits`, `regression_metrics`, `classification_metrics`)",
    ]

    if category == "00_foundations":
        return [
            "`scripts/scaffold_curriculum.py`: how this curriculum is generated (for curiosity)",
            "`src/evaluation.py`: time splits and metrics used later",
            *common,
        ]

    if category == "01_data":
        out = [
            "`src/fred_api.py`: FRED client (`fetch_series_meta`, `fetch_series_observations`, `observations_to_frame`)",
            "`src/census_api.py`: Census/ACS client (`fetch_variables`, `fetch_acs`)",
            "`src/macro.py`: GDP + labels (`gdp_growth_qoq`, `gdp_growth_yoy`, `technical_recession_label`, `monthly_to_quarterly`)",
            "`scripts/build_datasets.py`: end-to-end dataset builder",
            *common,
        ]
        if stem == "00_fred_api_and_caching":
            out.insert(0, "`scripts/fetch_fred.py`: CLI fetch for FRED")
        if stem == "04_census_api_microdata_fetch":
            out.insert(0, "`scripts/fetch_census.py`: CLI fetch for Census/ACS")
        return out

    if category == "02_regression":
        out = [
            "`src/econometrics.py`: OLS + robust SE (`fit_ols`, `fit_ols_hc3`, `fit_ols_hac`) + multicollinearity (`vif_table`)",
            "`src/macro.py`: GDP + labels (`gdp_growth_*`, `technical_recession_label`)",
            "`src/evaluation.py`: regression metrics helpers",
            *common,
        ]
        return out

    if category == "03_classification":
        return [
            "`src/evaluation.py`: classification metrics (ROC-AUC, PR-AUC, Brier, precision/recall)",
            "`scripts/train_recession.py`: training script that writes artifacts",
            "`scripts/predict_recession.py`: prediction script that loads artifacts",
            *common,
        ]

    if category == "04_unsupervised":
        return [
            "`src/features.py`: feature engineering helpers (standardization happens in notebooks)",
            "`data/sample/panel_monthly_sample.csv`: offline dataset for experimentation",
            *common,
        ]

    if category == "05_model_ops":
        return [
            "`configs/recession.yaml`: example config for recession training",
            "`scripts/build_datasets.py`: dataset builder (writes to data/processed/)",
            "`scripts/train_recession.py`: training script (writes to outputs/<run_id>/)",
            "`scripts/predict_recession.py`: prediction script (writes predictions.csv)",
        ]

    if category == "06_capstone":
        return [
            "`apps/streamlit_app.py`: dashboard that reads artifacts",
            "`reports/capstone_report.md`: report template/output",
            "`outputs/`: artifact bundles from training runs (models/metrics/preds/plots)",
        ]

    if category == "07_causal":
        return [
            "`src/causal.py`: panel + IV helpers (`to_panel_index`, `fit_twfe_panel_ols`, `fit_iv_2sls`)",
            "`scripts/build_datasets.py`: ACS panel builder (writes data/processed/census_county_panel.csv)",
            "`src/census_api.py`: Census/ACS client (`fetch_acs`)",
            "`configs/census_panel.yaml`: panel config (years + variables)",
            "`data/sample/census_county_panel_sample.csv`: offline panel dataset",
            *common,
        ]

    if category == "08_time_series_econ":
        return [
            "`data/sample/panel_monthly_sample.csv`: offline macro panel",
            "`src/features.py`: safe lag/diff/rolling feature helpers",
            "`src/macro.py`: GDP growth + label helpers (for context)",
            *common,
        ]

    return common


def write_guide(spec: NotebookSpec, root: Path) -> None:
    nb_path = Path(spec.path)
    stem = nb_path.stem
    category = nb_path.parts[1] if len(nb_path.parts) > 1 else "misc"
    out_path = root / "docs" / "guides" / category / f"{stem}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Category-oriented guides: explain technical terms in-context and focus heavily on
    # stats/ML interpretation (the notebooks remain TODO-driven).

    header = f"This guide accompanies the notebook `{spec.path}`."
    base_steps = [f"Complete notebook section: {s}" for s in spec.sections]

    if category == "00_foundations":
        intro = (
            f"{header}\n\n"
            "This foundations module builds core intuition you will reuse in every later notebook.\n\n"
            "### Key Terms (defined)\n"
            "- **Time series**: data indexed by time; ordering is meaningful and must be respected.\n"
            "- **Leakage**: using future information in features/labels, producing unrealistically good results.\n"
            "- **Train/test split**: separating data for model fitting vs evaluation.\n"
            "- **Multicollinearity**: predictors are highly correlated; coefficients can become unstable.\n"
        )

        checklist_items = [
            *base_steps,
            "Run the bootstrap cell and confirm `PROJECT_ROOT` points to the repo root.",
            "Complete all TODOs (no `...` left).",
            "Write a short paragraph explaining a leakage example you created.",
        ]

        alt_example = (
            "```python\n"
            "# Toy leakage example (not the notebook data):\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "idx = pd.date_range('2020-01-01', periods=200, freq='D')\n"
            "y = pd.Series(np.sin(np.linspace(0, 12, len(idx))) + 0.1*np.random.randn(len(idx)), index=idx)\n"
            "\n"
            "# Correct feature: yesterday\n"
            "x_lag1 = y.shift(1)\n"
            "\n"
            "# Leakage feature: tomorrow (do NOT do this)\n"
            "x_leak = y.shift(-1)\n"
            "```\n"
        )

        technical = concept("core_foundations")

        # Stem-specific deep dives (first introduction gets the deepest treatment).
        if stem == "01_time_series_basics":
            technical = (
                "This notebook introduces the two most important ideas for economic ML:\n"
                "1) time-aware evaluation, and\n"
                "2) leakage prevention.\n\n"
                + concept("core_foundations")
                + "\n\n"
                + concept_time_split_deep_dive()
                + "\n\n"
                + concept_leakage_deep_dive()
            )
        elif stem == "02_stats_basics_for_ml":
            technical = (
                "This notebook introduces the core statistical vocabulary used throughout the project.\n\n"
                + concept("core_foundations")
                + "\n\n"
                + concept("correlation_causation")
                + "\n\n"
                + concept("bias_variance_overfitting")
                + "\n\n"
                + concept_multicollinearity_vif_deep_dive()
                + "\n\n"
                + concept("hypothesis_testing")
            )

        mistakes_items = [
            "Using `train_test_split(shuffle=True)` on time-indexed data.",
            "Looking at the test set repeatedly while tuning (\"test leakage\").",
            "Assuming a significant p-value implies causation.",
            "Running many tests/specs and treating a small p-value as proof (multiple testing / p-hacking).",
        ]

        summary = (
            "You now have the tooling to avoid the two most common beginner mistakes in economic ML:\n"
            "1) leaking future information, and\n"
            "2) over-interpreting correlated coefficients.\n"
        )

        readings_items = [
            "Hyndman & Athanasopoulos: Forecasting: Principles and Practice (time series basics)",
            "Wooldridge: Introductory Econometrics (interpretation + pitfalls)",
            "scikit-learn docs: model evaluation and cross-validation",
        ]

    elif category == "01_data":
        intro = (
            f"{header}\n\n"
            "This data module builds the datasets used throughout the project: a macro panel (FRED) and a micro cross-section (Census/ACS).\n\n"
            "### Key Terms (defined)\n"
            "- **API endpoint**: a URL path that returns a specific dataset.\n"
            "- **Caching**: saving raw responses locally so experiments are reproducible and fast.\n"
            "- **Frequency alignment**: converting mixed-frequency series (daily/monthly/quarterly) onto a common timeline.\n"
            "- **Quarter-end timestamp**: representing a quarter by its final date to make merges unambiguous.\n"
        )

        checklist_items = [
            *base_steps,
            "Fetch or load sample data and inspect schemas (columns, dtypes, index).",
            "Build the GDP growth series and recession label exactly as specified.",
            "Create a quarterly modeling table with `target_recession_next_q` and no obvious leakage.",
        ]

        alt_example = (
            "```python\n"
            "# Toy GDP growth + technical recession label (not real GDP):\n"
            "import pandas as pd\n"
            "\n"
            "idx = pd.date_range('2018-03-31', periods=12, freq='QE')\n"
            "gdp = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 102, 103, 104, 105], index=idx)\n"
            "\n"
            "growth_qoq = 100 * (gdp / gdp.shift(1) - 1)\n"
            "recession = ((growth_qoq < 0) & (growth_qoq.shift(1) < 0)).astype(int)\n"
            "target_next_q = recession.shift(-1)\n"
            "```\n"
        )

        technical = concept("core_data")

        # Stem-specific deep dives.
        if stem == "00_fred_api_and_caching":
            technical = concept("core_data") + "\n\n" + concept_api_and_caching_deep_dive()
        elif stem == "01_build_macro_monthly_panel":
            technical = concept("core_data") + "\n\n" + concept_resampling_and_alignment_deep_dive()
        elif stem == "02_gdp_growth_and_recession_label":
            technical = concept("core_data") + "\n\n" + concept_gdp_growth_and_recession_label_deep_dive()
        elif stem == "03_build_macro_quarterly_features":
            technical = concept("core_data") + "\n\n" + concept_quarterly_feature_engineering_deep_dive()
        elif stem == "04_census_api_microdata_fetch":
            technical = concept("core_data") + "\n\n" + concept_api_and_caching_deep_dive()

        mistakes_items = [
            "Merging quarterly GDP with monthly predictors without explicit aggregation (silent misalignment).",
            "Using future quarterly features (e.g., lag -1) by accident.",
            "Forgetting that daily series need resampling before joining.",
        ]

        summary = (
            "You now have a reproducible macro dataset with an explicit recession label and a micro dataset for cross-sectional inference.\n"
            "From here, the project focuses on modeling and interpretation.\n"
        )

        readings_items = [
            "FRED API documentation (series, observations)",
            "US Census API documentation (ACS endpoints, geography parameters)",
            "pandas documentation: resampling, merging/joining time series",
        ]

    elif category == "02_regression":
        intro = (
            f"{header}\n\n"
            "This regression module covers both prediction and inference, with a strong emphasis on interpretation.\n\n"
            "### Key Terms (defined)\n"
            "- **OLS (Ordinary Least Squares)**: chooses coefficients that minimize squared prediction errors.\n"
            "- **Coefficient**: expected change in the target per unit change in a feature (holding others fixed).\n"
            "- **Standard error (SE)**: uncertainty estimate for a coefficient.\n"
            "- **p-value**: probability of observing an effect at least as extreme if the true effect were zero (under assumptions).\n"
            "- **Confidence interval (CI)**: a range of plausible coefficient values under assumptions.\n"
            "- **Heteroskedasticity**: non-constant error variance; common in cross-section.\n"
            "- **Autocorrelation**: errors correlated over time; common in time series.\n"
            "- **HAC/Newey-West**: robust SE for time-series autocorrelation/heteroskedasticity.\n"
        )

        checklist_items = [
            *base_steps,
            "Fit at least one plain OLS model and one robust-SE variant (HC3 or HAC).",
            "Interpret coefficients in units (or standardized units) and explain what they do *not* mean.",
            "Run at least one diagnostic: residual plot, VIF table, or rolling coefficient stability plot.",
        ]

        alt_example = (
            "```python\n"
            "# Toy OLS with robust SE (not the notebook data):\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import statsmodels.api as sm\n"
            "\n"
            "rng = np.random.default_rng(0)\n"
            "x = rng.normal(size=200)\n"
            "y = 2.0 + 0.5*x + rng.normal(scale=1 + 0.5*np.abs(x), size=200)  # heteroskedastic errors\n"
            "X = sm.add_constant(pd.DataFrame({'x': x}))\n"
            "res = sm.OLS(y, X).fit()\n"
            "res_hc3 = res.get_robustcov_results(cov_type='HC3')\n"
            "```\n"
        )

        technical = concept("core_regression")

        if stem == "04_inference_time_series_hac":
            technical = (
                concept("core_regression")
                + "\n\n"
                + concept("hypothesis_testing")
                + "\n\n"
                + concept_hac_newey_west_deep_dive()
            )
        elif stem == "00_single_factor_regression_micro":
            technical = (
                concept("core_regression")
                + "\n\n"
                + concept_micro_regression_log_log_deep_dive()
                + "\n\n"
                + concept_robust_se_hc3_deep_dive()
                + "\n\n"
                + concept("hypothesis_testing")
                + "\n\n"
                + concept("multicollinearity_vif")
            )
        elif stem == "01_multifactor_regression_micro_controls":
            technical = (
                concept("core_regression")
                + "\n\n"
                + concept_omitted_variable_bias_deep_dive()
                + "\n\n"
                + concept_robust_se_hc3_deep_dive()
                + "\n\n"
                + concept("hypothesis_testing")
            )
        elif stem == "02_single_factor_regression_macro":
            technical = (
                concept("core_regression")
                + "\n\n"
                + concept_hac_newey_west_deep_dive()
                + "\n\n"
                + concept("hypothesis_testing")
            )
        elif stem == "03_multifactor_regression_macro":
            technical = concept("core_regression") + "\n\n" + concept_multicollinearity_vif_deep_dive()
        elif stem == "05_regularization_ridge_lasso":
            technical = concept("core_regression") + "\n\n" + concept_regularization_ridge_lasso_deep_dive()
        elif stem == "06_rolling_regressions_stability":
            technical = concept("core_regression") + "\n\n" + concept_rolling_regression_stability_deep_dive()

        mistakes_items = [
            "Interpreting a coefficient as causal without a causal design.",
            "Ignoring multicollinearity (high VIF) and over-trusting coefficient signs.",
            "Using naive SE on time series and over-trusting p-values.",
        ]

        summary = (
            "Regression is the core bridge between statistics and ML. You should now be able to:\n"
            "- fit interpretable linear models,\n"
            "- quantify uncertainty (robust SE), and\n"
            "- diagnose when coefficients are unstable.\n"
        )

        readings_items = [
            "Wooldridge: Introductory Econometrics (OLS, robust SE, interpretation)",
            "Angrist & Pischke: Mostly Harmless Econometrics (causal thinking)",
            "statsmodels docs: robust covariance (HCx, HAC)",
        ]

    elif category == "03_classification":
        intro = (
            f"{header}\n\n"
            "This classification module predicts **next-quarter technical recession** from macro indicators.\n\n"
            "### Key Terms (defined)\n"
            "- **Logistic regression**: a linear model that outputs probabilities via a sigmoid function.\n"
            "- **Log-odds**: `log(p/(1-p))`; logistic regression is linear in log-odds.\n"
            "- **Threshold**: rule converting probability into class (e.g., 1 if p>=0.5).\n"
            "- **Precision/Recall**: trade off false positives vs false negatives.\n"
            "- **ROC-AUC / PR-AUC**: threshold-free ranking metrics.\n"
            "- **Calibration**: whether predicted probabilities match observed frequencies.\n"
            "- **Brier score**: mean squared error of probabilities (lower is better).\n"
        )

        checklist_items = [
            *base_steps,
            "Establish baselines before fitting any model.",
            "Fit at least one probabilistic classifier and evaluate ROC-AUC, PR-AUC, and Brier score.",
            "Pick a threshold intentionally (cost-based or metric-based) and justify it.",
        ]

        alt_example = (
            "```python\n"
            "# Toy logistic regression (not the notebook data):\n"
            "import numpy as np\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "rng = np.random.default_rng(0)\n"
            "X = rng.normal(size=(300, 3))\n"
            "p = 1 / (1 + np.exp(-(0.2 + 1.0*X[:,0] - 0.8*X[:,1])))\n"
            "y = rng.binomial(1, p)\n"
            "\n"
            "clf = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])\n"
            "clf.fit(X, y)\n"
            "```\n"
        )

        technical = concept("core_classification")

        if stem == "01_logistic_recession_classifier":
            technical = concept("core_classification") + "\n\n" + concept_logistic_regression_odds_deep_dive()
        elif stem == "00_recession_classifier_baselines":
            technical = concept("core_classification") + "\n\n" + concept_class_imbalance_and_metrics_deep_dive()
        elif stem == "02_calibration_and_costs":
            technical = concept("core_classification") + "\n\n" + concept_calibration_brier_deep_dive()
        elif stem == "03_tree_models_and_importance":
            technical = concept("core_classification") + "\n\n" + concept_tree_models_and_importance_deep_dive()
        elif stem == "04_walk_forward_validation":
            technical = concept("core_classification") + "\n\n" + concept_walk_forward_validation_deep_dive()

        mistakes_items = [
            "Reporting only accuracy (can be misleading if recessions are rare).",
            "Picking threshold=0.5 by default without considering costs.",
            "Evaluating with random splits (time leakage).",
        ]

        summary = (
            "You should now be able to build a recession probability model and explain:\n"
            "- what the probability means,\n"
            "- how you evaluated it, and\n"
            "- why your chosen threshold makes sense.\n"
        )

        readings_items = [
            "scikit-learn docs: classification metrics, calibration",
            "Murphy: Machine Learning (probabilistic interpretation)",
            "Applied time-series evaluation articles (walk-forward validation)",
        ]

    elif category == "04_unsupervised":
        intro = (
            f"{header}\n\n"
            "This unsupervised module explores macro structure: factors, regimes, and anomalies.\n\n"
            "### Key Terms (defined)\n"
            "- **Unsupervised learning**: learning patterns without a labeled target.\n"
            "- **PCA**: rotates correlated features into uncorrelated components (factors).\n"
            "- **Loadings**: how strongly each original variable contributes to a component.\n"
            "- **Clustering**: grouping similar periods into regimes.\n"
            "- **Anomaly detection**: flagging unusual points (often crisis periods).\n"
        )

        checklist_items = [
            *base_steps,
            "Standardize features before PCA/clustering (units matter).",
            "Interpret components/clusters economically (give them names).",
            "Compare regimes/anomalies to your recession labels (do they align?).",
        ]

        alt_example = (
            "```python\n"
            "# Toy PCA (not the notebook data):\n"
            "import numpy as np\n"
            "from sklearn.decomposition import PCA\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "X = np.random.randn(200, 5)\n"
            "X = StandardScaler().fit_transform(X)\n"
            "pca = PCA(n_components=2).fit(X)\n"
            "```\n"
        )

        technical = concept("core_unsupervised")

        if stem == "01_pca_macro_factors":
            technical = concept("core_unsupervised") + "\n\n" + concept_pca_loadings_deep_dive()
        elif stem == "02_clustering_macro_regimes":
            technical = concept("core_unsupervised") + "\n\n" + concept_clustering_regimes_deep_dive()
        elif stem == "03_anomaly_detection":
            technical = concept("core_unsupervised") + "\n\n" + concept_anomaly_detection_deep_dive()

        mistakes_items = [
            "Forgetting to standardize (PCA will just pick the biggest-unit variable).",
            "Interpreting a cluster label as a causal regime without validation.",
            "Using too many components/clusters and overfitting noise.",
        ]

        summary = (
            "Unsupervised tools help you understand macro data structure even before prediction.\n"
            "They are especially useful for detecting regime shifts and crisis periods.\n"
        )

        readings_items = [
            "Jolliffe: Principal Component Analysis",
            "scikit-learn docs: PCA, clustering, anomaly detection",
        ]

    elif category == "05_model_ops":
        intro = (
            f"{header}\n\n"
            "This module turns notebooks into a reproducible workflow with configs, scripts, and artifacts.\n\n"
            "### Key Terms (defined)\n"
            "- **Artifact**: a saved output (model file, metrics JSON, predictions CSV).\n"
            "- **Run ID**: a unique identifier for a training run.\n"
            "- **Config**: a file that records choices (series list, features, split rules).\n"
            "- **Reproducibility**: ability to re-run and get consistent outputs.\n"
        )

        checklist_items = [
            *base_steps,
            "Run the build/train scripts and confirm outputs/<run_id> is created.",
            "Inspect run metadata and explain what is captured (and what is missing).",
            "Make at least one CLI extension (flag or config option).",
        ]

        alt_example = (
            "```python\n"
            "# Toy artifact layout:\n"
            "# outputs/20250101_120000/\n"
            "#   model.joblib\n"
            "#   metrics.json\n"
            "#   predictions.csv\n"
            "```\n"
        )

        technical = concept("core_model_ops")

        mistakes_items = [
            "Overwriting outputs without run IDs (losing provenance).",
            "Not recording feature list used for training (cannot reproduce predictions).",
            "Mixing data preparation and modeling in one script without clear interfaces.",
        ]

        summary = (
            "You now have a minimal 'model ops' workflow: build datasets, train models, save artifacts, and load them for prediction.\n"
        )

        readings_items = [
            "Google: ML Test Score (checklist for ML systems)",
            "Model Cards (Mitchell et al.)",
        ]

    elif category == "06_capstone":
        intro = (
            f"{header}\n\n"
            "This is the capstone: you will synthesize macro concepts, statistical inference, model evaluation, and reproducibility.\n\n"
            "### Key Terms (defined)\n"
            "- **Capstone**: a final project that integrates the entire curriculum.\n"
            "- **Rubric**: the criteria you will be evaluated against.\n"
            "- **Monitoring**: how you would detect degradation in a model over time.\n"
        )

        checklist_items = [
            *base_steps,
            "Define your final target/feature set and justify key design choices.",
            "Run walk-forward evaluation and report stability over time.",
            "Produce a complete report and a working Streamlit dashboard based on artifacts.",
        ]

        alt_example = (
            "```python\n"
            "# Capstone structure (high level):\n"
            "# 1) build dataset\n"
            "# 2) train multiple candidate models\n"
            "# 3) choose threshold and evaluate\n"
            "# 4) interpret and write report\n"
            "# 5) serve results in dashboard\n"
            "```\n"
        )

        technical = concept("core_capstone")

        mistakes_items = [
            "Changing feature engineering after looking at test results without re-running from scratch.",
            "Reporting a single metric without describing decision costs.",
            "Overclaiming (\"this predicts recessions\") without scope/limitations.",
        ]

        summary = (
            "If you complete the capstone well, you will have a portfolio-quality project: reproducible code, a report, and an interactive dashboard.\n"
        )

        readings_items = [
            "Wooldridge (inference) + Hyndman (forecasting) as complementary perspectives",
            "Mitchell et al.: Model Cards",
        ]

    elif category == "07_causal":
        intro = (
            f"{header}\n\n"
            "This module adds identification-focused econometrics: panels, DiD/event studies, and IV.\n\n"
            "### Key Terms (defined)\n"
            "- **Identification**: assumptions that justify a causal interpretation.\n"
            "- **Fixed effects (FE)**: controls for time-invariant unit differences.\n"
            "- **Clustered SE**: allows correlated errors within groups (e.g., state).\n"
            "- **DiD**: compares changes over time between treated and control units.\n"
            "- **IV/2SLS**: uses an instrument to address endogeneity.\n"
        )

        checklist_items = [
            *base_steps,
            "Write the causal question and identification assumptions before estimating.",
            "Run at least one diagnostic/falsification (pre-trends, placebo, weak-IV check).",
            "Report clustered SE (and number of clusters) when appropriate.",
        ]

        alt_example = (
            "```python\n"
            "# Toy DiD setup (not the notebook data):\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "df = pd.DataFrame({\n"
            "  'group': ['T']*50 + ['C']*50,\n"
            "  'post':  [0]*25 + [1]*25 + [0]*25 + [1]*25,\n"
            "})\n"
            "df['treated'] = (df['group'] == 'T').astype(int)\n"
            "df['D'] = df['treated'] * df['post']\n"
            "```\n"
        )

        technical = concept("core_causal")
        if stem == "00_build_census_county_panel":
            technical = concept("core_causal") + "\n\n" + concept("panel_fixed_effects")
        elif stem == "01_panel_fixed_effects_clustered_se":
            technical = concept("core_causal") + "\n\n" + concept("panel_fixed_effects") + "\n\n" + concept("clustered_se")
        elif stem == "02_difference_in_differences_event_study":
            technical = (
                concept("core_causal")
                + "\n\n"
                + concept("difference_in_differences")
                + "\n\n"
                + concept("event_study_parallel_trends")
                + "\n\n"
                + concept("clustered_se")
            )
        elif stem == "03_instrumental_variables_2sls":
            technical = concept("core_causal") + "\n\n" + concept("instrumental_variables")

        mistakes_items = [
            "Jumping to regression output without writing identification assumptions.",
            "Treating Granger-type correlations as causal effects (wrong question).",
            "Ignoring clustered/serial correlation and using overly small SE.",
            "For DiD: not checking pre-trends (leads) before interpreting effects.",
            "For IV: using weak instruments (no meaningful first stage).",
        ]

        summary = (
            "You now have a toolkit for causal estimation under explicit assumptions (FE/DiD/IV).\n"
            "The goal is disciplined thinking: identification first, estimation second.\n"
        )

        readings_items = [
            "Angrist & Pischke: Mostly Harmless Econometrics (design-based causal inference)",
            "Wooldridge: Econometric Analysis of Cross Section and Panel Data (FE/IV foundations)",
        ]

    elif category == "08_time_series_econ":
        intro = (
            f"{header}\n\n"
            "This module covers classical time-series econometrics: stationarity, cointegration/ECM, and VAR/IRFs.\n\n"
            "### Key Terms (defined)\n"
            "- **Stationarity**: stable statistical properties over time.\n"
            "- **Unit root**: nonstationary process where shocks accumulate (random walk-like).\n"
            "- **Cointegration**: nonstationary series with a stationary long-run relationship.\n"
            "- **VAR**: multivariate autoregression.\n"
            "- **IRF**: impulse response function (shock propagation over time).\n"
        )

        checklist_items = [
            *base_steps,
            "Plot series in levels before running tests.",
            "Justify each transformation (diff/logdiff) in words.",
            "State what your IRF identification assumes (ordering or structure).",
        ]

        alt_example = (
            "```python\n"
            "# Random walk vs stationary series (ADF intuition):\n"
            "import numpy as np\n"
            "from statsmodels.tsa.stattools import adfuller\n"
            "\n"
            "rng = np.random.default_rng(0)\n"
            "rw = rng.normal(size=400).cumsum()\n"
            "st = rng.normal(size=400)\n"
            "\n"
            "adf_rw_p = adfuller(rw)[1]\n"
            "adf_st_p = adfuller(st)[1]\n"
            "adf_rw_p, adf_st_p\n"
            "```\n"
        )

        technical = concept("stationarity_unit_roots")
        if stem == "00_stationarity_unit_roots":
            technical = concept("stationarity_unit_roots")
        elif stem == "01_cointegration_error_correction":
            technical = concept("stationarity_unit_roots") + "\n\n" + concept("cointegration_ecm")
        elif stem == "02_var_impulse_responses":
            technical = concept("stationarity_unit_roots") + "\n\n" + concept("var_irf")

        mistakes_items = [
            "Running levels-on-levels regressions without checking stationarity (spurious regression).",
            "Interpreting Granger causality as structural causality.",
            "Choosing VAR lags mechanically without sanity checks.",
            "For IRFs: forgetting that orthogonalized IRFs depend on variable ordering.",
        ]

        summary = (
            "You now have a classical macro time-series toolkit that complements the ML workflow in this repo.\n"
            "Use it to avoid spurious inference and to reason about dynamics.\n"
        )

        readings_items = [
            "Hamilton: Time Series Analysis (classic reference)",
            "Hyndman & Athanasopoulos: Forecasting: Principles and Practice (applied)",
            "Stock & Watson: Introduction to Econometrics (time-series chapters)",
        ]

    else:
        intro = header
        checklist_items = [*base_steps, "Complete the notebook TODOs."]
        alt_example = "```python\n# See notebook for context.\n```\n"
        technical = "Keep the workflow reproducible and time-aware."
        mistakes_items = ["Avoid leakage."]
        summary = "Complete the notebook and write down what you learned."
        readings_items = ["pandas, statsmodels, scikit-learn documentation"]

    checklist = "\n".join([f"- {x}" for x in checklist_items])
    mistakes = "\n".join([f"- {x}" for x in mistakes_items])
    readings = "\n".join([f"- {x}" for x in readings_items])
    code_map = "\n".join([f"- {x}" for x in guide_code_map(category, stem)])

    base = load_template("guides/base.md.tmpl")
    out_path.write_text(
        render_template(
            base,
            {
                "STEM": stem,
                "INTRO": intro,
                "CHECKLIST": checklist,
                "ALT_EXAMPLE": alt_example,
                "TECHNICAL": technical,
                "CODE_MAP": code_map,
                "MISTAKES": mistakes,
                "SUMMARY": summary,
                "READINGS": readings,
            },
        ).rstrip()
        + "\n"
    )


def write_guides_index(specs: list[NotebookSpec], root: Path) -> None:
    """Write docs/guides/index.md listing guides by category."""

    by_category: dict[str, list[NotebookSpec]] = {}
    for spec in specs:
        nb_path = Path(spec.path)
        category = nb_path.parts[1] if len(nb_path.parts) > 1 else "misc"
        by_category.setdefault(category, []).append(spec)

    lines: list[str] = []
    lines.append("# Guides Index")
    lines.append("")
    lines.append("Guides are grouped to mirror `notebooks/` folders.")
    lines.append("")
    lines.append("Use `docs/index.md` for the recommended learning path.")
    lines.append("")

    for category in sorted(by_category.keys()):
        lines.append(f"## {category}")
        lines.append("")
        lines.append(f"- [Part overview]({category}/index.md)")
        lines.append("")
        for spec in sorted(by_category[category], key=lambda s: s.path):
            stem = Path(spec.path).stem
            lines.append(f"- [{stem}]({category}/{stem}.md)")
        lines.append("")

    out_path = root / "docs" / "guides" / "index.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n")


def write_guide_part_indexes(specs: list[NotebookSpec], root: Path) -> None:
    """Write docs/guides/<category>/index.md for each part."""

    by_category: dict[str, list[NotebookSpec]] = {}
    for spec in specs:
        nb_path = Path(spec.path)
        category = nb_path.parts[1] if len(nb_path.parts) > 1 else "misc"
        by_category.setdefault(category, []).append(spec)

    for category, items in by_category.items():
        tmpl_rel = f"guides/parts/{category}_index.md.tmpl"
        tmpl_path = TEMPLATES_ROOT / tmpl_rel
        if not tmpl_path.exists():
            # Skip unknown categories (should not happen in this curriculum).
            continue

        chapter_lines: list[str] = []
        for spec in sorted(items, key=lambda s: s.path):
            stem = Path(spec.path).stem
            nb_rel = f"../../../{spec.path}"
            chapter_lines.append(f"- [{stem}]({stem}.md) — Notebook: [{stem}.ipynb]({nb_rel})")

        rendered = render_template(load_template(tmpl_rel), {"CHAPTERS": "\n".join(chapter_lines)})
        out_path = root / "docs" / "guides" / category / "index.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered.rstrip() + "\n")


def write_docs_index(specs: list[NotebookSpec], root: Path) -> None:
    """Write docs/index.md as the main navigation hub (generator-of-record)."""

    order = [
        "00_foundations",
        "01_data",
        "02_regression",
        "07_causal",
        "08_time_series_econ",
        "03_classification",
        "04_unsupervised",
        "05_model_ops",
        "06_capstone",
    ]

    by_category: dict[str, list[NotebookSpec]] = {}
    for spec in specs:
        nb_path = Path(spec.path)
        category = nb_path.parts[1] if len(nb_path.parts) > 1 else "misc"
        by_category.setdefault(category, []).append(spec)

    def section_title(cat: str) -> str:
        titles = {
            "00_foundations": "Foundations",
            "01_data": "Data (Macro + Micro)",
            "02_regression": "Regression (Micro then Macro)",
            "07_causal": "Causal Inference (Panels + Quasi-Experiments)",
            "08_time_series_econ": "Time-Series Econometrics (Unit Roots \u2192 VAR)",
            "03_classification": "Classification (Technical Recession)",
            "04_unsupervised": "Unsupervised (Macro Structure)",
            "05_model_ops": "Model Ops",
            "06_capstone": "Capstone",
        }
        return titles.get(cat, cat)

    lines: list[str] = []
    lines.append("# Curriculum Index")
    lines.append("")
    lines.append("This is the navigation hub for the full tutorial project.")
    lines.append("")
    lines.append("## How To Use This Repo")
    lines.append("- Work through the notebooks in order (recommended path below).")
    lines.append("- Each notebook has a matching deep guide under `docs/guides/<category>/`.")
    lines.append("- Notebooks are hands-on: code cells are intentionally incomplete (TODO-driven).")
    lines.append("- Each notebook ends with a collapsed **Solutions (Reference)** section to self-check.")
    lines.append("")
    lines.append("## Recommended Learning Path")
    lines.append("")

    for cat in order:
        specs_in_cat = sorted(by_category.get(cat, []), key=lambda s: s.path)
        if not specs_in_cat:
            continue

        lines.append(f"### {section_title(cat)}")
        for i, spec in enumerate(specs_in_cat, start=1):
            stem = Path(spec.path).stem
            nb_rel = f"../{spec.path}"
            guide_rel = f"guides/{cat}/{stem}.md"
            lines.append(f"{i}. [Notebook: {stem}]({nb_rel})  ")
            lines.append(f"   Guide: [{stem}]({guide_rel})")
        lines.append("")

    lines.append("## Extra References")
    lines.append("- [Guides Index](guides/index.md)")
    lines.append("- [Monolithic Deep Dive (Optional)](technical_deep_dive.md)")
    lines.append("")

    out_path = root / "docs" / "index.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    specs = [
        NotebookSpec(
            path="notebooks/00_foundations/00_setup.ipynb",
            title="00 Setup",
            summary="Configure your environment, understand the repo layout, and load sample data.",
            sections=[
                "Environment bootstrap",
                "Verify API keys",
                "Load sample data",
                "Checkpoints",
            ],
        ),
        NotebookSpec(
            path="notebooks/00_foundations/01_time_series_basics.ipynb",
            title="01 Time Series Basics",
            summary="Resampling, lags, rolling windows, and leakage intuition.",
            sections=[
                "Toy series",
                "Resampling",
                "Lag and rolling features",
                "Leakage demo",
            ],
        ),
        NotebookSpec(
            path="notebooks/00_foundations/02_stats_basics_for_ml.ipynb",
            title="02 Stats Basics for ML",
            summary="Correlation, collinearity, bias/variance, and overfitting basics.",
            sections=[
                "Correlation vs causation",
                "Multicollinearity (VIF)",
                "Bias/variance",
                "Hypothesis testing",
            ],
        ),
        NotebookSpec(
            path="notebooks/01_data/00_fred_api_and_caching.ipynb",
            title="00 FRED API and Caching",
            summary="Fetch indicators from FRED and cache raw JSON.",
            sections=[
                "Choose series",
                "Fetch metadata",
                "Fetch + cache observations",
                "Fallback to sample",
            ],
        ),
        NotebookSpec(
            path="notebooks/01_data/01_build_macro_monthly_panel.ipynb",
            title="01 Build Macro Monthly Panel",
            summary="Align mixed-frequency series to a month-end panel.",
            sections=[
                "Load series",
                "Month-end alignment",
                "Missingness",
                "Save processed panel",
            ],
        ),
        NotebookSpec(
            path="notebooks/01_data/02_gdp_growth_and_recession_label.ipynb",
            title="02 GDP Growth and Recession Label",
            summary="Compute GDP growth variants and build a technical recession label.",
            sections=[
                "Fetch GDP",
                "Compute growth",
                "Define recession label",
                "Define next-quarter target",
            ],
        ),
        NotebookSpec(
            path="notebooks/01_data/03_build_macro_quarterly_features.ipynb",
            title="03 Build Macro Quarterly Features",
            summary="Aggregate monthly predictors to quarterly, add lags, and merge with targets.",
            sections=[
                "Aggregate monthly -> quarterly",
                "Add lags",
                "Merge with GDP/labels",
                "Save macro_quarterly.csv",
            ],
        ),
        NotebookSpec(
            path="notebooks/01_data/04_census_api_microdata_fetch.ipynb",
            title="04 Census API Microdata Fetch",
            summary="Fetch county-level ACS data and build a micro dataset.",
            sections=[
                "Browse variables",
                "Fetch county data",
                "Derived rates",
                "Save processed data",
            ],
        ),
        NotebookSpec(
            path="notebooks/02_regression/00_single_factor_regression_micro.ipynb",
            title="00 Single-Factor Regression (Micro)",
            summary="Single-factor log-log regression on county data; interpret coefficients.",
            sections=[
                "Load census data",
                "Build log variables",
                "Fit OLS + HC3",
                "Interpretation",
            ],
        ),
        NotebookSpec(
            path="notebooks/02_regression/01_multifactor_regression_micro_controls.ipynb",
            title="01 Multi-Factor Regression with Controls (Micro)",
            summary="Add controls, discuss omitted variable bias, robust SE.",
            sections=[
                "Choose controls",
                "Fit model",
                "Compare coefficients",
            ],
        ),
        NotebookSpec(
            path="notebooks/02_regression/02_single_factor_regression_macro.ipynb",
            title="02 Single-Factor Regression (Macro)",
            summary="GDP growth vs yield curve spread with time-series inference.",
            sections=[
                "Load macro data",
                "Fit OLS",
                "Fit HAC",
                "Interpretation",
            ],
        ),
        NotebookSpec(
            path="notebooks/02_regression/03_multifactor_regression_macro.ipynb",
            title="03 Multi-Factor Regression (Macro)",
            summary="Multi-factor GDP growth regression; weights and VIF.",
            sections=[
                "Choose features",
                "Fit model",
                "VIF + stability",
            ],
        ),
        NotebookSpec(
            path="notebooks/02_regression/04_inference_time_series_hac.ipynb",
            title="04 Inference for Time Series (HAC/Newey-West)",
            summary="Why naive SE break; HAC SE and interpretation.",
            sections=[
                "Assumptions",
                "Autocorrelation",
                "HAC SE",
            ],
        ),
        NotebookSpec(
            path="notebooks/02_regression/05_regularization_ridge_lasso.ipynb",
            title="05 Regularization: Ridge and Lasso",
            summary="Shrinkage, coefficient paths, feature selection.",
            sections=[
                "Build feature matrix",
                "Fit ridge/lasso",
                "Coefficient paths",
            ],
        ),
        NotebookSpec(
            path="notebooks/02_regression/06_rolling_regressions_stability.ipynb",
            title="06 Rolling Regressions and Stability",
            summary="Rolling windows to see regime changes.",
            sections=[
                "Rolling regression",
                "Coefficient drift",
                "Regime interpretation",
            ],
        ),
        NotebookSpec(
            path="notebooks/07_causal/00_build_census_county_panel.ipynb",
            title="00 Build Census County Panel",
            summary="Build a county-year ACS panel for panel/DiD methods.",
            sections=[
                "Choose years + variables",
                "Fetch/cache ACS tables",
                "Build panel + FIPS",
                "Save processed panel",
            ],
        ),
        NotebookSpec(
            path="notebooks/07_causal/01_panel_fixed_effects_clustered_se.ipynb",
            title="01 Panel Fixed Effects + Clustered SE",
            summary="Pooled vs two-way fixed effects and clustered standard errors.",
            sections=[
                "Load panel and define variables",
                "Pooled OLS baseline",
                "Two-way fixed effects",
                "Clustered standard errors",
            ],
        ),
        NotebookSpec(
            path="notebooks/07_causal/02_difference_in_differences_event_study.ipynb",
            title="02 Difference-in-Differences + Event Study",
            summary="TWFE DiD and event studies with synthetic adoption and diagnostics.",
            sections=[
                "Synthetic adoption + treatment",
                "TWFE DiD",
                "Event study (leads/lags)",
                "Diagnostics: pre-trends + placebo",
            ],
        ),
        NotebookSpec(
            path="notebooks/07_causal/03_instrumental_variables_2sls.ipynb",
            title="03 Instrumental Variables (2SLS)",
            summary="Endogeneity, instruments, and two-stage least squares (2SLS).",
            sections=[
                "Simulate endogeneity",
                "OLS vs 2SLS",
                "First-stage + weak IV checks",
                "Interpretation + limitations",
            ],
        ),
        NotebookSpec(
            path="notebooks/08_time_series_econ/00_stationarity_unit_roots.ipynb",
            title="00 Stationarity and Unit Roots",
            summary="ADF/KPSS, differencing, and spurious regression intuition.",
            sections=[
                "Load macro series",
                "Transformations",
                "ADF/KPSS tests",
                "Spurious regression demo",
            ],
        ),
        NotebookSpec(
            path="notebooks/08_time_series_econ/01_cointegration_error_correction.ipynb",
            title="01 Cointegration and Error Correction",
            summary="Engle-Granger cointegration and error correction models (ECM).",
            sections=[
                "Construct cointegrated pair",
                "Engle-Granger test",
                "Error correction model",
                "Interpretation",
            ],
        ),
        NotebookSpec(
            path="notebooks/08_time_series_econ/02_var_impulse_responses.ipynb",
            title="02 VAR and Impulse Responses",
            summary="Fit VARs, test Granger causality, and interpret IRFs.",
            sections=[
                "Build stationary dataset",
                "Fit VAR + choose lags",
                "Granger causality",
                "IRFs + forecasting",
            ],
        ),
        NotebookSpec(
            path="notebooks/03_classification/00_recession_classifier_baselines.ipynb",
            title="00 Recession Classifier Baselines",
            summary="Baselines and class imbalance for next-quarter recession prediction.",
            sections=[
                "Load data",
                "Define baselines",
                "Evaluate metrics",
            ],
        ),
        NotebookSpec(
            path="notebooks/03_classification/01_logistic_recession_classifier.ipynb",
            title="01 Logistic Recession Classifier",
            summary="Logistic regression, ROC/PR, and threshold tuning.",
            sections=[
                "Train/test split",
                "Fit logistic",
                "ROC/PR",
                "Threshold tuning",
            ],
        ),
        NotebookSpec(
            path="notebooks/03_classification/02_calibration_and_costs.ipynb",
            title="02 Calibration and Decision Costs",
            summary="Calibration curves, Brier score, and cost-based thresholds.",
            sections=[
                "Calibration",
                "Brier score",
                "Decision costs",
            ],
        ),
        NotebookSpec(
            path="notebooks/03_classification/03_tree_models_and_importance.ipynb",
            title="03 Tree Models and Feature Importance",
            summary="Tree classifiers and feature importance vs permutation.",
            sections=[
                "Fit tree model",
                "Compare metrics",
                "Interpret importance",
            ],
        ),
        NotebookSpec(
            path="notebooks/03_classification/04_walk_forward_validation.ipynb",
            title="04 Walk-Forward Validation",
            summary="Stability of recession prediction across time.",
            sections=[
                "Walk-forward splits",
                "Metric stability",
                "Failure analysis",
            ],
        ),
        NotebookSpec(
            path="notebooks/04_unsupervised/01_pca_macro_factors.ipynb",
            title="01 PCA Macro Factors",
            summary="Extract factors and interpret loadings.",
            sections=[
                "Standardize",
                "Fit PCA",
                "Interpret loadings",
            ],
        ),
        NotebookSpec(
            path="notebooks/04_unsupervised/02_clustering_macro_regimes.ipynb",
            title="02 Clustering Macro Regimes",
            summary="Cluster regimes and relate them to recessions.",
            sections=[
                "Clustering",
                "Choose k",
                "Relate to recessions",
            ],
        ),
        NotebookSpec(
            path="notebooks/04_unsupervised/03_anomaly_detection.ipynb",
            title="03 Anomaly Detection",
            summary="Detect macro anomalies and interpret crisis periods.",
            sections=[
                "Fit detector",
                "Inspect anomalies",
                "Compare to recessions",
            ],
        ),
        NotebookSpec(
            path="notebooks/05_model_ops/01_reproducible_pipeline_design.ipynb",
            title="01 Reproducible Pipeline Design",
            summary="Configs, run IDs, dataset hashes, and artifact layout.",
            sections=[
                "Configs",
                "Outputs",
                "Reproducibility",
            ],
        ),
        NotebookSpec(
            path="notebooks/05_model_ops/02_build_cli_train_predict.ipynb",
            title="02 Build CLI Train/Predict",
            summary="Extend the CLI to control features/models and generate artifacts.",
            sections=[
                "Training CLI",
                "Prediction CLI",
                "Artifacts",
            ],
        ),
        NotebookSpec(
            path="notebooks/05_model_ops/03_model_cards_and_reporting.ipynb",
            title="03 Model Cards and Reporting",
            summary="Document your model: intended use, risks, limitations, monitoring.",
            sections=[
                "Model card",
                "Reporting",
                "Limitations",
            ],
        ),
        NotebookSpec(
            path="notebooks/06_capstone/00_capstone_brief.ipynb",
            title="00 Capstone Brief",
            summary="Project brief + rubric + deliverables.",
            sections=[
                "Deliverables",
                "Rubric",
                "Scope selection",
            ],
        ),
        NotebookSpec(
            path="notebooks/06_capstone/01_capstone_workspace.ipynb",
            title="01 Capstone Workspace",
            summary="Your end-to-end build space: model, evaluation, report, dashboard.",
            sections=[
                "Data",
                "Modeling",
                "Interpretation",
                "Artifacts",
            ],
        ),
    ]

    for spec in specs:
        write_notebook(spec, root)
        write_guide(spec, root)

    write_guide_part_indexes(specs, root)
    write_guides_index(specs, root)
    write_docs_index(specs, root)


if __name__ == "__main__":
    main()
