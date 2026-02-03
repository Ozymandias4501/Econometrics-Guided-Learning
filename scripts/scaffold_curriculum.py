"""Scaffold curriculum notebooks and per-notebook guides.

This script writes the tutorial structure described in docs/index.md:
- 28 notebooks under notebooks/
- 28 guides under docs/guides/

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
    success: list[str] = [
        "You can explain what you built and why each step exists.",
        "You can run your work end-to-end without undefined variables.",
    ]
    pitfalls: list[str] = [
        "Running cells top-to-bottom without reading the instructions.",
        "Leaving `...` placeholders in code cells.",
    ]

    why = "This notebook builds a core piece of the project."

    if category == "00_foundations":
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
        why = (
            "Classification notebooks turn the recession label into a **probability model**.\n"
            "You will learn how to evaluate rare-event prediction and how to choose thresholds intentionally.\n"
        )
        pitfalls += [
            "Reporting only accuracy on imbalanced data.",
            "Using threshold=0.5 by default without considering costs.",
        ]

    if category == "04_unsupervised":
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
        why = (
            "Model ops notebooks turn your work into reproducible runs with saved artifacts.\n"
            "The goal is: someone else can run your pipeline and see the same metrics.\n"
        )
        pitfalls += [
            "Not recording which dataset/config a model was trained on.",
            "Overwriting artifacts without run IDs.",
        ]

    if category == "06_capstone":
        why = (
            "Capstone notebooks integrate everything:\n"
            "- data pipeline,\n"
            "- modeling + evaluation,\n"
            "- interpretation + limitations,\n"
            "- reproducible artifacts,\n"
            "- report + dashboard.\n"
        )

    # Concrete deliverables (where applicable)
    if stem == "01_build_macro_monthly_panel":
        deliverables = ["data/processed/panel_monthly.csv"]
    elif stem == "02_gdp_growth_and_recession_label":
        deliverables = ["data/processed/gdp_quarterly.csv"]
    elif stem == "03_build_macro_quarterly_features":
        deliverables = ["data/processed/macro_quarterly.csv"]
    elif stem == "04_census_api_microdata_fetch":
        deliverables = ["data/processed/census_county_<year>.csv"]
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
        "## What You Will Produce\n"
        f"{deliverable_text}\n"
        "\n"
        "## Success Criteria\n"
        f"{bullets(success)}\n"
        "\n"
        "## Common Pitfalls\n"
        f"{bullets(pitfalls)}\n"
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
                "z = rng.normal(size=n)\n"
                "x = z + rng.normal(scale=0.8, size=n)\n"
                "w = z + rng.normal(scale=0.8, size=n)\n"
                "y = 2.0*z + rng.normal(scale=1.0, size=n)\n"
                "df = pd.DataFrame({'x': x, 'w': w, 'y': y})\n"
                "df.corr()\n"
            ),
            "Multicollinearity (VIF)": (
                "import statsmodels.api as sm\n"
                "from src.econometrics import vif_table\n"
                "\n"
                "# Make x2 highly correlated with x\n"
                "df['x2'] = df['x'] * 0.95 + np.random.default_rng(1).normal(scale=0.2, size=len(df))\n"
                "vif_table(df, ['x', 'x2'])\n"
                "\n"
                "X = sm.add_constant(df[['x', 'x2']])\n"
                "res = sm.OLS(df['y'], X).fit()\n"
                "res.summary()\n"
            ),
            "Bias/variance": (
                "from sklearn.linear_model import LinearRegression\n"
                "from sklearn.tree import DecisionTreeRegressor\n"
                "from sklearn.metrics import mean_squared_error\n"
                "\n"
                "X = df[['x']].to_numpy()\n"
                "y = df['y'].to_numpy()\n"
                "split = int(len(df) * 0.8)\n"
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
                    "```python",
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
            "- This notebook is hands-on. Most code cells are incomplete on purpose.\n"
            "- Complete each TODO, then run the cell.\n"
            f"- Use the matching guide (`{guide_path}`) for deep explanations and alternative examples.\n"
            "- Write short interpretation notes as you go (what changed, why it matters).\n"
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
            md("## Concept\nYou will build intuition for resampling, lags, rolling windows, and leakage."),
            md("## Your Turn: Build a Toy Time Series"),
            code(
                "import numpy as np\nimport pandas as pd\n\n# TODO: Create a daily toy series with a trend + noise\n# Hint: pd.date_range(..., freq='D')\n..."
            ),
            md("## Your Turn: Resample"),
            code(
                "# TODO: Resample to month-end and compute monthly mean and last\n..."
            ),
            md("## Your Turn: Lags + Rolling"),
            code(
                "# TODO: Create lag-1 and lag-7 features\n# TODO: Create a 14-day rolling mean\n..."
            ),
            md("## Leakage Demo"),
            code(
                "# TODO: Create a target that is tomorrow's value\n# TODO: Compare a random split vs time split\n# Write down what changes and why\n..."
            ),
        ]

    if spec.path.endswith("02_stats_basics_for_ml.ipynb"):
        cells += [
            md("## Concept\nCore statistics ideas that show up constantly in ML: correlation, collinearity, bias/variance, and overfitting."),
            md("## Your Turn: Correlation vs Causation"),
            code(
                "import numpy as np\nimport pandas as pd\n\n# TODO: Simulate two correlated variables and a target\n# Then compute correlations and explain why correlation != causation\n..."
            ),
            md("## Your Turn: Multicollinearity (VIF)"),
            code(
                "# TODO: Create two nearly identical features\n# Compute VIF using src.econometrics.vif_table\n..."
            ),
            md("## Your Turn: Bias/Variance"),
            code(
                "# TODO: Fit a simple model vs a more flexible model on synthetic data\n# Compare train vs test performance\n..."
            ),
        ]

    # Data notebooks
    if spec.path.endswith("00_fred_api_and_caching.ipynb"):
        cells += [
            md("## Goal\nFetch a few FRED series, cache raw JSON, and convert observations into tidy DataFrames."),
            md("## Your Turn: Choose Series"),
            code(
                "# TODO: Define a list of FRED series IDs\nseries_ids = [...]\n\n# Suggested: UNRATE, FEDFUNDS, CPIAUCSL, INDPRO, RSAFS, T10Y2Y\n"
            ),
            md("## Your Turn: Fetch Metadata"),
            code(
                "from src import fred_api\n\n# TODO: Pick one series and fetch metadata with fred_api.fetch_series_meta\n# Print title, units, frequency, seasonal_adjustment\n..."
            ),
            md("## Your Turn: Fetch + Cache Observations"),
            code(
                "from src import data as data_utils\n\n# TODO: For each series_id, cache JSON under data/raw/fred/<id>.json\n# Hint: data_utils.load_or_fetch_json + fred_api.fetch_series_observations\n..."
            ),
            md("## Fallback"),
            code(
                "# TODO: If you do not have FRED_API_KEY set, load data/sample/panel_monthly_sample.csv\n..."
            ),
        ]

    if spec.path.endswith("01_build_macro_monthly_panel.ipynb"):
        cells += [
            md("## Goal\nBuild a clean month-end panel of predictors (mixed daily/monthly series)."),
            md("## Your Turn: Load Raw Series"),
            code(
                "import pandas as pd\nfrom src import data as data_utils\n\n# TODO: Load cached JSON series into individual DataFrames\n# Convert to numeric and set DatetimeIndex\n..."
            ),
            md("## Your Turn: Align to Month-End"),
            code(
                "# TODO: resample to month-end (ME) and forward-fill\n# Save to data/processed/panel_monthly.csv\n..."
            ),
            md("## Checkpoint"),
            code(
                "# TODO: Assert no missing values after forward-fill (or explain remaining NaNs)\n..."
            ),
        ]

    if spec.path.endswith("02_gdp_growth_and_recession_label.ipynb"):
        cells += [
            md("## Goal\nCompute GDP growth (QoQ, annualized, YoY) and define a technical recession label from GDP growth."),
            md("## Your Turn: Load GDP"),
            code(
                "# TODO: Fetch GDPC1 from FRED and build a quarterly DataFrame\n# Hint: convert dates to quarter-end timestamps\n..."
            ),
            md("## Your Turn: Compute Growth Variants"),
            code(
                "from src import macro\n\n# TODO: Compute qoq, qoq_annualized, yoy growth\n..."
            ),
            md("## Your Turn: Technical Recession Label"),
            code(
                "# TODO: Implement recession_t = 1 if growth_t < 0 and growth_{t-1} < 0\n# Then define target_recession_next_q = recession.shift(-1)\n..."
            ),
            md("## Reflection"),
            md("- Why is this only a proxy for 'recession'?\n- How could this differ from an official recession indicator?"),
        ]

    if spec.path.endswith("03_build_macro_quarterly_features.ipynb"):
        cells += [
            md("## Goal\nAggregate monthly predictors to quarterly features (mean vs last), add lags, and merge with GDP + targets."),
            md("## Your Turn: Load panel_monthly + gdp_quarterly"),
            code(
                "from src import data as data_utils\n\n# TODO: Load panel_monthly.csv and gdp_quarterly.csv\n..."
            ),
            md("## Your Turn: Quarterly Aggregation"),
            code(
                "from src import macro\n\n# TODO: Create quarterly features using BOTH methods: mean and last\n# Compare them visually/statistically\n..."
            ),
            md("## Your Turn: Add Lags"),
            code(
                "# TODO: Add quarterly lags (1, 2, 4) for each predictor\n..."
            ),
            md("## Your Turn: Merge and Save"),
            code(
                "# TODO: Merge predictors with GDP growth + recession labels\n# Save to data/processed/macro_quarterly.csv\n..."
            ),
        ]

    if spec.path.endswith("04_census_api_microdata_fetch.ipynb"):
        cells += [
            md("## Goal\nFetch ACS county-level data and build a micro dataset for regression/classification."),
            md("## Your Turn: Browse variables.json"),
            code(
                "import json\n\n# TODO: Load variables metadata (either from API or data/raw/census/variables_<year>.json)\n# Find variables related to income, rent, poverty, labor force\n..."
            ),
            md("## Your Turn: Fetch County Data"),
            code(
                "from src import census_api\n\n# TODO: Fetch ACS data at county level and save to data/processed/census_county_<year>.csv\n# Hint: use for_geo='county:*' and in_geo='state:*'\n..."
            ),
            md("## Fallback"),
            code(
                "# TODO: If API is unavailable, load data/sample/census_county_sample.csv\n..."
            ),
        ]

    # Regression notebooks
    if spec.path.endswith("00_single_factor_regression_micro.ipynb"):
        cells += [
            md("## Goal\nFit a single-factor regression on cross-sectional (county) data and interpret the coefficient like an elasticity."),
            md("## Your Turn: Load Census Data"),
            code(
                "import numpy as np\nimport pandas as pd\n\nfrom src import data as data_utils\n\n# TODO: Load census_county_<year>.csv OR the sample\n..."
            ),
            md("## Your Turn: Build log-log variables"),
            code(
                "# TODO: Create y = log(rent) and x = log(income)\n# Handle missing/zero values carefully\n..."
            ),
            md("## Your Turn: Fit OLS + Robust SE (HC3)"),
            code(
                "from src import econometrics\n\n# TODO: Fit OLS with HC3 robust SE\n# Print summary and interpret the slope\n..."
            ),
            md("## Reflection"),
            md("- What does the coefficient mean in log-log form?\n- What assumptions would make it 'causal'?"),
        ]

    if spec.path.endswith("01_multifactor_regression_micro_controls.ipynb"):
        cells += [
            md("## Goal\nFit a multi-factor regression with controls and discuss omitted variable bias."),
            md("## Your Turn: Choose controls"),
            code(
                "# TODO: Choose at least 2 controls (example: log(population), poverty_rate)\n# Compare coefficient on log(income) with vs without controls\n..."
            ),
            md("## Your Turn: Robust SE + Optional Clustering"),
            code(
                "# TODO: Fit HC3 robust SE\n# Advanced: cluster by state (research statsmodels cov_type='cluster')\n..."
            ),
        ]

    if spec.path.endswith("02_single_factor_regression_macro.ipynb"):
        cells += [
            md("## Goal\nSingle-factor macro regression: GDP growth vs yield curve spread."),
            md("## Your Turn: Load macro_quarterly.csv"),
            code(
                "from src import data as data_utils\n\n# TODO: Load macro_quarterly.csv (or sample)\n..."
            ),
            md("## Your Turn: Fit OLS and then HAC"),
            code(
                "from src import econometrics\n\n# TODO: Fit plain OLS and compare to HAC robust SE\n# Pick maxlags (try 1, 2, 4) and compare\n..."
            ),
            md("## Interpretation"),
            md("- Does the sign match your expectation?\n- How sensitive is the result to HAC maxlags?"),
        ]

    if spec.path.endswith("03_multifactor_regression_macro.ipynb"):
        cells += [
            md("## Goal\nMulti-factor regression: GDP growth vs multiple indicators; inspect weights and multicollinearity."),
            md("## Your Turn: Choose feature set"),
            code(
                "# TODO: Choose a set of predictors (levels + lags)\n# Consider standardizing them and comparing standardized coefficients\n..."
            ),
            md("## Your Turn: VIF"),
            code(
                "from src import econometrics\n\n# TODO: Compute VIF for your chosen predictors\n# What does high VIF imply for coefficient stability?\n..."
            ),
        ]

    if spec.path.endswith("04_inference_time_series_hac.ipynb"):
        cells += [
            md("## Goal\nUnderstand why standard errors break for time series, and use HAC/Newey-West."),
            md("## Your Turn: Residual autocorrelation"),
            code(
                "# TODO: Fit a simple macro regression\n# Plot residuals over time and compute their autocorrelation\n..."
            ),
            md("## Your Turn: Compare SE"),
            code(
                "# TODO: Compare naive OLS SE to HAC SE\n# Explain what changes and why\n..."
            ),
        ]

    if spec.path.endswith("05_regularization_ridge_lasso.ipynb"):
        cells += [
            md("## Goal\nUse ridge and lasso to handle correlated features and understand coefficient shrinkage."),
            md("## Your Turn: Build X/y"),
            code(
                "# TODO: Choose a regression target (GDP growth) and build a feature matrix\n# Split by time\n..."
            ),
            md("## Your Turn: Ridge vs Lasso"),
            code(
                "# TODO: Fit Ridge and Lasso across a range of alphas\n# Plot coefficient paths\n..."
            ),
        ]

    if spec.path.endswith("06_rolling_regressions_stability.ipynb"):
        cells += [
            md("## Goal\nRolling regressions: see how relationships change over time."),
            md("## Your Turn: Rolling window fit"),
            code(
                "# TODO: Fit a rolling-window regression of GDP growth on yield spread\n# Plot the coefficient over time\n..."
            ),
            md("## Reflection"),
            md("- When does the sign or magnitude change?\n- What macro regimes might explain it?"),
        ]

    # Classification notebooks
    if spec.path.endswith("00_recession_classifier_baselines.ipynb"):
        cells += [
            md("## Goal\nBuild baselines for predicting next-quarter technical recession."),
            md("## Your Turn: Load macro_quarterly.csv"),
            code(
                "from src import data as data_utils\n\n# TODO: Load macro_quarterly.csv (or sample)\n..."
            ),
            md("## Your Turn: Baselines"),
            code(
                "from src import evaluation\n\n# TODO: Compute baseline predictions (majority class, persistence, simple threshold)\n# Evaluate with ROC-AUC, PR-AUC, Brier\n..."
            ),
        ]

    if spec.path.endswith("01_logistic_recession_classifier.ipynb"):
        cells += [
            md("## Goal\nTrain a logistic regression classifier for next-quarter recession and interpret coefficients/odds."),
            md("## Your Turn: Fit model"),
            code(
                "# TODO: Build train/test split by time\n# Fit StandardScaler + LogisticRegression\n# Evaluate metrics and plot ROC/PR\n..."
            ),
            md("## Your Turn: Threshold tuning"),
            code(
                "# TODO: Evaluate at thresholds 0.3, 0.5, 0.7\n# Discuss precision/recall tradeoffs\n..."
            ),
        ]

    if spec.path.endswith("02_calibration_and_costs.ipynb"):
        cells += [
            md("## Goal\nCalibrate probabilities and choose thresholds based on decision costs."),
            md("## Your Turn: Calibration"),
            code(
                "# TODO: Compute calibration curve and Brier score\n# Compare calibrated vs uncalibrated probabilities\n..."
            ),
            md("## Your Turn: Cost-based threshold"),
            code(
                "# TODO: Define a simple cost matrix (FP vs FN)\n# Pick a threshold that minimizes expected cost\n..."
            ),
        ]

    if spec.path.endswith("03_tree_models_and_importance.ipynb"):
        cells += [
            md("## Goal\nCompare a tree-based classifier to logistic regression; interpret feature importance."),
            md("## Your Turn: Fit a tree model"),
            code(
                "# TODO: Fit RandomForestClassifier (or GradientBoostingClassifier)\n# Compare metrics to logistic regression\n..."
            ),
            md("## Your Turn: Importance"),
            code(
                "# TODO: Compare built-in feature importances vs permutation importance\n# Discuss why they can disagree\n..."
            ),
        ]

    if spec.path.endswith("04_walk_forward_validation.ipynb"):
        cells += [
            md("## Goal\nWalk-forward evaluation: measure stability across time."),
            md("## Your Turn: Implement walk-forward"),
            code(
                "from src import evaluation\n\n# TODO: Use evaluation.walk_forward_splits to evaluate across folds\n# Plot metrics over time and identify unstable periods\n..."
            ),
        ]

    # Unsupervised
    if spec.path.endswith("01_pca_macro_factors.ipynb"):
        cells += [
            md("## Goal\nUse PCA to extract macro factors and interpret loadings."),
            md("## Your Turn: Standardize data"),
            code(
                "# TODO: Load panel_monthly.csv (or sample)\n# Standardize features and run PCA\n..."
            ),
            md("## Your Turn: Interpret"),
            code(
                "# TODO: Inspect explained variance and loadings\n# Name the first 1-2 factors in economic terms\n..."
            ),
        ]

    if spec.path.endswith("02_clustering_macro_regimes.ipynb"):
        cells += [
            md("## Goal\nCluster macro regimes and relate them to technical recessions."),
            md("## Your Turn: Choose representation"),
            code(
                "# TODO: Cluster either PCA factors or standardized raw features\n# Try k=3..6 and interpret clusters\n..."
            ),
            md("## Your Turn: Link to recession"),
            code(
                "# TODO: Compare cluster assignments with recession labels\n# Are some clusters recession-heavy?\n..."
            ),
        ]

    if spec.path.endswith("03_anomaly_detection.ipynb"):
        cells += [
            md("## Goal\nDetect anomalies (crisis periods) and compare to recession labels."),
            md("## Your Turn: Fit anomaly detector"),
            code(
                "# TODO: Use IsolationForest or z-score rules on standardized macro data\n# Flag anomalies and plot over time\n..."
            ),
        ]

    # Model ops
    if spec.path.endswith("01_reproducible_pipeline_design.ipynb"):
        cells += [
            md("## Goal\nUnderstand run IDs, configs, and artifact layout in outputs/."),
            md("## Your Turn: Inspect scripts and configs"),
            code(
                "# TODO: Open configs/recession.yaml and scripts/train_recession.py\n# List what is configurable and what is hard-coded\n..."
            ),
            md("## Your Turn: Run a pipeline"),
            md("Run in terminal:\n- python scripts/build_datasets.py --recession-config configs/recession.yaml --census-config configs/census.yaml\n- python scripts/train_recession.py --config configs/recession.yaml\nThen inspect outputs/<run_id>"),
        ]

    if spec.path.endswith("02_build_cli_train_predict.ipynb"):
        cells += [
            md("## Goal\nBuild/extend a CLI that trains and predicts while saving artifacts."),
            md("## Your Turn: Extend the training script"),
            code(
                "# TODO: Add a CLI flag to include/exclude GDP-derived features\n# TODO: Add a CLI flag to choose logistic vs random_forest\n# Implement and re-run training\n..."
            ),
            md("## Your Turn: Predict script"),
            code(
                "# TODO: Modify scripts/predict_recession.py to accept a date filter\n# Example: only output the last N rows\n..."
            ),
        ]

    if spec.path.endswith("03_model_cards_and_reporting.ipynb"):
        cells += [
            md("## Goal\nWrite a model card and connect it to your run artifacts."),
            md("## Your Turn: Model card"),
            code(
                "# TODO: Copy reports/capstone_report.md template and fill it for one run\n# Focus on intended use, limitations, and monitoring ideas\n..."
            ),
        ]

    # Capstone
    if spec.path.endswith("00_capstone_brief.ipynb"):
        cells += [
            md("## Capstone Brief\nYou will produce a final model + report + dashboard."),
            md("### Deliverables\n- A reproducible run under outputs/\n- A written report in reports/capstone_report.md\n- A Streamlit dashboard that loads your artifacts"),
            md("### Rubric\n- Problem framing and label definition\n- Data pipeline correctness (no leakage)\n- Evaluation quality (time-aware)\n- Interpretation and limitations\n- Reproducibility"),
            md("## Your Turn: Pick scope"),
            md("Option A: Macro-only recession prediction\nOption B: Macro + Micro (add a cross-sectional module to your report)"),
        ]

    if spec.path.endswith("01_capstone_workspace.ipynb"):
        cells += [
            md("## Capstone Workspace\nThis is your working area. It should end with artifacts + report + dashboard."),
            md("## Your Turn: Load final dataset"),
            code(
                "# TODO: Load data/processed/macro_quarterly.csv\n# Choose your final feature set and target\n..."
            ),
            md("## Your Turn: Train final model"),
            code(
                "# TODO: Train at least 2 models and select one\n# Use time split and walk-forward\n..."
            ),
            md("## Your Turn: Interpret"),
            code(
                "# TODO: Provide coefficient/importance analysis and error analysis\n..."
            ),
            md("## Your Turn: Write artifacts"),
            code(
                "# TODO: Save model + metrics + predictions to outputs/ and update reports/capstone_report.md\n..."
            ),
            md("## Your Turn: Run dashboard"),
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
        "`src/data.py`: caching helpers and JSON load/save utilities",
        "`src/features.py`: feature engineering helpers (lags, pct changes, rolling features)",
    ]

    if category == "00_foundations":
        return [
            "`scripts/scaffold_curriculum.py`: how this curriculum is generated (for curiosity)",
            "`src/evaluation.py`: time splits and metrics used later",
            *common,
        ]

    if category == "01_data":
        out = [
            "`src/fred_api.py`: FRED client (metadata + observations)",
            "`src/census_api.py`: Census/ACS client",
            "`src/macro.py`: GDP growth and technical recession label helpers",
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
            "`src/econometrics.py`: OLS + robust SE (HC3/HAC) + VIF",
            "`src/macro.py`: GDP growth + label utilities (macro notebooks)",
            "`src/evaluation.py`: regression metrics helpers",
            *common,
        ]
        return out

    if category == "03_classification":
        return [
            "`src/evaluation.py`: classification metrics (ROC-AUC, PR-AUC, Brier)",
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
                concept_omitted_variable_bias_deep_dive()
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
            chapter_lines.append(f"- [{stem}]({stem}.md) (Notebook: {nb_rel})")

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
