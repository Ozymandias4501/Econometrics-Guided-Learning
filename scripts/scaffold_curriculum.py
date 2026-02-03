"""Scaffold curriculum notebooks and per-notebook guides.

This script writes the tutorial structure described in docs/index.md:
- 28 notebooks under notebooks/
- 28 guides under docs/guides/

It is safe to re-run; it will overwrite files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class NotebookSpec:
    path: str
    title: str
    summary: str
    sections: List[str]


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

    toc_lines = ["## Table of Contents", *[f"- {s}" for s in spec.sections]]

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
        md("## Environment Bootstrap\nRun this cell first. It makes the repo importable and defines common directories."),
        code(BOOTSTRAP),
    ]

    # Add notebook-specific scaffold blocks.
    # Keep these short; the detailed explanations live in docs/guides/.
    if spec.path.endswith("00_setup.ipynb"):
        cells += [
            md("## Your Turn: Verify Keys\nSet env vars in your shell, then confirm they are visible here."),
            code(
                "import os\n\n# TODO: Print whether FRED_API_KEY is set\n# TODO: Print whether CENSUS_API_KEY is set\n\n..."
            ),
            md("## Your Turn: Explore Sample Data\nLoad a sample dataset so you can work offline."),
            code(
                "import pandas as pd\n\n# TODO: Load data/sample/macro_quarterly_sample.csv\n# Hint: use index_col=0, parse_dates=True\n\ndf = ...\ndf.head()"
            ),
            md("## Checkpoint"),
            code(
                "# TODO: Assert the index is sorted and the dataset has a target column\n..."
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


GUIDE_TEMPLATE = """# Guide: {stem}

## Table of Contents
- [Intro of Concept Explained](#intro)
- [Step-by-Step and Alternative Examples](#step-by-step)
- [Technical Explanations (Code + Math + Interpretation)](#technical)
- [Summary + Suggested Readings](#summary)

<a id="intro"></a>
## Intro of Concept Explained

{intro}

<a id="step-by-step"></a>
## Step-by-Step and Alternative Examples

### What You Should Implement (Checklist)
{checklist}

### Alternative Example (Not the Notebook Solution)
{alt_example}

<a id="technical"></a>
## Technical Explanations (Code + Math + Interpretation)

{technical}

### Common Mistakes
{mistakes}

<a id="summary"></a>
## Summary + Suggested Readings

{summary}

Suggested readings:
{readings}
"""


def concept_leakage_deep_dive() -> str:
    return (
        "### Deep Dive: Leakage (What It Is, How It Happens, How To Detect It)\n"
        "\n"
        "**Leakage** means your model (or your features) accidentally uses information that would not be available at prediction time.\n"
        "In time series, leakage is especially easy because the index *looks* like just another column, but it encodes causality constraints.\n"
        "\n"
        "**Common leakage types**\n"
        "- **Target leakage**: a feature is derived from the target (directly or indirectly).\n"
        "- **Temporal leakage**: a feature uses future values (wrong shift direction, centered rolling windows, etc.).\n"
        "- **Split leakage**: random splits mix future and past, letting the model learn patterns it wouldn't have in production.\n"
        "- **Preprocessing leakage**: scaling/imputation fitted on all data (train + test) instead of train only.\n"
        "\n"
        "**How to spot leakage (symptoms)**\n"
        "- Test metrics that look \"too good to be true\" for the problem.\n"
        "- A single feature dominates and seems to predict perfectly.\n"
        "- Performance collapses when you switch from random split to time split.\n"
        "\n"
        "**Debug checklist (time-series)**\n"
        "1. For each feature, ask: *would I know this value at time t when making a prediction for t+1?*\n"
        "2. Verify every shift direction: `shift(+k)` uses the past; `shift(-k)` leaks future.\n"
        "3. Verify rolling window alignment: `rolling(..., center=True)` leaks future.\n"
        "4. Ensure preprocessing (scalers, imputers) are fitted on training data only.\n"
        "\n"
        "**Python demo: the classic `shift(-1)` leak**\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.metrics import r2_score\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "idx = pd.date_range('2020-01-01', periods=200, freq='D')\n"
        "y = pd.Series(np.cumsum(rng.normal(size=len(idx))), index=idx)\n"
        "\n"
        "# Predict tomorrow (t+1)\n"
        "target = y.shift(-1)\n"
        "\n"
        "# Legit feature: yesterday (t-1)\n"
        "x_lag1 = y.shift(1)\n"
        "\n"
        "# LEAK feature: tomorrow (t+1)\n"
        "x_leak = y.shift(-1)\n"
        "\n"
        "df = pd.DataFrame({'target': target, 'x_lag1': x_lag1, 'x_leak': x_leak}).dropna()\n"
        "\n"
        "X_ok = df[['x_lag1']].to_numpy()\n"
        "X_leak = df[['x_leak']].to_numpy()\n"
        "y_arr = df['target'].to_numpy()\n"
        "\n"
        "# Time split\n"
        "split = int(len(df) * 0.8)\n"
        "X_ok_tr, X_ok_te = X_ok[:split], X_ok[split:]\n"
        "X_leak_tr, X_leak_te = X_leak[:split], X_leak[split:]\n"
        "y_tr, y_te = y_arr[:split], y_arr[split:]\n"
        "\n"
        "m_ok = LinearRegression().fit(X_ok_tr, y_tr)\n"
        "m_leak = LinearRegression().fit(X_leak_tr, y_tr)\n"
        "\n"
        "print('R2 legit:', r2_score(y_te, m_ok.predict(X_ok_te)))\n"
        "print('R2 leak :', r2_score(y_te, m_leak.predict(X_leak_te)))\n"
        "```\n"
        "\n"
        "**Python demo: rolling-window leakage (centered windows)**\n"
        "```python\n"
        "import pandas as pd\n"
        "\n"
        "# This uses future values because the window is centered.\n"
        "feature_leaky = y.rolling(window=7, center=True).mean()\n"
        "```\n"
        "\n"
        "**Practical interpretation (economics)**\n"
        "- A recession model with leakage will appear to \"predict\" recessions, but it is usually just reading signals from the future.\n"
        "- The real goal is to predict with only information available *at the time*.\n"
    )
def concept_time_split_deep_dive() -> str:
    return (
        "### Deep Dive: Train/Test Splits for Time Series\n"
        "\n"
        "**Train/test split** means you fit your model on one subset (train) and evaluate on a later, untouched subset (test).\n"
        "In time series, the split must respect chronology.\n"
        "\n"
        "**Random split vs time split**\n"
        "- Random split: mixes past and future in both train and test.\n"
        "- Time split: train is earlier, test is later.\n"
        "\n"
        "**Why time split matters**\n"
        "- The future can look statistically different (regimes).\n"
        "- The model must operate in real time: train on the past, predict the future.\n"
        "\n"
        "**Python demo: random vs time split**\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.metrics import mean_squared_error\n"
        "from sklearn.linear_model import LinearRegression\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "idx = pd.date_range('2010-01-01', periods=400, freq='D')\n"
        "\n"
        "# A simple AR(1)-like process\n"
        "y = np.zeros(len(idx))\n"
        "for t in range(1, len(y)):\n"
        "    y[t] = 0.9*y[t-1] + rng.normal(scale=1.0)\n"
        "s = pd.Series(y, index=idx)\n"
        "\n"
        "df = pd.DataFrame({'y': s, 'y_lag1': s.shift(1)}).dropna()\n"
        "X = df[['y_lag1']].to_numpy()\n"
        "y_arr = df['y'].to_numpy()\n"
        "\n"
        "# Random split\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y_arr, test_size=0.2, shuffle=True, random_state=0)\n"
        "m = LinearRegression().fit(X_tr, y_tr)\n"
        "rmse_rand = mean_squared_error(y_te, m.predict(X_te), squared=False)\n"
        "\n"
        "# Time split\n"
        "split = int(len(df) * 0.8)\n"
        "X_tr2, X_te2 = X[:split], X[split:]\n"
        "y_tr2, y_te2 = y_arr[:split], y_arr[split:]\n"
        "m2 = LinearRegression().fit(X_tr2, y_tr2)\n"
        "rmse_time = mean_squared_error(y_te2, m2.predict(X_te2), squared=False)\n"
        "\n"
        "print('RMSE random:', rmse_rand)\n"
        "print('RMSE time  :', rmse_time)\n"
        "```\n"
        "\n"
        "**Walk-forward validation (preview)**\n"
        "- Instead of one split, you evaluate across multiple chronological folds.\n"
        "- This reveals stability: the model can look good in one era and fail in another.\n"
    )
def concept_multicollinearity_vif_deep_dive() -> str:
    return (
        "### Deep Dive: Multicollinearity (Why Coefficients Become Unstable)\n"
        "\n"
        "**Multicollinearity** means two or more predictors contain overlapping information (they are highly correlated).\n"
        "\n"
        "**What multicollinearity does (and does not do)**\n"
        "- It often does **not** hurt prediction much.\n"
        "- It **does** make individual coefficients unstable and their standard errors large.\n"
        "- It makes p-values fragile: you can flip signs or lose significance by adding/removing correlated features.\n"
        "\n"
        "**Variance Inflation Factor (VIF)**\n"
        "- VIF for feature j: how much the variance of β_j is inflated because x_j is correlated with other features.\n"
        "- Rule of thumb: VIF > 5 (or 10) suggests serious multicollinearity.\n"
        "\n"
        "**Python demo: correlated predictors -> unstable coefficients**\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import statsmodels.api as sm\n"
        "from src.econometrics import vif_table\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 500\n"
        "x1 = rng.normal(size=n)\n"
        "x2 = x1 * 0.95 + rng.normal(scale=0.2, size=n)  # highly correlated with x1\n"
        "y = 1.0 + 2.0*x1 + rng.normal(scale=1.0, size=n)\n"
        "\n"
        "df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})\n"
        "print(vif_table(df, ['x1', 'x2']))\n"
        "\n"
        "X = sm.add_constant(df[['x1', 'x2']])\n"
        "res = sm.OLS(df['y'], X).fit()\n"
        "print(res.params)\n"
        "print(res.bse)\n"
        "```\n"
        "\n"
        "**Mitigations (practical)**\n"
        "- Drop one of the correlated variables (choose based on interpretability).\n"
        "- Combine them (PCA/factors, or domain-driven composite indices).\n"
        "- Use regularization (ridge tends to stabilize coefficients).\n"
        "\n"
        "**Economic interpretation warning**\n"
        "- Macro indicators often move together.\n"
        "- If two indicators are highly correlated, \"holding one fixed\" is not an economically realistic counterfactual.\n"
        "- Treat coefficients as conditional correlations, not causal effects.\n"
    )
def concept_hac_newey_west_deep_dive() -> str:
    return (
        "### Deep Dive: HAC / Newey-West Standard Errors (Time-Series Inference)\n"
        "\n"
        "**Autocorrelation** means errors are correlated over time.\n"
        "**Heteroskedasticity** means the error variance changes over time.\n"
        "\n"
        "In time series, both are common. If you ignore them, your coefficient estimates can still be unbiased in some settings,\n"
        "but your **standard errors** (and therefore p-values/CI) can be wrong.\n"
        "\n"
        "**What HAC/Newey-West does**\n"
        "- Keeps the same OLS coefficients.\n"
        "- Adjusts the covariance estimate to be robust to autocorrelation and heteroskedasticity.\n"
        "\n"
        "**Python demo: AR(1) errors break naive SE**\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import statsmodels.api as sm\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 200\n"
        "x = rng.normal(size=n)\n"
        "\n"
        "# AR(1) errors\n"
        "eps = np.zeros(n)\n"
        "for t in range(1, n):\n"
        "    eps[t] = 0.8*eps[t-1] + rng.normal(scale=1.0)\n"
        "\n"
        "y = 1.0 + 0.5*x + eps\n"
        "\n"
        "X = sm.add_constant(pd.DataFrame({'x': x}))\n"
        "res = sm.OLS(y, X).fit()\n"
        "res_hac = res.get_robustcov_results(cov_type='HAC', cov_kwds={'maxlags': 4})\n"
        "\n"
        "print('naive SE:', res.bse)\n"
        "print('HAC SE  :', res_hac.bse)\n"
        "```\n"
        "\n"
        "**Choosing `maxlags`**\n"
        "- There is no perfect choice.\n"
        "- Try a small set (1, 2, 4 for quarterly) and see if inference is stable.\n"
        "- If your inference flips sign/significance with small changes, that is a warning sign.\n"
        "\n"
        "**What p-values mean (and don't)**\n"
        "- A small p-value is evidence against a zero effect *under the model assumptions*.\n"
        "- It is not proof of causality.\n"
        "- In macro time series, structural breaks and regime shifts can invalidate assumptions.\n"
    )
def concept_logistic_regression_odds_deep_dive() -> str:
    return (
        "### Deep Dive: Logistic Regression, Odds, and Interpreting Coefficients\n"
        "\n"
        "Logistic regression models:\n"
        "- score: `z = β0 + β1 x1 + ...`\n"
        "- probability: `p = 1 / (1 + exp(-z))`\n"
        "\n"
        "**Odds and log-odds**\n"
        "- odds = `p / (1-p)`\n"
        "- log-odds = `log(p/(1-p))`\n"
        "- Logistic regression is linear in log-odds.\n"
        "\n"
        "**Coefficient interpretation (key idea)**\n"
        "- A +1 increase in feature x_j adds β_j to log-odds.\n"
        "- `exp(β_j)` is the odds multiplier (holding other features fixed).\n"
        "\n"
        "**Python demo: odds multipliers**\n"
        "```python\n"
        "import numpy as np\n"
        "beta = 0.7\n"
        "print('odds multiplier:', np.exp(beta))\n"
        "```\n"
        "\n"
        "**Scaling matters**\n"
        "- If x is in large units, β will be small; if x is standardized, β is per 1 std dev.\n"
        "- For interpretation, standardize or be explicit about units.\n"
    )
def concept_calibration_brier_deep_dive() -> str:
    return (
        "### Deep Dive: Calibration and Brier Score (Probabilities You Can Trust)\n"
        "\n"
        "**Calibration** asks: when the model says 0.70, does the event happen ~70% of the time?\n"
        "\n"
        "**Brier score** is mean squared error of probabilities:\n"
        "- `mean((p - y)^2)` where y is 0/1.\n"
        "- Lower is better.\n"
        "\n"
        "**Python demo: calibration curve**\n"
        "```python\n"
        "import numpy as np\n"
        "from sklearn.calibration import calibration_curve\n"
        "\n"
        "# y_true: 0/1, y_prob: predicted probabilities\n"
        "# prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)\n"
        "```\n"
        "\n"
        "**Why this matters for recession prediction**\n"
        "- A probability model is only useful if you can make decisions from it.\n"
        "- Poor calibration means your 30% and 70% signals are not comparable.\n"
    )
def concept_walk_forward_validation_deep_dive() -> str:
    return (
        "### Deep Dive: Walk-Forward Validation (Stability Over Time)\n"
        "\n"
        "**Walk-forward validation** repeatedly trains on the past and tests on the next time block.\n"
        "It answers: \"does my model work across multiple eras, or only in one?\"\n"
        "\n"
        "**Why it's important in economics**\n"
        "- Relationships change (structural breaks).\n"
        "- Policy regimes shift.\n"
        "- A single split can hide fragility.\n"
        "\n"
        "**Pseudo-code**\n"
        "```python\n"
        "# for each fold:\n"
        "#   train = data[:t]\n"
        "#   test  = data[t:t+h]\n"
        "#   fit model\n"
        "#   compute metrics\n"
        "```\n"
        "\n"
        "**Interpretation**\n"
        "- If metrics vary widely across folds, your model is regime-sensitive.\n"
        "- This can be a reason to retrain more frequently or include regime features.\n"
    )
def concept_pca_loadings_deep_dive() -> str:
    return (
        "### Deep Dive: PCA and Loadings (Turning Many Indicators Into Factors)\n"
        "\n"
        "**PCA** finds orthogonal directions that explain maximum variance.\n"
        "\n"
        "**Loadings** tell you which original variables contribute to a component.\n"
        "A component can often be interpreted as a macro \"factor\" (e.g., growth, inflation, rates).\n"
        "\n"
        "**Critical step: standardization**\n"
        "- Without standardization, PCA mostly learns the biggest-unit variable.\n"
        "\n"
        "**Python demo**\n"
        "```python\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "\n"
        "# X_scaled = StandardScaler().fit_transform(X)\n"
        "# pca = PCA(n_components=3).fit(X_scaled)\n"
        "# loadings = pca.components_  # rows are components\n"
        "```\n"
        "\n"
        "**Interpretation playbook**\n"
        "- Look at the largest positive/negative loadings.\n"
        "- Give the factor a name.\n"
        "- Check if that factor spikes during known historical episodes.\n"
    )


def concept_api_and_caching_deep_dive() -> str:
    return (
        "### Deep Dive: APIs + Caching (How Real-World Data Ingestion Works)\n"
        "\n"
        "This project intentionally makes you interact with real APIs because data ingestion is where many ML projects fail.\n"
        "In practice, you need to understand *protocols*, *schemas*, and *reproducibility*.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **API (Application Programming Interface)**: a contract for requesting and receiving structured data.\n"
        "- **HTTP**: the protocol used for requests/responses (what `requests.get(...)` uses under the hood).\n"
        "- **Endpoint**: a specific API path (e.g., `series/observations`).\n"
        "- **Query parameters**: key/value inputs in the URL (e.g., `series_id=UNRATE`).\n"
        "- **Status code**: tells you whether a request succeeded (200) or failed (4xx/5xx).\n"
        "- **Timeout**: how long you wait before giving up (prevents hanging forever).\n"
        "- **Retry/backoff**: re-attempting requests after transient failures.\n"
        "- **Schema**: expected structure of the JSON payload.\n"
        "- **Cache**: stored copy of responses used to avoid repeated calls and make runs reproducible.\n"
        "\n"
        "#### Why caching matters for learning (and for production)\n"
        "- **Speed**: you can iterate without waiting on the network.\n"
        "- **Reproducibility**: your results do not change because an endpoint changed or the data was revised.\n"
        "- **Debuggability**: you can inspect raw payloads when parsing fails.\n"
        "\n"
        "#### Raw vs processed data\n"
        "- `data/raw/`: the API's response (JSON) with minimal transformation.\n"
        "- `data/processed/`: tables you created (CSV) after cleaning and aligning time.\n"
        "\n"
        "#### Python demo: minimal caching pattern\n"
        "```python\n"
        "from __future__ import annotations\n"
        "\n"
        "import json\n"
        "from pathlib import Path\n"
        "import requests\n"
        "\n"
        "def load_or_fetch_json(path: Path, fetch_fn):\n"
        "    if path.exists():\n"
        "        return json.loads(path.read_text())\n"
        "    payload = fetch_fn()\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text(json.dumps(payload))\n"
        "    return payload\n"
        "\n"
        "def fetch_example(url: str, params: dict):\n"
        "    r = requests.get(url, params=params, timeout=30)\n"
        "    r.raise_for_status()  # raises on 4xx/5xx\n"
        "    return r.json()\n"
        "```\n"
        "\n"
        "#### Debug playbook: when your API code fails\n"
        "1. Print the full URL + params (so you can reproduce outside Python).\n"
        "2. Inspect the status code and response body.\n"
        "3. Cache the raw payload and re-run parsing offline.\n"
        "4. Validate schema assumptions (`payload.keys()`, sample rows).\n"
        "5. Add defensive parsing (type conversion, missing markers, etc.).\n"
        "\n"
        "#### Economics caveat: revisions and vintages\n"
        "- Some macro series are revised after initial release (GDP is a classic example).\n"
        "- If you re-fetch later, historical values can change; caching avoids silent drift.\n"
    )


def concept_resampling_and_alignment_deep_dive() -> str:
    return (
        "### Deep Dive: Frequency Alignment (Daily/Monthly/Quarterly)\n"
        "\n"
        "Economic indicators arrive at different frequencies:\n"
        "- GDP is quarterly.\n"
        "- CPI/unemployment are monthly.\n"
        "- Yield curve spreads can be daily.\n"
        "\n"
        "To build a single modeling table, you must choose a *timeline* and convert everything onto it.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **Resampling**: converting a time series to a new frequency (e.g., daily -> monthly).\n"
        "- **Aggregation**: combining many observations into one (mean/last/sum).\n"
        "- **Forward-fill (ffill)**: carrying the last known value forward until updated.\n"
        "- **As-of merge**: joining on the most recent available observation (common in finance).\n"
        "\n"
        "#### Why month-end is a pragmatic choice\n"
        "- Many macro series are monthly.\n"
        "- Daily series can be summarized to month-end (`last`) or month-average (`mean`).\n"
        "- Month-end timestamps make quarterly aggregation easier.\n"
        "\n"
        "#### Python demo: daily -> month-end (last vs mean)\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "idx = pd.date_range('2020-01-01', periods=120, freq='D')\n"
        "x_daily = pd.Series(np.random.default_rng(0).normal(size=len(idx)).cumsum(), index=idx)\n"
        "\n"
        "x_me_last = x_daily.resample('ME').last()\n"
        "x_me_mean = x_daily.resample('ME').mean()\n"
        "```\n"
        "\n"
        "#### Missing data and forward-fill (what it assumes)\n"
        "- Forward-fill assumes the last published value remains \"true\" until updated.\n"
        "- This is often reasonable for policy rates between meetings, but can be questionable for volatile indicators.\n"
        "- Always inspect long gaps: forward-filling across multi-month gaps can create fake stability.\n"
        "\n"
        "#### Debug checks\n"
        "- Are timestamps sorted and unique?\n"
        "- After resampling, do you have the expected frequency (`ME` rows, `QE` rows)?\n"
        "- Do joins create NaNs? If so, which series and why?\n"
    )


def concept_gdp_growth_and_recession_label_deep_dive() -> str:
    return (
        "### Deep Dive: GDP Growth Math + Technical Recession Labels\n"
        "\n"
        "GDP is a *level* series (a quantity). A recession label requires turning levels into *growth rates*.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **Level**: the raw value of a series (e.g., real GDP in chained dollars).\n"
        "- **Growth rate**: percent change of a level over a period.\n"
        "- **QoQ (quarter-over-quarter)**: compares GDP_t to GDP_{t-1}.\n"
        "- **YoY (year-over-year)**: compares GDP_t to GDP_{t-4}.\n"
        "- **Annualized growth**: converting a quarterly growth rate into an annual pace.\n"
        "- **Technical recession (proxy)**: two consecutive negative quarters of QoQ GDP growth.\n"
        "\n"
        "#### Growth formulas (and what they mean)\n"
        "- QoQ percent growth:\n"
        "  - `g_qoq[t] = 100 * (GDP[t]/GDP[t-1] - 1)`\n"
        "- QoQ annualized percent growth:\n"
        "  - `g_ann[t] = 100 * ((GDP[t]/GDP[t-1])**4 - 1)`\n"
        "- YoY percent growth:\n"
        "  - `g_yoy[t] = 100 * (GDP[t]/GDP[t-4] - 1)`\n"
        "\n"
        "Why multiple growth measures?\n"
        "- QoQ is more \"responsive\" but noisier.\n"
        "- YoY is smoother but reacts later.\n"
        "\n"
        "#### Label construction (this project)\n"
        "- Recession at quarter t:\n"
        "  - `recession[t] = 1 if (g_qoq[t] < 0 and g_qoq[t-1] < 0) else 0`\n"
        "- Next-quarter target:\n"
        "  - `target_recession_next_q[t] = recession[t+1]`\n"
        "\n"
        "#### Critical limitation (interpretation)\n"
        "- This is a clean teaching label, but it is **not** an official recession dating rule.\n"
        "- Official recession dating uses multiple indicators and can disagree with the 2-quarter rule.\n"
        "\n"
        "#### Python demo: label edge cases\n"
        "```python\n"
        "import pandas as pd\n"
        "\n"
        "growth = pd.Series([1.0, -0.1, -0.2, 0.3, -0.1, -0.1])\n"
        "recession = ((growth < 0) & (growth.shift(1) < 0)).astype(int)\n"
        "target_next = recession.shift(-1)\n"
        "pd.DataFrame({'growth': growth, 'recession': recession, 'target_next': target_next})\n"
        "```\n"
        "\n"
        "#### Macro caveat: revisions\n"
        "- GDP is revised. If you re-fetch later, the computed label can change.\n"
        "- That is one reason we emphasize caching raw data.\n"
    )


def concept_quarterly_feature_engineering_deep_dive() -> str:
    return (
        "### Deep Dive: Monthly -> Quarterly Features (No-Leakage Engineering)\n"
        "\n"
        "Your target is quarterly (GDP growth / recession label), but many predictors are monthly/daily.\n"
        "You must transform predictors into quarterly features that were available at the time.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **Aggregation window**: the period you summarize (e.g., quarter).\n"
        "- **Quarter-average**: average monthly values within the quarter.\n"
        "- **Quarter-end**: last available value in the quarter.\n"
        "- **Lag**: a past value used as a feature (e.g., x_{t-1}).\n"
        "- **Horizon**: how far ahead you are predicting (here: next quarter).\n"
        "\n"
        "#### Two defensible quarterly feature definitions\n"
        "- Quarter-average features (mean): captures typical conditions during the quarter.\n"
        "- Quarter-end features (last): captures conditions at the end of the quarter.\n"
        "\n"
        "#### Leakage risk: using information from inside the target quarter\n"
        "Be explicit about what your prediction timestamp means.\n"
        "- If you predict recession_{t+1} using quarter t features, ensure features only use information up to end of quarter t.\n"
        "- If you predict *during* quarter t, you would need partial-quarter features (nowcasting).\n"
        "\n"
        "#### Python demo: quarterly aggregation + lags\n"
        "```python\n"
        "import pandas as pd\n"
        "\n"
        "panel_monthly = pd.read_csv('data/sample/panel_monthly_sample.csv', index_col=0, parse_dates=True)\n"
        "\n"
        "q_mean = panel_monthly.resample('QE').mean()\n"
        "q_last = panel_monthly.resample('QE').last()\n"
        "\n"
        "# Example lag features\n"
        "q = q_mean.add_prefix('mean_')\n"
        "q['mean_UNRATE_lag1'] = q['mean_UNRATE'].shift(1)\n"
        "q = q.dropna()\n"
        "```\n"
        "\n"
        "#### Debug checks for leakage\n"
        "1. Are all lags non-negative (shift(+k))?\n"
        "2. Does `target_recession_next_q` shift the correct direction?\n"
        "3. Does the feature table end at the same quarter as the target (after dropna)?\n"
        "4. If you recompute with a different aggregation (mean vs last), do conclusions change?\n"
    )


def concept_micro_regression_log_log_deep_dive() -> str:
    return (
        "### Deep Dive: Log-Log Regression (Elasticity-Style Interpretation)\n"
        "\n"
        "In microeconomic cross-sectional data, log transforms are common because:\n"
        "- relationships are often multiplicative (percent changes matter), and\n"
        "- log transforms compress heavy tails (income, population, housing values).\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **Cross-sectional data**: many units (counties) observed at one time.\n"
        "- **Log transform**: `log(x)`; turns multiplicative relationships into additive ones.\n"
        "- **Elasticity (informal here)**: in a log-log model, the slope approximates a percent-percent relationship.\n"
        "\n"
        "#### Model and interpretation\n"
        "- Log-log model: `log(y) = a + b*log(x) + e`\n"
        "- Interpretation of b (rule of thumb): a 1% increase in x is associated with ~b% increase in y.\n"
        "\n"
        "#### Pitfall: zeros and missing values\n"
        "- `log(0)` is undefined.\n"
        "- You must filter out non-positive values or use transformations like `log1p` (context-dependent).\n"
        "\n"
        "#### Python demo: interpreting b\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import statsmodels.api as sm\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 800\n"
        "x = np.exp(rng.normal(size=n))\n"
        "y = 2.0 * (x ** 0.3) * np.exp(rng.normal(scale=0.2, size=n))\n"
        "\n"
        "df = pd.DataFrame({'x': x, 'y': y})\n"
        "df['lx'] = np.log(df['x'])\n"
        "df['ly'] = np.log(df['y'])\n"
        "\n"
        "X = sm.add_constant(df[['lx']])\n"
        "res = sm.OLS(df['ly'], X).fit()\n"
        "print(res.params)  # slope ~ 0.3\n"
        "```\n"
    )


def concept_robust_se_hc3_deep_dive() -> str:
    return (
        "### Deep Dive: Robust Standard Errors (HC3) for Cross-Sectional Data\n"
        "\n"
        "In cross-sectional economics, heteroskedasticity is common: richer counties often have different variance in outcomes.\n"
        "Naive OLS standard errors assume constant variance; HC3 relaxes that.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **Heteroskedasticity**: error variance changes with x.\n"
        "- **Robust SE**: covariance estimates that remain valid under certain violations.\n"
        "- **HC3**: a popular heteroskedasticity-robust SE variant (often conservative).\n"
        "\n"
        "#### What changes when you use HC3?\n"
        "- Coefficients (beta) do not change.\n"
        "- Standard errors / p-values / confidence intervals change.\n"
        "\n"
        "#### Python demo: heteroskedastic errors and robust SE\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import statsmodels.api as sm\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 400\n"
        "x = rng.normal(size=n)\n"
        "\n"
        "# Error variance increases with |x|\n"
        "eps = rng.normal(scale=1 + 2*np.abs(x), size=n)\n"
        "y = 1.0 + 0.5*x + eps\n"
        "\n"
        "X = sm.add_constant(pd.DataFrame({'x': x}))\n"
        "res = sm.OLS(y, X).fit()\n"
        "res_hc3 = res.get_robustcov_results(cov_type='HC3')\n"
        "\n"
        "print('naive SE:', res.bse)\n"
        "print('HC3 SE  :', res_hc3.bse)\n"
        "```\n"
        "\n"
        "#### Interpretation warning\n"
        "- A small p-value is not a causal certificate.\n"
        "- Robust SE helps with *uncertainty* under heteroskedasticity, not with confounding.\n"
    )


def concept_omitted_variable_bias_deep_dive() -> str:
    return (
        "### Deep Dive: Omitted Variable Bias (Why Adding Controls Changes Coefficients)\n"
        "\n"
        "**Omitted variable bias (OVB)** happens when:\n"
        "1) you omit a variable Z that affects Y, and\n"
        "2) Z is correlated with an included regressor X.\n"
        "\n"
        "Then the coefficient on X partly absorbs Z's effect.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **Confounder**: a variable related to both X and Y.\n"
        "- **Control variable**: a variable included to reduce confounding.\n"
        "- **Specification**: the set of variables you include in a regression.\n"
        "\n"
        "#### Python demo: a confounder makes X look important\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import statsmodels.api as sm\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 2000\n"
        "\n"
        "# Z affects Y, and Z also affects X.\n"
        "z = rng.normal(size=n)\n"
        "x = 0.8*z + rng.normal(scale=1.0, size=n)\n"
        "y = 2.0*z + rng.normal(scale=1.0, size=n)\n"
        "\n"
        "df = pd.DataFrame({'y': y, 'x': x, 'z': z})\n"
        "\n"
        "# Omitted Z: biased coefficient on x\n"
        "res_omit = sm.OLS(df['y'], sm.add_constant(df[['x']])).fit()\n"
        "\n"
        "# Include Z: coefficient on x shrinks toward 0\n"
        "res_full = sm.OLS(df['y'], sm.add_constant(df[['x', 'z']])).fit()\n"
        "\n"
        "print('omit z:', res_omit.params)\n"
        "print('full  :', res_full.params)\n"
        "```\n"
        "\n"
        "#### Practical rule\n"
        "- If your coefficient flips sign or changes drastically when adding plausible controls,\n"
        "  your original interpretation was likely fragile.\n"
    )


def concept_regularization_ridge_lasso_deep_dive() -> str:
    return (
        "### Deep Dive: Regularization (Ridge vs Lasso)\n"
        "\n"
        "Regularization adds a penalty to the loss function to reduce overfitting and stabilize coefficients.\n"
        "In macro data with correlated indicators, it is often essential.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **Regularization**: adding a penalty term to discourage large coefficients.\n"
        "- **L2 penalty (ridge)**: penalizes squared coefficients.\n"
        "- **L1 penalty (lasso)**: penalizes absolute coefficients; can drive some to exactly 0.\n"
        "- **Alpha (lambda)**: strength of the penalty (hyperparameter).\n"
        "- **Coefficient path**: coefficients as a function of alpha.\n"
        "\n"
        "#### Objectives (math)\n"
        "- OLS: minimize `||y - Xβ||^2`\n"
        "- Ridge: minimize `||y - Xβ||^2 + α * ||β||_2^2`\n"
        "- Lasso: minimize `||y - Xβ||^2 + α * ||β||_1`\n"
        "\n"
        "#### Why standardization matters\n"
        "- Penalties depend on coefficient magnitudes.\n"
        "- If features are on different scales (percent vs index points), the penalty is uneven.\n"
        "- Standardize (`StandardScaler`) before ridge/lasso.\n"
        "\n"
        "#### Ridge vs lasso when predictors are correlated\n"
        "- Ridge tends to shrink correlated predictors together (grouping effect).\n"
        "- Lasso often picks one feature from a correlated group (can be unstable across samples).\n"
        "\n"
        "#### Python demo: coefficient instability vs stabilization\n"
        "```python\n"
        "import numpy as np\n"
        "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.pipeline import Pipeline\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 300\n"
        "x1 = rng.normal(size=n)\n"
        "x2 = x1 * 0.98 + rng.normal(scale=0.2, size=n)  # highly correlated\n"
        "y = 1.0 + 2.0*x1 + rng.normal(scale=1.0, size=n)\n"
        "X = np.column_stack([x1, x2])\n"
        "\n"
        "ols = LinearRegression().fit(X, y)\n"
        "ridge = Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=5.0))]).fit(X, y)\n"
        "lasso = Pipeline([('sc', StandardScaler()), ('m', Lasso(alpha=0.1, max_iter=10000))]).fit(X, y)\n"
        "\n"
        "print('ols  coef:', ols.coef_)\n"
        "print('ridge coef:', ridge.named_steps['m'].coef_)\n"
        "print('lasso coef:', lasso.named_steps['m'].coef_)\n"
        "```\n"
        "\n"
        "#### Interpretation warning\n"
        "- Regularized coefficients are biased by design.\n"
        "- They can be excellent for prediction, but do not treat them as classical OLS inference objects.\n"
        "- Prefer out-of-sample evaluation and stability checks.\n"
    )


def concept_rolling_regression_stability_deep_dive() -> str:
    return (
        "### Deep Dive: Rolling Regressions (Stability and Structural Breaks)\n"
        "\n"
        "A rolling regression repeatedly re-fits a model on a moving window of past data.\n"
        "This is a practical way to detect coefficient drift and regime sensitivity.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **Rolling window**: a fixed-size window that moves forward through time.\n"
        "- **Expanding window**: a window that grows over time (always includes all past).\n"
        "- **Structural break**: the relationship between X and Y changes.\n"
        "- **Regime**: an era where relationships are relatively stable.\n"
        "\n"
        "#### Why this matters in macro\n"
        "- Policy regimes change.\n"
        "- Financial structure changes.\n"
        "- Data definitions change.\n"
        "A single \"global\" coefficient can hide these shifts.\n"
        "\n"
        "#### Python demo: relationship changes mid-sample\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import statsmodels.api as sm\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 200\n"
        "x = rng.normal(size=n)\n"
        "\n"
        "# Coefficient changes halfway\n"
        "beta = np.r_[np.repeat(0.2, n//2), np.repeat(-0.2, n - n//2)]\n"
        "y = 1.0 + beta*x + rng.normal(scale=1.0, size=n)\n"
        "\n"
        "idx = pd.date_range('1970-03-31', periods=n, freq='QE')\n"
        "df = pd.DataFrame({'y': y, 'x': x}, index=idx)\n"
        "\n"
        "window = 60\n"
        "betas = []\n"
        "dates = []\n"
        "for end in range(window, len(df)+1):\n"
        "    sub = df.iloc[end-window:end]\n"
        "    res = sm.OLS(sub['y'], sm.add_constant(sub[['x']])).fit()\n"
        "    betas.append(res.params['x'])\n"
        "    dates.append(sub.index[-1])\n"
        "\n"
        "out = pd.Series(betas, index=dates)\n"
        "out.head()\n"
        "```\n"
        "\n"
        "#### Interpretation\n"
        "- If coefficients drift, your model is not describing a single stable mechanism.\n"
        "- For prediction, you may prefer recent windows.\n"
        "- For inference, you must be careful about claiming a single \"effect\" across eras.\n"
    )


def concept_class_imbalance_and_metrics_deep_dive() -> str:
    return (
        "### Deep Dive: Class Imbalance and Why Accuracy Lies\n"
        "\n"
        "Recessions are rare. That means classification is an imbalanced problem.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **Base rate**: prevalence of the positive class (fraction of recession quarters).\n"
        "- **Imbalanced data**: one class is much rarer than the other.\n"
        "- **Precision**: among predicted positives, how many were true positives?\n"
        "- **Recall**: among true positives, how many did we catch?\n"
        "- **ROC-AUC**: ranking quality across thresholds (can look good even if precision is poor).\n"
        "- **PR-AUC**: focuses on positive-class retrieval (often more honest for rare events).\n"
        "- **Proper scoring rule**: rewards calibrated probabilities (log loss, Brier score).\n"
        "\n"
        "#### The accuracy trap\n"
        "- If recessions happen 10% of the time, a model that always predicts \"no recession\" has 90% accuracy.\n"
        "- But it is useless.\n"
        "\n"
        "#### Baselines you should always compute\n"
        "- Majority class (always 0)\n"
        "- Persistence (predict next = current)\n"
        "- Simple heuristic (e.g., yield curve inversion rule)\n"
        "\n"
        "#### Python demo: why PR matters\n"
        "```python\n"
        "import numpy as np\n"
        "from sklearn.metrics import roc_auc_score, average_precision_score\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 500\n"
        "y = rng.binomial(1, 0.1, size=n)  # 10% positives\n"
        "\n"
        "# A weak signal score\n"
        "score = 0.2*y + rng.normal(scale=1.0, size=n)\n"
        "\n"
        "print('ROC-AUC:', roc_auc_score(y, score))\n"
        "print('PR-AUC :', average_precision_score(y, score))\n"
        "```\n"
        "\n"
        "#### Decision framing\n"
        "- Choose thresholds based on *costs* (false positives vs false negatives), not vibes.\n"
        "- Calibration matters if you want probabilities you can act on.\n"
    )


def concept_tree_models_and_importance_deep_dive() -> str:
    return (
        "### Deep Dive: Tree Models + Feature Importance (What To Trust)\n"
        "\n"
        "Tree models can capture non-linear relationships and interactions that linear models miss.\n"
        "But they are easier to overfit and harder to interpret.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **Decision tree**: splits data by thresholds on features.\n"
        "- **Random forest**: averages many trees trained on bootstrapped samples.\n"
        "- **Gradient boosting**: sequentially adds trees to correct errors.\n"
        "- **Overfitting**: learning noise; appears as high train performance, low test performance.\n"
        "- **Gini importance**: impurity-based importance from trees (can be biased).\n"
        "- **Permutation importance**: importance from shuffling a feature and measuring performance drop.\n"
        "\n"
        "#### Why tree feature importances can mislead\n"
        "- Impurity-based importance can favor:\n"
        "  - high-cardinality features,\n"
        "  - noisy continuous features,\n"
        "  - correlated features (importance can be split or concentrated unpredictably).\n"
        "\n"
        "#### Python demo: impurity vs permutation importance\n"
        "```python\n"
        "import numpy as np\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.inspection import permutation_importance\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.metrics import roc_auc_score\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 800\n"
        "\n"
        "# Two correlated features + one noise feature\n"
        "x1 = rng.normal(size=n)\n"
        "x2 = x1 * 0.9 + rng.normal(scale=0.5, size=n)\n"
        "x3 = rng.normal(size=n)\n"
        "X = np.column_stack([x1, x2, x3])\n"
        "p = 1 / (1 + np.exp(-(0.5 + 1.0*x1)))\n"
        "y = rng.binomial(1, p)\n"
        "\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)\n"
        "rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_tr, y_tr)\n"
        "print('AUC:', roc_auc_score(y_te, rf.predict_proba(X_te)[:,1]))\n"
        "print('gini importances:', rf.feature_importances_)\n"
        "\n"
        "pi = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=0, scoring='roc_auc')\n"
        "print('perm importances:', pi.importances_mean)\n"
        "```\n"
        "\n"
        "#### Practical interpretation\n"
        "- Treat importance as \"usefulness for prediction\", not causal influence.\n"
        "- Compare importances across eras (walk-forward) to see if drivers change.\n"
    )


def concept_clustering_regimes_deep_dive() -> str:
    return (
        "### Deep Dive: Clustering as Regime Discovery\n"
        "\n"
        "Clustering groups time periods with similar indicator patterns.\n"
        "You can treat clusters as candidate \"macro regimes\" and then compare them to recessions.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **k-means**: clustering method that minimizes within-cluster squared distances to centroids.\n"
        "- **Inertia**: k-means objective value (lower is better, always decreases with k).\n"
        "- **Silhouette score**: measures separation between clusters (higher is better).\n"
        "- **Standardization**: required because distance depends on scale.\n"
        "\n"
        "#### Choosing k\n"
        "- There is no single \"correct\" k.\n"
        "- Use elbow plots (inertia), silhouette scores, and interpretability.\n"
        "\n"
        "#### Python demo: k-means + silhouette\n"
        "```python\n"
        "import numpy as np\n"
        "from sklearn.cluster import KMeans\n"
        "from sklearn.metrics import silhouette_score\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "X = rng.normal(size=(200, 4))\n"
        "X = StandardScaler().fit_transform(X)\n"
        "\n"
        "for k in [2, 3, 4, 5]:\n"
        "    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)\n"
        "    sil = silhouette_score(X, km.labels_)\n"
        "    print(k, 'inertia', km.inertia_, 'sil', sil)\n"
        "```\n"
        "\n"
        "#### Interpretation playbook\n"
        "1. Compute cluster centroids in original units (undo scaling) for interpretability.\n"
        "2. Name clusters in economic language (\"high inflation/high rates\", etc.).\n"
        "3. Check whether certain clusters are recession-heavy.\n"
    )


def concept_anomaly_detection_deep_dive() -> str:
    return (
        "### Deep Dive: Anomaly Detection (Crisis Periods)\n"
        "\n"
        "Anomaly detection flags observations that look unusual relative to the bulk of the data.\n"
        "In macro, anomalies often correspond to crises (e.g., 2008, 2020), but not always.\n"
        "\n"
        "#### Key Terms (defined)\n"
        "- **Outlier**: an observation far from typical behavior.\n"
        "- **Anomaly score**: a numeric measure of \"unusualness\".\n"
        "- **Isolation Forest**: detects anomalies by how easily points are isolated by random splits.\n"
        "- **Contamination**: expected fraction of anomalies (hyperparameter).\n"
        "\n"
        "#### Python demo: Isolation Forest intuition\n"
        "```python\n"
        "import numpy as np\n"
        "from sklearn.ensemble import IsolationForest\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "X_normal = rng.normal(size=(300, 3))\n"
        "X_anom = rng.normal(loc=6.0, scale=1.0, size=(10, 3))\n"
        "X = np.vstack([X_normal, X_anom])\n"
        "X = StandardScaler().fit_transform(X)\n"
        "\n"
        "iso = IsolationForest(contamination=0.05, random_state=0).fit(X)\n"
        "scores = -iso.score_samples(X)  # higher = more anomalous\n"
        "scores[-10:].round(3)\n"
        "```\n"
        "\n"
        "#### Interpreting anomalies\n"
        "- An anomaly is not automatically \"bad\" or \"recession\".\n"
        "- It means the pattern of indicators is rare.\n"
        "- Compare anomaly flags to your recession label to see overlaps and differences.\n"
        "\n"
        "#### Debug tips\n"
        "- Always standardize before distance/forest methods.\n"
        "- Sensitivity-check the contamination parameter.\n"
        "- Inspect which features contribute (e.g., via z-scores) for flagged periods.\n"
    )


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

        technical = (
            "### Why Time Ordering Changes Everything\n"
            "- Random splits destroy chronology. In forecasting tasks, your model must only use information available at prediction time.\n"
            "\n"
            "### Correlation vs Causation\n"
            "- **Correlation**: variables move together.\n"
            "- **Causation**: changing X changes Y (a stronger claim that needs design/assumptions).\n"
            "- In macro/micro data, you will see many correlated variables. Treat coefficients as *associational* unless you have a causal identification strategy.\n"
            "\n"
            "### Multicollinearity and VIF\n"
            "- **VIF (Variance Inflation Factor)**: how much a coefficient's variance is inflated due to collinearity.\n"
            "- High VIF implies unstable coefficients and wide confidence intervals, even if predictions are okay.\n"
        )

        # Stem-specific deep dives (first introduction gets the deepest treatment).
        if stem == "01_time_series_basics":
            technical = (
                "This notebook introduces the two most important ideas for economic ML:\n"
                "1) time-aware evaluation, and\n"
                "2) leakage prevention.\n\n"
                + concept_time_split_deep_dive()
                + "\n\n"
                + concept_leakage_deep_dive()
            )
        elif stem == "02_stats_basics_for_ml":
            technical = (
                "This notebook is about the statistical failure modes that make models look smart when they are not.\n\n"
                "### Deep Dive: Correlation vs Causation (Practical)\n"
                "- **Correlation** answers: do X and Y move together?\n"
                "- **Causation** answers: if we intervene on X, does Y change?\n"
                "- Most observational economic datasets support correlation claims by default, not causal claims.\n\n"
                "### Deep Dive: Overfitting (What It Looks Like)\n"
                "- Overfitting is when a model learns noise specific to the training sample.\n"
                "- Symptom: training performance improves while test performance stagnates or worsens.\n"
                "- Fixes: simpler models, more data, regularization, better features, better splits.\n\n"
                + concept_multicollinearity_vif_deep_dive()
            )

        mistakes_items = [
            "Using `train_test_split(shuffle=True)` on time-indexed data.",
            "Looking at the test set repeatedly while tuning (\"test leakage\").",
            "Assuming a significant p-value implies causation.",
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

        technical = (
            "### GDP Growth Formulas\n"
            "- QoQ growth (percent): `100 * (GDP_t / GDP_{t-1} - 1)`\n"
            "- Annualized QoQ (percent): `100 * ((GDP_t / GDP_{t-1})^4 - 1)`\n"
            "- YoY growth (percent): `100 * (GDP_t / GDP_{t-4} - 1)`\n"
            "\n"
            "### Technical Recession Label (Proxy)\n"
            "- We define a recession quarter as: two consecutive quarters of negative QoQ GDP growth.\n"
            "- This is *not* an official recession dating rule, but a clear, computable proxy for teaching.\n"
            "\n"
            "### Frequency Alignment and Aggregation Choices\n"
            "- Monthly predictors aggregated to quarterly can use:\n"
            "  - **quarter-average**: captures typical conditions across the quarter.\n"
            "  - **quarter-end**: captures end-of-quarter conditions.\n"
            "- Both can be defensible; the point is to choose intentionally and document the choice.\n"
        )

        # Stem-specific deep dives.
        if stem == "00_fred_api_and_caching":
            technical = concept_api_and_caching_deep_dive()
        elif stem == "01_build_macro_monthly_panel":
            technical = concept_resampling_and_alignment_deep_dive()
        elif stem == "02_gdp_growth_and_recession_label":
            technical = concept_gdp_growth_and_recession_label_deep_dive()
        elif stem == "03_build_macro_quarterly_features":
            technical = concept_quarterly_feature_engineering_deep_dive()

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

        technical = (
            "### OLS Objective\n"
            "- Model: `y = Xβ + ε`\n"
            "- OLS chooses `β` to minimize `Σ (y_i - ŷ_i)^2`.\n"
            "\n"
            "### Interpreting Coefficients\n"
            "- In a simple regression with one feature, the slope is the expected change in `y` for a +1 change in `x`.\n"
            "- In a multi-factor regression, the slope is the expected change in `y` for a +1 change in `x_j` *holding other X fixed*.\n"
            "- If features are correlated (multicollinearity), \"holding others fixed\" can be a fragile, unrealistic counterfactual.\n"
            "\n"
            "### Inference vs Prediction\n"
            "- Inference: emphasize coefficient uncertainty and assumptions.\n"
            "- Prediction: emphasize out-of-sample performance.\n"
            "- You can have strong prediction with weak/unstable coefficients.\n"
            "\n"
            "### Robust Standard Errors\n"
            "- **HC3** addresses heteroskedasticity (common for cross-sectional county data).\n"
            "- **HAC/Newey-West** addresses autocorrelation + heteroskedasticity (common for quarterly macro time series).\n"
        )

        if stem == "04_inference_time_series_hac":
            technical = technical + "\n\n" + concept_hac_newey_west_deep_dive()
        elif stem == "00_single_factor_regression_micro":
            technical = (
                concept_micro_regression_log_log_deep_dive()
                + "\n\n"
                + concept_robust_se_hc3_deep_dive()
                + "\n\n"
                + technical
            )
        elif stem == "01_multifactor_regression_micro_controls":
            technical = (
                concept_omitted_variable_bias_deep_dive()
                + "\n\n"
                + concept_robust_se_hc3_deep_dive()
                + "\n\n"
                + technical
            )
        elif stem == "05_regularization_ridge_lasso":
            technical = concept_regularization_ridge_lasso_deep_dive() + "\n\n" + technical
        elif stem == "06_rolling_regressions_stability":
            technical = concept_rolling_regression_stability_deep_dive() + "\n\n" + technical

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

        technical = (
            "### Logistic Regression Mechanics\n"
            "- Score: `z = β0 + β1 x1 + ...`\n"
            "- Probability: `p = 1 / (1 + exp(-z))`\n"
            "- Training minimizes **log loss** (cross-entropy), not squared error.\n"
            "\n"
            "### Metrics: When to Use Which\n"
            "- ROC-AUC: good for ranking; can be optimistic with heavy class imbalance.\n"
            "- PR-AUC: focuses on the positive class; often more informative for rare recessions.\n"
            "- Brier score: penalizes miscalibrated probabilities.\n"
            "\n"
            "### Thresholds and Decision Costs\n"
            "- If false positives are expensive (crying wolf), raise threshold.\n"
            "- If missing a recession is expensive, lower threshold.\n"
        )

        if stem == "01_logistic_recession_classifier":
            technical = technical + "\n\n" + concept_logistic_regression_odds_deep_dive()
        elif stem == "00_recession_classifier_baselines":
            technical = technical + "\n\n" + concept_class_imbalance_and_metrics_deep_dive()
        elif stem == "02_calibration_and_costs":
            technical = technical + "\n\n" + concept_calibration_brier_deep_dive()
        elif stem == "03_tree_models_and_importance":
            technical = technical + "\n\n" + concept_tree_models_and_importance_deep_dive()
        elif stem == "04_walk_forward_validation":
            technical = technical + "\n\n" + concept_walk_forward_validation_deep_dive()

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

        technical = (
            "### PCA Intuition\n"
            "- PCA finds directions that explain maximum variance.\n"
            "- Components are orthogonal (uncorrelated by construction).\n"
            "- Loadings help you interpret what each factor represents.\n"
            "\n"
            "### Clustering Intuition\n"
            "- k-means finds k centroids and assigns each point to the closest.\n"
            "- Choosing k is a modeling decision; use elbow plots and interpretability.\n"
        )

        if stem == "01_pca_macro_factors":
            technical = technical + "\n\n" + concept_pca_loadings_deep_dive()
        elif stem == "02_clustering_macro_regimes":
            technical = technical + "\n\n" + concept_clustering_regimes_deep_dive()
        elif stem == "03_anomaly_detection":
            technical = technical + "\n\n" + concept_anomaly_detection_deep_dive()

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

        technical = (
            "### Why configs matter\n"
            "- They turn hidden notebook state into explicit, reviewable decisions.\n"
            "\n"
            "### Dataset hashing\n"
            "- Hashes help you confirm which dataset a model was trained on.\n"
            "- In production you would also track schema versions and feature code versions.\n"
        )

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

        technical = (
            "### What a strong capstone includes\n"
            "- A clear target definition and a defensible label.\n"
            "- Time-aware evaluation (no leakage).\n"
            "- Interpretation: drivers, failure cases, and limitations.\n"
            "- Reproducible artifacts and documentation.\n"
        )

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

    out_path.write_text(
        GUIDE_TEMPLATE.format(
            stem=stem,
            intro=intro,
            checklist=checklist,
            alt_example=alt_example,
            technical=technical,
            mistakes=mistakes,
            summary=summary,
            readings=readings,
        )
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
        for spec in sorted(by_category[category], key=lambda s: s.path):
            stem = Path(spec.path).stem
            lines.append(f"- [{stem}]({category}/{stem}.md)")
        lines.append("")

    out_path = root / "docs" / "guides" / "index.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n")


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

    write_guides_index(specs, root)
    write_docs_index(specs, root)


if __name__ == "__main__":
    main()
