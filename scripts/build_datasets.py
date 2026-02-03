"""Build processed datasets for the tutorial.

This script is intentionally transparent and not "too magical". It builds:
- Macro monthly panel from FRED indicators
- Quarterly GDP growth + technical recession labels
- Macro quarterly modeling table (features + targets)
- Census/ACS county dataset

Usage:
  python scripts/build_datasets.py \
    --recession-config configs/recession.yaml \
    --census-config configs/census.yaml

Outputs:
  data/processed/panel_monthly.csv
  data/processed/gdp_quarterly.csv
  data/processed/macro_quarterly.csv
  data/processed/census_county_<year>.csv

If API keys are missing, notebooks can fall back to `data/sample/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import census_api
from src import data as data_utils
from src import fred_api
from src import macro

from scripts._common import load_yaml, project_root


def build_macro_monthly_panel(cfg: dict, *, root: Path) -> pd.DataFrame:
    fred_cfg = cfg["fred"]
    start = fred_cfg.get("start_date")
    end = fred_cfg.get("end_date")
    series_ids = list(fred_cfg.get("series", []))

    raw_dir = root / "data" / "raw" / "fred"
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for sid in series_ids:
        payload = data_utils.load_or_fetch_json(
            raw_dir / f"{sid}.json",
            lambda sid=sid: fred_api.fetch_series_observations(sid, start_date=start, end_date=end),
        )
        frames.append(fred_api.observations_to_frame(payload, sid))

    raw_panel = pd.concat(frames, axis=1).sort_index()

    # Normalize to monthly frequency:
    # - daily series -> month-end
    # - monthly series -> month-end
    # Then forward-fill to handle gaps and differing start dates.
    monthly = raw_panel.resample("ME").last().ffill()
    return monthly


def build_gdp_quarterly(cfg: dict, *, root: Path) -> pd.DataFrame:
    fred_cfg = cfg["fred"]
    start = fred_cfg.get("start_date")
    end = fred_cfg.get("end_date")
    gdp_series = str(fred_cfg["gdp_series"])

    raw_dir = root / "data" / "raw" / "fred"
    raw_dir.mkdir(parents=True, exist_ok=True)

    payload = data_utils.load_or_fetch_json(
        raw_dir / f"{gdp_series}.json",
        lambda: fred_api.fetch_series_observations(gdp_series, start_date=start, end_date=end),
    )

    gdp = fred_api.observations_to_frame(payload, gdp_series)

    # FRED quarterly dates are often quarter-start timestamps. Normalize to quarter-end
    # so they merge cleanly with resampled quarterly predictors.
    gdp.index = gdp.index.to_period("Q").to_timestamp("Q")

    levels = gdp[gdp_series]
    out = pd.DataFrame({gdp_series: levels})
    out["gdp_growth_qoq"] = macro.gdp_growth_qoq(levels)
    out["gdp_growth_qoq_annualized"] = macro.gdp_growth_qoq_annualized(levels)
    out["gdp_growth_yoy"] = macro.gdp_growth_yoy(levels)

    out["recession"] = macro.technical_recession_label(out["gdp_growth_qoq"])
    out["target_recession_next_q"] = macro.next_period_target(out["recession"])

    return out


def build_macro_quarterly_table(cfg: dict, *, root: Path, panel_monthly: pd.DataFrame, gdp_q: pd.DataFrame) -> pd.DataFrame:
    method = str(cfg.get("features", {}).get("monthly_to_quarterly", {}).get("method", "mean"))
    lags = list(cfg.get("features", {}).get("lags_quarters", [1, 2, 4]))

    if method not in {"mean", "last"}:
        raise ValueError("features.monthly_to_quarterly.method must be 'mean' or 'last'")

    q = macro.monthly_to_quarterly(panel_monthly, how=method)

    # Add quarterly lags (safe: positive lags only)
    feat = q.copy()
    for col in q.columns:
        for lag in lags:
            if lag <= 0:
                raise ValueError("lags_quarters must be positive")
            feat[f"{col}_lag{lag}"] = q[col].shift(lag)

    # Merge predictors with GDP growth + labels.
    df = feat.join(gdp_q, how="inner")

    # Drop rows with missing values introduced by lagging / growth calculation.
    df = df.dropna().copy()
    return df


def build_census_county(cfg: dict, *, root: Path) -> pd.DataFrame:
    acs = cfg["acs"]
    year = int(acs["year"])
    dataset = str(acs.get("dataset", "acs/acs5"))
    get_vars = list(acs.get("get", []))
    geo_for = str(acs["geography"]["for"])
    geo_in = str(acs["geography"].get("in")) if acs["geography"].get("in") else None

    raw_dir = root / "data" / "raw" / "census"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = raw_dir / f"acs_county_{year}.csv"
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
    else:
        df = census_api.fetch_acs(year=year, dataset=dataset, get=get_vars, for_geo=geo_for, in_geo=geo_in)
        df.to_csv(raw_csv, index=False)

    # Derived rates (safe guards included).
    def safe_div(num, den):
        den = den.replace({0: pd.NA})
        return (num / den).astype(float)

    if {"B23025_002E", "B23025_005E"}.issubset(df.columns):
        df["unemployment_rate"] = safe_div(df["B23025_005E"].astype(float), df["B23025_002E"].astype(float))

    if {"B17001_002E", "B01003_001E"}.issubset(df.columns):
        df["poverty_rate"] = safe_div(df["B17001_002E"].astype(float), df["B01003_001E"].astype(float))

    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recession-config", required=True)
    parser.add_argument("--census-config", required=True)
    args = parser.parse_args()

    root = project_root()
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    recession_cfg = load_yaml(args.recession_config)
    census_cfg = load_yaml(args.census_config)

    panel_monthly = build_macro_monthly_panel(recession_cfg, root=root)
    data_utils.save_csv(panel_monthly, processed_dir / "panel_monthly.csv")

    gdp_q = build_gdp_quarterly(recession_cfg, root=root)
    data_utils.save_csv(gdp_q, processed_dir / "gdp_quarterly.csv")

    macro_q = build_macro_quarterly_table(recession_cfg, root=root, panel_monthly=panel_monthly, gdp_q=gdp_q)
    data_utils.save_csv(macro_q, processed_dir / "macro_quarterly.csv")

    census_df = build_census_county(census_cfg, root=root)
    year = int(census_cfg["acs"]["year"])
    data_utils.save_csv(census_df.set_index(["state", "county"], drop=False), processed_dir / f"census_county_{year}.csv")

    print("Wrote processed datasets to", processed_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
