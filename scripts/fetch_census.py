"""Fetch and cache US Census (ACS) data.

Usage:
  python scripts/fetch_census.py --config configs/census.yaml
  python scripts/fetch_census.py --config configs/census_panel.yaml

Writes:
  data/raw/census/variables_<year>.json
  data/raw/census/acs_county_<year>.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import census_api

from scripts._common import load_yaml, project_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    root = project_root()
    out_dir = root / "data" / "raw" / "census"
    out_dir.mkdir(parents=True, exist_ok=True)

    def cache_one_year(*, year: int, dataset: str, get_vars: list[str], geo_for: str, geo_in: str | None) -> None:
        variables_path = out_dir / f"variables_{year}.json"
        if not variables_path.exists():
            payload = census_api.fetch_variables(year=year, dataset=dataset)
            variables_path.write_text(json.dumps(payload, indent=2))

        data_path = out_dir / f"acs_county_{year}.csv"
        if not data_path.exists():
            df = census_api.fetch_acs(year=year, dataset=dataset, get=get_vars, for_geo=geo_for, in_geo=geo_in)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(data_path, index=False)

    if "acs" in cfg:
        acs = cfg["acs"]
        cache_one_year(
            year=int(acs["year"]),
            dataset=str(acs.get("dataset", "acs/acs5")),
            get_vars=list(acs.get("get", [])),
            geo_for=str(acs["geography"]["for"]),
            geo_in=str(acs["geography"].get("in")) if acs["geography"].get("in") else None,
        )

    if "acs_panel" in cfg:
        acs = cfg["acs_panel"]
        years = list(acs.get("years", []))
        if not years:
            raise ValueError("acs_panel.years must be a non-empty list")
        for y in years:
            cache_one_year(
                year=int(y),
                dataset=str(acs.get("dataset", "acs/acs5")),
                get_vars=list(acs.get("get", [])),
                geo_for=str(acs["geography"]["for"]),
                geo_in=str(acs["geography"].get("in")) if acs["geography"].get("in") else None,
            )

    print(f"Cached ACS data under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
