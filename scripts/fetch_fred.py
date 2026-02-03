"""Fetch and cache FRED series JSON.

Usage:
  python scripts/fetch_fred.py --config configs/recession.yaml

Writes:
  data/raw/fred/<SERIES_ID>.json
  data/raw/fred/<SERIES_ID>_meta.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import data as data_utils
from src import fred_api

from scripts._common import load_yaml, project_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    fred_cfg = cfg["fred"]

    start = fred_cfg.get("start_date")
    end = fred_cfg.get("end_date")
    series_ids = list(fred_cfg.get("series", []))
    gdp_series = fred_cfg.get("gdp_series")
    if gdp_series:
        series_ids = [gdp_series, *series_ids]

    root = project_root()
    out_dir = root / "data" / "raw" / "fred"
    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in series_ids:
        json_path = out_dir / f"{sid}.json"
        meta_path = out_dir / f"{sid}_meta.json"

        payload = data_utils.load_or_fetch_json(
            json_path,
            lambda sid=sid: fred_api.fetch_series_observations(sid, start_date=start, end_date=end),
        )
        data_utils.load_or_fetch_json(
            meta_path,
            lambda sid=sid: fred_api.fetch_series_info(sid),
        )

        # Touch a small validation: ensure observations exist.
        if not payload.get("observations"):
            raise RuntimeError(f"No observations returned for {sid}")

    print(f"Cached {len(series_ids)} series under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
