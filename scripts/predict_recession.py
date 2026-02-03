"""Run a trained recession model to produce predictions.

Usage:
  python scripts/predict_recession.py --model outputs/<run_id>/model.joblib

By default uses data/processed/macro_quarterly.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src import data as data_utils

from scripts._common import project_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to outputs/<run_id>/model.joblib")
    parser.add_argument("--data", default=None, help="Path to macro_quarterly.csv")
    parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()

    root = project_root()
    model_path = Path(args.model)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    data_path = Path(args.data) if args.data else root / "data" / "processed" / "macro_quarterly.csv"
    df = data_utils.load_csv(data_path)

    x = df[feature_cols].to_numpy(dtype=float)
    prob = model.predict_proba(x)[:, 1]

    out_path = Path(args.out) if args.out else model_path.parent / "predictions_latest.csv"
    out_df = pd.DataFrame({"date": df.index, "prob_recession_next_q": prob})
    out_df.to_csv(out_path, index=False)

    latest = out_df.iloc[-1]
    print("Wrote", out_path)
    print("Latest date:", latest["date"], "prob:", float(latest["prob_recession_next_q"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
