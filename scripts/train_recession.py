"""Train a recession classifier and write artifacts to outputs/.

Usage:
  python scripts/train_recession.py --config configs/recession.yaml

Requires:
  data/processed/macro_quarterly.csv (build with scripts/build_datasets.py)

Writes:
  outputs/<run_id>/model.joblib
  outputs/<run_id>/metrics.json
  outputs/<run_id>/predictions.csv
  outputs/<run_id>/run_metadata.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import data as data_utils
from src import evaluation

from scripts._common import load_yaml, project_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None, help="Path to macro_quarterly.csv")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = project_root()

    data_path = Path(args.data) if args.data else root / "data" / "processed" / "macro_quarterly.csv"
    df = data_utils.load_csv(data_path)

    target = "target_recession_next_q"
    if target not in df.columns:
        raise RuntimeError(f"Missing target column {target} in {data_path}")

    # Default feature selection: exclude label and GDP-derived columns to avoid triviality.
    drop_cols = {target, "recession", "GDPC1", "gdp_growth_qoq", "gdp_growth_qoq_annualized", "gdp_growth_yoy"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    x = df[feature_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=int)

    split = evaluation.time_train_test_split_index(len(df), test_size=float(cfg.get("split", {}).get("test_size", 0.2)))
    x_train, x_test = x[split.train_slice], x[split.test_slice]
    y_train, y_test = y[split.train_slice], y[split.test_slice]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000)),
        ]
    )
    model.fit(x_train, y_train)

    prob_test = model.predict_proba(x_test)[:, 1]
    metrics = evaluation.classification_metrics(y_test, prob_test, threshold=float(cfg.get("model", {}).get("threshold", 0.5)))

    # Save artifacts
    run_id = data_utils.run_id_timestamp()
    out_dir = root / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "feature_cols": feature_cols}, out_dir / "model.joblib")

    data_utils.cache_json(out_dir / "metrics.json", metrics)

    # Save predictions for the full dataset.
    prob_all = model.predict_proba(x)[:, 1]
    pred_df = pd.DataFrame({"date": df.index, "prob_recession_next_q": prob_all, "target": y})
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    data_utils.write_run_metadata(
        out_dir,
        {
            "run_id": run_id,
            "data_path": str(data_path),
            "data_sha256": data_utils.sha256_file(data_path),
            "config_path": str(args.config),
            "n_rows": int(len(df)),
            "n_features": int(len(feature_cols)),
        },
    )

    print("Wrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
