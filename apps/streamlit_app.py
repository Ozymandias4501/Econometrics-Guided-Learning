from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    st.set_page_config(page_title="Recession Capstone", layout="wide")
    st.title("Capstone Dashboard: Recession Prediction")

    root = _project_root()
    outputs_dir = root / "outputs"

    st.sidebar.header("Run Selection")
    if not outputs_dir.exists():
        st.info("No outputs found yet. Train a model first: `python scripts/train_recession.py --config configs/recession.yaml`.")
        return

    run_dirs = sorted([p for p in outputs_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    if not run_dirs:
        st.info("No runs found in outputs/. Train a model first.")
        return

    run_id = st.sidebar.selectbox("Run", [p.name for p in run_dirs])
    run_dir = outputs_dir / run_id

    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "predictions.csv"

    if not preds_path.exists():
        st.error(f"Missing predictions.csv in {run_dir}")
        return

    preds = pd.read_csv(preds_path, parse_dates=["date"])

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Recession Probability (Next Quarter)")
        chart_df = preds.set_index("date")["prob_recession_next_q"]
        st.line_chart(chart_df)

        st.caption("This is the model's predicted probability that the NEXT quarter is a technical recession.")

    with col2:
        st.subheader("Run Metrics")
        if metrics_path.exists():
            st.json(json.loads(metrics_path.read_text()))
        else:
            st.info("No metrics.json found for this run.")

        st.subheader("Latest Prediction")
        latest = preds.sort_values("date").iloc[-1]
        st.metric("Latest prob", f"{latest['prob_recession_next_q']:.3f}")
        st.write("Date:", latest["date"].date())

    st.subheader("Predictions Table")
    st.dataframe(preds.tail(25), use_container_width=True)


if __name__ == "__main__":
    main()
