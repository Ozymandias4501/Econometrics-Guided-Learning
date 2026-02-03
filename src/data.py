"""Data loading, caching, and validation utilities."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def load_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text())


def load_or_fetch_json(
    path: str | Path,
    fetch_fn: Callable[[], Dict],
) -> Dict:
    path = Path(path)
    if path.exists():
        return load_json(path)
    payload = fetch_fn()
    cache_json(path, payload)
    return payload


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_id_timestamp() -> str:
    """Generate a stable run id for outputs (UTC timestamp)."""

    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def write_run_metadata(out_dir: str | Path, payload: Dict) -> None:
    """Write a small metadata JSON for a run."""

    out_dir = ensure_dir(out_dir)
    cache_json(out_dir / "run_metadata.json", payload)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def validate_no_missing(df: pd.DataFrame, columns: Optional[list[str]] = None) -> None:
    cols = columns or list(df.columns)
    missing = df[cols].isna().sum()
    if (missing > 0).any():
        raise ValueError(f"Missing values detected:\n{missing[missing > 0]}")
