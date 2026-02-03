"""Minimal FRED API client."""
from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
from typing import Dict, Iterable, Optional

import requests
import pandas as pd

BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_TIMEOUT = 30


class FredApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeriesMeta:
    """Parsed subset of FRED series metadata.

    FRED returns a much larger payload. We only keep the fields that matter for
    frequency/units sanity checks and interpretation.
    """

    series_id: str
    title: str
    units: str
    frequency: str
    seasonal_adjustment: str
    last_updated: str


def get_api_key(explicit: Optional[str] = None) -> str:
    api_key = explicit or os.getenv("FRED_API_KEY")
    if not api_key:
        raise FredApiError(
            "Missing FRED_API_KEY. Set it in your environment or pass api_key explicitly."
        )
    return api_key


def _request(
    endpoint: str,
    params: Dict[str, str],
    api_key: Optional[str] = None,
    *,
    max_retries: int = 4,
    backoff_seconds: float = 1.0,
) -> Dict:
    key = get_api_key(api_key)
    url = f"{BASE_URL}/{endpoint}"
    params = dict(params)
    params["api_key"] = key
    params["file_type"] = "json"

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise FredApiError(f"FRED request failed: {exc}") from exc
            sleep_s = backoff_seconds * (2**attempt) + random.random() * 0.25
            time.sleep(sleep_s)
            continue

        # Retry on rate limits and transient server errors.
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt >= max_retries:
                raise FredApiError(f"FRED request failed {resp.status_code}: {resp.text}")
            sleep_s = backoff_seconds * (2**attempt) + random.random() * 0.25
            time.sleep(sleep_s)
            continue

        if not resp.ok:
            raise FredApiError(f"FRED request failed {resp.status_code}: {resp.text}")
        return resp.json()

    raise FredApiError("FRED request failed unexpectedly (retry loop exhausted).")


def fetch_series_info(series_id: str, api_key: Optional[str] = None) -> Dict:
    return _request("series", {"series_id": series_id}, api_key=api_key)


def fetch_series_meta(series_id: str, api_key: Optional[str] = None) -> SeriesMeta:
    payload = fetch_series_info(series_id, api_key=api_key)
    rows = payload.get("seriess", [])
    if not rows:
        raise FredApiError(f"Missing 'seriess' in metadata response for {series_id}")
    row = rows[0]
    return SeriesMeta(
        series_id=series_id,
        title=str(row.get("title", "")),
        units=str(row.get("units", "")),
        frequency=str(row.get("frequency", "")),
        seasonal_adjustment=str(row.get("seasonal_adjustment", "")),
        last_updated=str(row.get("last_updated", "")),
    )


def fetch_series_observations(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict:
    params: Dict[str, str] = {"series_id": series_id}
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date
    return _request("series/observations", params, api_key=api_key)


def fetch_many_observations(
    series_ids: Iterable[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Dict]:
    data: Dict[str, Dict] = {}
    for series_id in series_ids:
        data[series_id] = fetch_series_observations(
            series_id,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )
    return data


def observations_to_frame(payload: Dict, value_name: str) -> pd.DataFrame:
    observations = payload.get("observations", [])
    df = pd.DataFrame(observations)
    if df.empty:
        return pd.DataFrame(columns=[value_name])
    df["date"] = pd.to_datetime(df["date"])
    df[value_name] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")[[value_name]].sort_index()
