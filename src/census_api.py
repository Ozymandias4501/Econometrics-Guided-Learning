"""Minimal US Census API client (ACS).

This module focuses on:
- fetching variable metadata (variables.json)
- fetching ACS data tables by geography

Docs: https://api.census.gov/data.html
"""

from __future__ import annotations

import os
import random
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

BASE_URL = "https://api.census.gov/data"
DEFAULT_TIMEOUT = 30


class CensusApiError(RuntimeError):
    pass


def get_api_key(explicit: Optional[str] = None) -> Optional[str]:
    # Census key is optional for basic usage.
    return explicit or os.getenv("CENSUS_API_KEY")


def _request(
    url: str,
    params: Dict[str, str],
    *,
    api_key: Optional[str] = None,
    max_retries: int = 4,
    backoff_seconds: float = 1.0,
):
    params = dict(params)
    key = get_api_key(api_key)
    if key:
        params["key"] = key

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise CensusApiError(f"Census request failed: {exc}") from exc
            time.sleep(backoff_seconds * (2**attempt) + random.random() * 0.25)
            continue

        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt >= max_retries:
                raise CensusApiError(f"Census request failed {resp.status_code}: {resp.text}")
            time.sleep(backoff_seconds * (2**attempt) + random.random() * 0.25)
            continue

        if not resp.ok:
            raise CensusApiError(f"Census request failed {resp.status_code}: {resp.text}")

        # Some endpoints are JSON objects, others are JSON arrays.
        return resp.json()

    raise CensusApiError("Census request failed unexpectedly (retry loop exhausted).")


def variables_url(*, year: int, dataset: str = "acs/acs5") -> str:
    return f"{BASE_URL}/{year}/{dataset}/variables.json"


def data_url(*, year: int, dataset: str = "acs/acs5") -> str:
    return f"{BASE_URL}/{year}/{dataset}"


def fetch_variables(*, year: int, dataset: str = "acs/acs5", api_key: Optional[str] = None) -> Dict:
    url = variables_url(year=year, dataset=dataset)
    return _request(url, {}, api_key=api_key)


def fetch_acs(
    *,
    year: int,
    get: Iterable[str],
    for_geo: str,
    in_geo: Optional[str] = None,
    dataset: str = "acs/acs5",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch ACS data and return a DataFrame.

    Args:
        year: ACS year.
        get: list of variables to request. Include "NAME" if you want a label.
        for_geo: e.g., "county:*"
        in_geo: e.g., "state:*" (optional)
        dataset: usually "acs/acs5".

    Returns:
        DataFrame with columns for requested variables plus geography columns.
    """

    url = data_url(year=year, dataset=dataset)

    get_list = list(get)
    if not get_list:
        raise ValueError("get must contain at least one variable")

    params: Dict[str, str] = {"get": ",".join(get_list), "for": for_geo}
    if in_geo:
        params["in"] = in_geo

    payload = _request(url, params, api_key=api_key)
    if not isinstance(payload, list) or not payload:
        raise CensusApiError("Unexpected Census response shape (expected non-empty list)")

    header = payload[0]
    rows = payload[1:]
    df = pd.DataFrame(rows, columns=header)

    # Cast numeric columns when possible.
    for col in df.columns:
        if col in {"NAME", "state", "county"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df
