"""CMS / Socrata Open Data API client for health economics datasets.

Follows the same patterns as fred_api.py: retry logic, dataclass metadata,
and DataFrame conversion. No API key is required for CMS public datasets.

Primary data source: data.cms.gov (Socrata-powered)
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

CMS_BASE_URL = "https://data.cms.gov/resource"
DEFAULT_TIMEOUT = 30


class CmsApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetMeta:
    """Lightweight metadata for a CMS dataset."""

    dataset_id: str
    title: str
    description: str


# ── Popular CMS dataset identifiers (Socrata 4x4 codes) ─────────────────
# These are stable public endpoints on data.cms.gov.
DATASETS = {
    "inpatient_charges": DatasetMeta(
        dataset_id="97k6-zzx3",
        title="Medicare Inpatient Hospitals — by Provider and Service",
        description="Average covered charges, total payments, and Medicare payments "
        "for the top DRGs by participating hospital.",
    ),
    "outpatient_charges": DatasetMeta(
        dataset_id="fyjz-6xak",
        title="Medicare Outpatient Hospitals — by Provider and Service",
        description="Average estimated submitted charges and Medicare payments "
        "for outpatient APCs by hospital.",
    ),
    "hospital_spending": DatasetMeta(
        dataset_id="nrth-mfg3",
        title="Medicare Hospital Spending per Patient (MSPB)",
        description="Medicare Spending Per Beneficiary measure comparing "
        "hospital spending to the national median.",
    ),
}


def _request(
    url: str,
    params: Optional[Dict[str, str]] = None,
    *,
    max_retries: int = 4,
    backoff_seconds: float = 1.0,
) -> List[Dict]:
    """GET with retry and exponential backoff. Returns parsed JSON."""

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise CmsApiError(f"CMS request failed: {exc}") from exc
            time.sleep(backoff_seconds * (2**attempt) + random.random() * 0.25)
            continue

        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt >= max_retries:
                raise CmsApiError(f"CMS request failed {resp.status_code}: {resp.text}")
            time.sleep(backoff_seconds * (2**attempt) + random.random() * 0.25)
            continue

        if not resp.ok:
            raise CmsApiError(f"CMS request failed {resp.status_code}: {resp.text}")
        return resp.json()

    raise CmsApiError("CMS request failed unexpectedly (retry loop exhausted).")


def fetch_cms_dataset(
    dataset_id: str,
    *,
    limit: int = 5000,
    offset: int = 0,
    where: Optional[str] = None,
    select: Optional[str] = None,
    order: Optional[str] = None,
) -> List[Dict]:
    """Fetch rows from a CMS Socrata dataset.

    Args:
        dataset_id: Socrata 4x4 identifier (e.g. "97k6-zzx3").
        limit: max rows to return (Socrata default is 1000; max 50000).
        offset: pagination offset.
        where: SoQL WHERE clause (e.g. "state = 'CA'").
        select: SoQL SELECT clause (e.g. "provider_id, avg_total_payments").
        order: SoQL ORDER BY clause.

    Returns:
        List of dicts (one per row).
    """

    url = f"{CMS_BASE_URL}/{dataset_id}.json"
    params: Dict[str, str] = {"$limit": str(limit), "$offset": str(offset)}
    if where:
        params["$where"] = where
    if select:
        params["$select"] = select
    if order:
        params["$order"] = order
    return _request(url, params)


def fetch_cms_to_frame(
    dataset_id: str,
    *,
    limit: int = 5000,
    offset: int = 0,
    where: Optional[str] = None,
    select: Optional[str] = None,
    order: Optional[str] = None,
    numeric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fetch a CMS dataset and convert to DataFrame.

    Args:
        dataset_id: Socrata 4x4 identifier.
        limit: max rows.
        offset: pagination offset.
        where: SoQL WHERE clause.
        select: SoQL SELECT clause.
        order: SoQL ORDER BY clause.
        numeric_cols: columns to coerce to numeric (pd.to_numeric).

    Returns:
        DataFrame with one row per record.
    """

    rows = fetch_cms_dataset(
        dataset_id,
        limit=limit,
        offset=offset,
        where=where,
        select=select,
        order=order,
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def list_datasets() -> Dict[str, DatasetMeta]:
    """Return the registry of known CMS dataset identifiers."""
    return dict(DATASETS)
