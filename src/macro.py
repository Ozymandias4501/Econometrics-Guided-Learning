"""Macro-specific transforms: GDP growth, technical recession labels, and aggregation.

Design goals:
- Keep transformations explicit and easy to inspect.
- Prefer simple, interpretable definitions suitable for teaching.

Notes:
- GDP series (GDPC1) is quarterly.
- Many predictor indicators are monthly or daily and must be aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


def gdp_growth_qoq(levels: pd.Series) -> pd.Series:
    """Quarter-over-quarter growth rate in percent."""
    return 100.0 * (levels / levels.shift(1) - 1.0)


def gdp_growth_qoq_annualized(levels: pd.Series) -> pd.Series:
    """Annualized QoQ growth (assuming quarterly compounding)."""
    return 100.0 * ((levels / levels.shift(1)) ** 4 - 1.0)


def gdp_growth_yoy(levels: pd.Series) -> pd.Series:
    """Year-over-year growth rate in percent (4-quarter change)."""
    return 100.0 * (levels / levels.shift(4) - 1.0)


def technical_recession_label(growth_qoq: pd.Series) -> pd.Series:
    """Technical recession label from GDP growth.

    Definition used in this project:
    - recession_t = 1 if growth_t < 0 AND growth_{t-1} < 0

    This is a teaching proxy, not an official recession dating rule.
    """

    return ((growth_qoq < 0) & (growth_qoq.shift(1) < 0)).astype(int)


def next_period_target(label: pd.Series) -> pd.Series:
    """Predict next period's label."""
    return label.shift(-1)


QuarterAgg = Literal["last", "mean"]


def monthly_to_quarterly(df: pd.DataFrame, *, how: QuarterAgg) -> pd.DataFrame:
    """Aggregate a monthly DataFrame to quarterly.

    Args:
        how:
            - "last": quarter-end value
            - "mean": quarter-average value
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    if how == "last":
        return df.resample("QE").last()
    if how == "mean":
        return df.resample("QE").mean()

    raise ValueError(f"Unsupported how={how}")
