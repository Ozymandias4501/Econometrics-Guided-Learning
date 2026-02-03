"""Feature engineering helpers.

These helpers are intentionally simple and explicit so learners can inspect and
modify them in notebooks.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def to_monthly(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Resample a DatetimeIndex DataFrame to month-end."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by datetime.")

    monthly = df.resample("ME").last()
    if method == "ffill":
        monthly = monthly.ffill()
    return monthly


def add_lag_features(df: pd.DataFrame, columns: Iterable[str], lags: Iterable[int]) -> pd.DataFrame:
    """Add lagged features.

    Lag must be positive. Negative/zero lags leak future information.
    """

    out = df.copy()
    for col in columns:
        for lag in lags:
            if lag <= 0:
                raise ValueError("lags must be positive (negative/zero lags leak future information)")
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_diff_features(df: pd.DataFrame, columns: Iterable[str], periods: int = 1) -> pd.DataFrame:
    """Add difference features: x_t - x_{t-k}."""

    if periods <= 0:
        raise ValueError("periods must be positive")

    out = df.copy()
    for col in columns:
        out[f"{col}_diff{periods}"] = out[col].diff(periods)
    return out


def add_pct_change_features(df: pd.DataFrame, columns: Iterable[str], periods: int = 1) -> pd.DataFrame:
    """Add percent change features in percent units."""

    if periods <= 0:
        raise ValueError("periods must be positive")

    out = df.copy()
    for col in columns:
        out[f"{col}_pct_change{periods}"] = out[col].pct_change(periods) * 100.0
    return out


def add_log_diff_features(df: pd.DataFrame, columns: Iterable[str], periods: int = 1) -> pd.DataFrame:
    """Add log-difference features: log(x_t) - log(x_{t-k}).

    Works best for strictly positive series. Non-positive values become NaN.
    """

    if periods <= 0:
        raise ValueError("periods must be positive")

    out = df.copy()
    for col in columns:
        x = out[col].astype(float)
        x = x.where(x > 0)
        out[f"{col}_logdiff{periods}"] = np.log(x).diff(periods)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    columns: Iterable[str],
    windows: Iterable[int],
) -> pd.DataFrame:
    """Add rolling mean and rolling std features."""

    out = df.copy()
    for col in columns:
        for window in windows:
            roll = out[col].rolling(window)
            out[f"{col}_roll{window}_mean"] = roll.mean()
            out[f"{col}_roll{window}_std"] = roll.std()
    return out


def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().copy()
