"""Causal inference helpers (panels, DiD/FE, IV) built on linearmodels.

These are intentionally small wrappers. The notebooks should still show how to
use `linearmodels` directly, but these helpers reduce boilerplate and keep
common conventions (panel indexing, FIPS ids) consistent across the curriculum.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import statsmodels.api as sm

try:
    from linearmodels.iv import IV2SLS
    from linearmodels.panel import PanelOLS
except ImportError as exc:  # pragma: no cover
    IV2SLS = None  # type: ignore[assignment]
    PanelOLS = None  # type: ignore[assignment]
    _LINEARMODELS_IMPORT_ERROR = exc
else:  # pragma: no cover
    _LINEARMODELS_IMPORT_ERROR = None


def make_fips(state: str | int, county: str | int) -> str:
    """Create a 5-digit county FIPS code from state + county codes."""

    return str(state).zfill(2) + str(county).zfill(3)


def to_panel_index(df: pd.DataFrame, *, entity_col: str = "fips", time_col: str = "year") -> pd.DataFrame:
    """Ensure a canonical (entity, time) MultiIndex for panel estimators."""

    if isinstance(df.index, pd.MultiIndex) and list(df.index.names[:2]) == [entity_col, time_col]:
        return df.sort_index()

    missing = [c for c in (entity_col, time_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for panel index: {missing}")

    out = df.copy()
    out[entity_col] = out[entity_col].astype(str)
    # Prefer int years when possible; otherwise leave as-is (e.g., dates).
    if time_col in out.columns:
        try:
            out[time_col] = out[time_col].astype(int)
        except (TypeError, ValueError):
            pass

    out = out.set_index([entity_col, time_col], drop=False)
    return out.sort_index()


def fit_twfe_panel_ols(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_cols: Sequence[str],
    entity_effects: bool = True,
    time_effects: bool = True,
    cluster_col: str | None = None,
):
    """Fit a two-way fixed effects panel regression (PanelOLS).

    Notes:
    - This wrapper does not add a constant by default; FE absorb intercepts.
    - If `cluster_col` is provided, uses one-way clustered covariance.
    """

    if PanelOLS is None:  # pragma: no cover
        raise ImportError("linearmodels is required for panel estimators") from _LINEARMODELS_IMPORT_ERROR

    panel = to_panel_index(df)
    cols = [y_col, *list(x_cols)]
    if cluster_col:
        cols.append(cluster_col)
    tmp = panel[cols].dropna().copy()

    y = tmp[y_col].astype(float)
    x = tmp[list(x_cols)].astype(float)

    mod = PanelOLS(y, x, entity_effects=entity_effects, time_effects=time_effects)
    if cluster_col:
        clusters = tmp[cluster_col].astype(str)
        return mod.fit(cov_type="clustered", clusters=clusters)
    return mod.fit(cov_type="robust")


def fit_iv_2sls(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_endog: str,
    x_exog: Iterable[str],
    z_cols: Iterable[str],
    cov_type: str = "robust",
):
    """Fit a 2SLS model using linearmodels IV2SLS.

    Args:
        y_col: outcome column.
        x_endog: endogenous regressor column (single).
        x_exog: exogenous regressors (controls). A constant is automatically added.
        z_cols: instruments for the endogenous regressor.
        cov_type: linearmodels covariance type (e.g., "robust", "unadjusted").
    """

    if IV2SLS is None:  # pragma: no cover
        raise ImportError("linearmodels is required for IV estimators") from _LINEARMODELS_IMPORT_ERROR

    x_exog = list(x_exog)
    z_cols = list(z_cols)
    cols = [y_col, x_endog, *x_exog, *z_cols]
    tmp = df[cols].dropna().copy()

    y = tmp[y_col].astype(float)
    endog = tmp[[x_endog]].astype(float)
    exog = sm.add_constant(tmp[x_exog].astype(float) if x_exog else pd.DataFrame(index=tmp.index), has_constant="add")
    instruments = tmp[z_cols].astype(float)

    mod = IV2SLS(y, exog, endog, instruments)
    return mod.fit(cov_type=cov_type)

