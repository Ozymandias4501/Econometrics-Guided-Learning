"""Econometrics/inference helpers built on statsmodels.

These utilities are intentionally small wrappers. The notebooks should still show
users how to use statsmodels directly, but these helpers reduce boilerplate.

Focus areas:
- OLS regression
- robust standard errors (HC3 for cross-section, HAC/Newey-West for time series)
- multicollinearity (VIF)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def design_matrix(df: pd.DataFrame, x_cols: Iterable[str], *, add_const: bool = True) -> pd.DataFrame:
    x = df[list(x_cols)].astype(float)
    if add_const:
        x = sm.add_constant(x, has_constant="add")
    return x


def fit_ols(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_cols: Iterable[str],
    add_const: bool = True,
):
    """Fit plain OLS (non-robust covariance)."""

    y = df[y_col].astype(float)
    x = design_matrix(df, x_cols, add_const=add_const)
    model = sm.OLS(y, x, missing="drop")
    return model.fit()


def fit_ols_hc3(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_cols: Iterable[str],
    add_const: bool = True,
):
    """Fit OLS and report HC3 robust standard errors (cross-sectional default)."""

    res = fit_ols(df, y_col=y_col, x_cols=x_cols, add_const=add_const)
    return res.get_robustcov_results(cov_type="HC3")


def fit_ols_hac(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_cols: Iterable[str],
    maxlags: int,
    add_const: bool = True,
):
    """Fit OLS and report HAC/Newey-West robust standard errors (time-series default)."""

    res = fit_ols(df, y_col=y_col, x_cols=x_cols, add_const=add_const)
    return res.get_robustcov_results(cov_type="HAC", maxlags=int(maxlags))


def vif_table(df: pd.DataFrame, x_cols: Iterable[str]) -> pd.DataFrame:
    """Compute variance inflation factors (VIF) for multicollinearity checks."""

    x = df[list(x_cols)].astype(float)
    vifs = []
    values = x.to_numpy()
    for i, col in enumerate(x.columns):
        vifs.append({"feature": col, "vif": float(variance_inflation_factor(values, i))})
    return pd.DataFrame(vifs).sort_values("vif", ascending=False)
