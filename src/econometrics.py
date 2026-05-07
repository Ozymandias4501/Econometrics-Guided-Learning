"""Econometrics/inference helpers built on statsmodels.

These utilities are intentionally small wrappers. The notebooks should still show
users how to use statsmodels directly, but these helpers reduce boilerplate.

Focus areas:
- OLS regression
- robust standard errors (HC3 for cross-section, HAC/Newey-West for time series)
- multicollinearity (VIF)
- stationarity tests (ADF, KPSS)
- structural break tests (Chow)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as _scipy_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss


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


# ── Stationarity tests ──────────────────────────────────────────────────


@dataclass
class StationarityResult:
    """Compact summary of a single stationarity test."""

    test: str
    null_hypothesis: str
    statistic: float
    pvalue: float
    used_lag: int
    nobs: int
    critical_values: dict
    reject_at_5pct: bool

    def as_dict(self) -> dict:
        return {
            "test": self.test,
            "null": self.null_hypothesis,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "used_lag": self.used_lag,
            "nobs": self.nobs,
            "critical_5pct": float(self.critical_values.get("5%", float("nan"))),
            "reject_at_5pct": self.reject_at_5pct,
        }


def adf_test(
    series: pd.Series,
    *,
    regression: str = "c",
    autolag: Optional[str] = "AIC",
    maxlag: Optional[int] = None,
) -> StationarityResult:
    """Augmented Dickey-Fuller test.

    Null: the series has a unit root (i.e., is non-stationary).
    Reject the null (small p-value) → evidence the series is stationary.

    Args:
        regression: "c" (constant), "ct" (constant + trend), "ctt", or "n" (none).
        autolag: criterion for lag selection ("AIC", "BIC", "t-stat") or None to use maxlag.
        maxlag: optional upper bound on lag length.
    """

    s = pd.Series(series).astype(float).dropna()
    stat, pvalue, used_lag, nobs, crit, _ = adfuller(
        s, regression=regression, autolag=autolag, maxlag=maxlag
    )
    return StationarityResult(
        test="ADF",
        null_hypothesis="unit root (non-stationary)",
        statistic=float(stat),
        pvalue=float(pvalue),
        used_lag=int(used_lag),
        nobs=int(nobs),
        critical_values={k: float(v) for k, v in crit.items()},
        reject_at_5pct=bool(pvalue < 0.05),
    )


def kpss_test(
    series: pd.Series,
    *,
    regression: str = "c",
    nlags: str | int = "auto",
) -> StationarityResult:
    """KPSS test (complement to ADF).

    Null: the series IS stationary (around a constant or trend).
    Reject the null (small p-value) → evidence the series is non-stationary.

    Use ADF and KPSS together: agreement is more credible than either alone.
    """

    s = pd.Series(series).astype(float).dropna()
    # statsmodels emits a warning when p-value is at the boundary; that is fine for teaching.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, used_lag, crit = kpss(s, regression=regression, nlags=nlags)

    return StationarityResult(
        test="KPSS",
        null_hypothesis="stationary",
        statistic=float(stat),
        pvalue=float(pvalue),
        used_lag=int(used_lag),
        nobs=int(len(s)),
        critical_values={k: float(v) for k, v in crit.items()},
        reject_at_5pct=bool(pvalue < 0.05),
    )


def stationarity_table(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    regression: str = "c",
) -> pd.DataFrame:
    """Run ADF and KPSS on each column and return a single tidy table."""

    rows = []
    for col in columns:
        adf = adf_test(df[col], regression=regression)
        kp = kpss_test(df[col], regression=regression)
        rows.append({
            "series": col,
            "adf_stat": adf.statistic,
            "adf_pvalue": adf.pvalue,
            "adf_reject_unit_root_5pct": adf.reject_at_5pct,
            "kpss_stat": kp.statistic,
            "kpss_pvalue": kp.pvalue,
            "kpss_reject_stationarity_5pct": kp.reject_at_5pct,
            "verdict": _stationarity_verdict(adf, kp),
        })
    return pd.DataFrame(rows)


def _stationarity_verdict(adf: StationarityResult, kp: StationarityResult) -> str:
    if adf.reject_at_5pct and not kp.reject_at_5pct:
        return "stationary"
    if not adf.reject_at_5pct and kp.reject_at_5pct:
        return "non-stationary (unit root)"
    if adf.reject_at_5pct and kp.reject_at_5pct:
        return "conflicting (likely difference- or trend-stationary)"
    return "inconclusive (low power)"


# ── Structural break test ───────────────────────────────────────────────


@dataclass
class ChowResult:
    """Chow test for a single known break point."""

    f_statistic: float
    pvalue: float
    df_num: int
    df_denom: int
    rss_pooled: float
    rss_pre: float
    rss_post: float
    n_pre: int
    n_post: int
    k_params: int

    def as_dict(self) -> dict:
        return {
            "f_statistic": self.f_statistic,
            "pvalue": self.pvalue,
            "df_num": self.df_num,
            "df_denom": self.df_denom,
            "rss_pooled": self.rss_pooled,
            "rss_pre": self.rss_pre,
            "rss_post": self.rss_post,
        }


def chow_test(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_cols: Sequence[str],
    break_index,
    add_const: bool = True,
) -> ChowResult:
    """Chow test for a structural break at a known index value.

    Null: coefficients are equal in the pre- and post-break sub-samples.

    `break_index` is the first observation of the *post* sub-sample.
    For a DatetimeIndex, pass a Timestamp; for a positional index, an int.
    """

    s = df[[y_col] + list(x_cols)].dropna().copy()
    pre = s.loc[s.index < break_index]
    post = s.loc[s.index >= break_index]

    n_pre, n_post = len(pre), len(post)
    k = len(x_cols) + (1 if add_const else 0)
    if n_pre <= k or n_post <= k:
        raise ValueError(
            f"Each sub-sample needs more than {k} observations "
            f"(got pre={n_pre}, post={n_post})."
        )

    res_pool = fit_ols(s, y_col=y_col, x_cols=x_cols, add_const=add_const)
    res_pre = fit_ols(pre, y_col=y_col, x_cols=x_cols, add_const=add_const)
    res_post = fit_ols(post, y_col=y_col, x_cols=x_cols, add_const=add_const)

    rss_pool = float(np.sum(res_pool.resid**2))
    rss_pre = float(np.sum(res_pre.resid**2))
    rss_post = float(np.sum(res_post.resid**2))

    df_num = k
    df_denom = n_pre + n_post - 2 * k
    f_stat = ((rss_pool - (rss_pre + rss_post)) / df_num) / (
        (rss_pre + rss_post) / df_denom
    )
    pvalue = float(1.0 - _scipy_stats.f.cdf(f_stat, df_num, df_denom))

    return ChowResult(
        f_statistic=float(f_stat),
        pvalue=pvalue,
        df_num=df_num,
        df_denom=df_denom,
        rss_pooled=rss_pool,
        rss_pre=rss_pre,
        rss_post=rss_post,
        n_pre=n_pre,
        n_post=n_post,
        k_params=k,
    )
