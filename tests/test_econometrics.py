import numpy as np
import pandas as pd
import pytest

from src import econometrics


# ── design_matrix ────────────────────────────────────────────────────────

def test_design_matrix_adds_constant():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    X = econometrics.design_matrix(df, ["x"], add_const=True)
    assert "const" in X.columns
    assert (X["const"] == 1.0).all()


def test_design_matrix_no_constant():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    X = econometrics.design_matrix(df, ["x"], add_const=False)
    assert "const" not in X.columns


# ── fit_ols ──────────────────────────────────────────────────────────────

def _make_ols_data(n=200, beta=2.0, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 1.0 + beta * x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x}), beta


def test_fit_ols_recovers_slope():
    df, beta_true = _make_ols_data()
    res = econometrics.fit_ols(df, y_col="y", x_cols=["x"])
    assert abs(res.params["x"] - beta_true) < 0.3


def test_fit_ols_returns_results_object():
    df, _ = _make_ols_data()
    res = econometrics.fit_ols(df, y_col="y", x_cols=["x"])
    assert hasattr(res, "params")
    assert hasattr(res, "bse")
    assert hasattr(res, "rsquared")


# ── fit_ols_hc3 ─────────────────────────────────────────────────────────

def test_hc3_se_differs_under_heteroskedasticity():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(1, 10, size=n)
    y = 2.0 * x + rng.normal(scale=x, size=n)  # variance grows with x
    df = pd.DataFrame({"y": y, "x": x})

    res_plain = econometrics.fit_ols(df, y_col="y", x_cols=["x"])
    res_hc3 = econometrics.fit_ols_hc3(df, y_col="y", x_cols=["x"])

    # HC3 SE should differ from naive SE under heteroskedasticity
    # robust results return bse as numpy array; index by position (1 = x)
    assert res_hc3.bse[1] != pytest.approx(res_plain.bse["x"], rel=0.01)


# ── fit_ols_hac ──────────────────────────────────────────────────────────

def test_hac_se_differs_under_autocorrelation():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    # AR(1) errors with rho=0.7
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + rng.normal()
    y = 1.0 + 2.0 * x + e
    df = pd.DataFrame({"y": y, "x": x})

    res_plain = econometrics.fit_ols(df, y_col="y", x_cols=["x"])
    res_hac = econometrics.fit_ols_hac(df, y_col="y", x_cols=["x"], maxlags=4)

    # HAC SE should typically be larger when errors are autocorrelated
    # robust results return bse as numpy array; index by position (1 = x)
    assert res_hac.bse[1] != pytest.approx(res_plain.bse["x"], rel=0.01)


def test_hac_recovers_slope():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    y = 1.0 + 3.0 * x + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "x": x})

    res = econometrics.fit_ols_hac(df, y_col="y", x_cols=["x"], maxlags=2)
    # robust results return params as numpy array; index by position (1 = x)
    assert abs(res.params[1] - 3.0) < 0.3


# ── vif_table ────────────────────────────────────────────────────────────

def test_vif_uncorrelated_near_one():
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    vif = econometrics.vif_table(df, ["x1", "x2"])
    assert set(vif["feature"]) == {"x1", "x2"}
    for _, row in vif.iterrows():
        assert row["vif"] < 2.0  # uncorrelated predictors should have VIF ~ 1


def test_vif_collinear_high():
    rng = np.random.default_rng(4)
    n = 500
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.05, size=n)  # nearly identical
    df = pd.DataFrame({"x1": x1, "x2": x2})
    vif = econometrics.vif_table(df, ["x1", "x2"])
    for _, row in vif.iterrows():
        assert row["vif"] > 10.0  # highly collinear
