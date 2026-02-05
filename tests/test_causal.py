import numpy as np
import pandas as pd

from src import causal


def test_to_panel_index_sets_multiindex_and_preserves_rows():
    df = pd.DataFrame(
        {
            "fips": ["01001", "01001", "06059"],
            "year": [2019, 2020, 2019],
            "y": [1.0, 2.0, 3.0],
        }
    )

    out = causal.to_panel_index(df)
    assert isinstance(out.index, pd.MultiIndex)
    assert out.index.names[:2] == ["fips", "year"]
    assert len(out) == len(df)


def test_fit_twfe_panel_ols_runs_and_recovers_slope():
    import pytest

    pytest.importorskip("linearmodels")

    rng = np.random.default_rng(0)

    fips = [str(i).zfill(5) for i in range(30)]
    years = list(range(2010, 2018))

    rows = []
    time_fe = {y: 0.05 * (y - years[0]) for y in years}
    entity_fe = {f: rng.normal(scale=0.5) for f in fips}

    beta_true = 2.0
    for f in fips:
        for y in years:
            x = rng.normal()
            eps = rng.normal(scale=0.1)
            y_val = beta_true * x + entity_fe[f] + time_fe[y] + eps
            rows.append({"fips": f, "year": y, "x": x, "y": y_val})

    df = pd.DataFrame(rows)
    res = causal.fit_twfe_panel_ols(df, y_col="y", x_cols=["x"], entity_effects=True, time_effects=True)

    assert "x" in res.params.index
    assert np.isfinite(res.params["x"])
    assert abs(float(res.params["x"]) - beta_true) < 0.25


def test_fit_iv_2sls_runs_and_returns_finite_param():
    import pytest

    pytest.importorskip("linearmodels")

    rng = np.random.default_rng(1)
    n = 4000

    z = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.8 * z + 0.8 * u + rng.normal(size=n)
    eps = 0.8 * u + rng.normal(size=n)

    beta_true = 1.5
    y = beta_true * x + eps

    df = pd.DataFrame({"y": y, "x": x, "z": z})
    res = causal.fit_iv_2sls(df, y_col="y", x_endog="x", x_exog=[], z_cols=["z"])

    assert "x" in res.params.index
    assert np.isfinite(res.params["x"])
    assert abs(float(res.params["x"]) - beta_true) < 0.25
