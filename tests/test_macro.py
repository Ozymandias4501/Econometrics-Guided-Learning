import numpy as np
import pandas as pd

from src import macro


def test_gdp_growth_qoq():
    idx = pd.date_range("2020-01-01", periods=3, freq="QE")
    s = pd.Series([100.0, 110.0, 121.0], index=idx)
    g = macro.gdp_growth_qoq(s)
    assert np.isnan(g.iloc[0])
    assert abs(g.iloc[1] - 10.0) < 1e-9
    assert abs(g.iloc[2] - 10.0) < 1e-9


def test_technical_recession_label():
    idx = pd.date_range("2020-01-01", periods=5, freq="QE")
    growth = pd.Series([-1.0, -2.0, 1.0, -1.0, -1.0], index=idx)
    rec = macro.technical_recession_label(growth)
    assert rec.tolist() == [0, 1, 0, 0, 1]


def test_monthly_to_quarterly_last_and_mean():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}, index=idx)

    q_last = macro.monthly_to_quarterly(df, how="last")
    q_mean = macro.monthly_to_quarterly(df, how="mean")

    # Q1: Jan-Mar, Q2: Apr-Jun
    assert q_last["x"].tolist() == [3, 6]
    assert q_mean["x"].tolist() == [2.0, 5.0]
