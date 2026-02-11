import numpy as np
import pandas as pd
import pytest

from src import features


# ── to_monthly ───────────────────────────────────────────────────────────

def test_to_monthly_resamples_daily():
    idx = pd.date_range("2020-01-01", periods=90, freq="D")
    df = pd.DataFrame({"x": range(90)}, index=idx)
    out = features.to_monthly(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert len(out) == 3  # Jan, Feb, Mar


def test_to_monthly_raises_on_non_datetime_index():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="datetime"):
        features.to_monthly(df)


# ── add_lag_features ─────────────────────────────────────────────────────

def test_add_lag_creates_columns():
    df = pd.DataFrame({"x": [10, 20, 30, 40, 50]})
    out = features.add_lag_features(df, ["x"], [1, 2])
    assert "x_lag1" in out.columns
    assert "x_lag2" in out.columns
    assert np.isnan(out["x_lag1"].iloc[0])
    assert out["x_lag1"].iloc[1] == 10


def test_add_lag_rejects_zero_lag():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="positive"):
        features.add_lag_features(df, ["x"], [0])


def test_add_lag_rejects_negative_lag():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="positive"):
        features.add_lag_features(df, ["x"], [-1])


def test_add_lag_does_not_mutate_input():
    df = pd.DataFrame({"x": [1, 2, 3]})
    original_cols = list(df.columns)
    features.add_lag_features(df, ["x"], [1])
    assert list(df.columns) == original_cols


# ── add_diff_features ───────────────────────────────────────────────────

def test_add_diff_computes_first_difference():
    df = pd.DataFrame({"x": [10.0, 13.0, 18.0, 20.0]})
    out = features.add_diff_features(df, ["x"])
    assert np.isnan(out["x_diff1"].iloc[0])
    assert out["x_diff1"].iloc[1] == pytest.approx(3.0)
    assert out["x_diff1"].iloc[2] == pytest.approx(5.0)


def test_add_diff_rejects_zero_periods():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="positive"):
        features.add_diff_features(df, ["x"], periods=0)


# ── add_pct_change_features ─────────────────────────────────────────────

def test_pct_change_returns_percent_units():
    df = pd.DataFrame({"x": [100.0, 110.0, 121.0]})
    out = features.add_pct_change_features(df, ["x"])
    assert out["x_pct_change1"].iloc[1] == pytest.approx(10.0)
    assert out["x_pct_change1"].iloc[2] == pytest.approx(10.0)


def test_pct_change_rejects_negative_periods():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="positive"):
        features.add_pct_change_features(df, ["x"], periods=-1)


# ── add_rolling_features ────────────────────────────────────────────────

def test_rolling_produces_mean_and_std():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = features.add_rolling_features(df, ["x"], [3])
    assert "x_roll3_mean" in out.columns
    assert "x_roll3_std" in out.columns
    assert out["x_roll3_mean"].iloc[2] == pytest.approx(2.0)


# ── add_log_diff_features ───────────────────────────────────────────────

def test_log_diff_positive_series():
    df = pd.DataFrame({"x": [100.0, 110.0, 121.0]})
    out = features.add_log_diff_features(df, ["x"])
    expected = np.log(110.0) - np.log(100.0)
    assert out["x_logdiff1"].iloc[1] == pytest.approx(expected)


def test_log_diff_nonpositive_produces_nan():
    df = pd.DataFrame({"x": [1.0, -1.0, 3.0]})
    out = features.add_log_diff_features(df, ["x"])
    assert np.isnan(out["x_logdiff1"].iloc[1])
    assert np.isnan(out["x_logdiff1"].iloc[2])


# ── drop_na_rows ────────────────────────────────────────────────────────

def test_drop_na_removes_rows_with_nans():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, 6.0]})
    out = features.drop_na_rows(df)
    assert len(out) == 2
    assert not out.isna().any().any()
