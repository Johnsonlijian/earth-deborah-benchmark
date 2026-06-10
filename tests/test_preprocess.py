import numpy as np
import pandas as pd

from edb.signal.preprocess import remove_linear_trend, remove_seasonal_cycle, resample_time_series, robust_zscore


def test_robust_zscore_constant_returns_zero() -> None:
    out = robust_zscore([3.0, 3.0, 3.0])

    np.testing.assert_allclose(out, np.zeros(3))


def test_remove_linear_trend_removes_known_line() -> None:
    t = np.arange(10, dtype=float)
    x = 2.0 * t + 5.0

    out = remove_linear_trend(t, x)

    assert abs(float(np.nanmean(out))) < 1e-10
    assert float(np.nanstd(out)) < 1e-10


def test_remove_seasonal_cycle_reduces_repeated_monthly_cycle() -> None:
    dates = pd.date_range("2020-01-01", periods=730, freq="D")
    values = np.sin(dates.dayofyear.to_numpy() / 366 * 2 * np.pi) + 0.5
    df = pd.DataFrame({"time": dates, "value": values})

    out = remove_seasonal_cycle(df, "time", "value", freq="D")

    assert "seasonal_cycle" in out.columns
    assert abs(float(out["value"].median())) < 1e-8


def test_resample_time_series_daily_mean() -> None:
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 12:00", "2020-01-02 00:00"]),
            "value": [1.0, 3.0, 5.0],
        }
    )

    out = resample_time_series(df, "time", "value", "D")

    assert out.shape[0] == 2
    np.testing.assert_allclose(out["value"], [2.0, 5.0])
