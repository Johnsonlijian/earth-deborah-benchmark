from io import StringIO

import numpy as np
import pandas as pd

from edb.data.grdc import (
    build_discharge_series,
    filter_river_records,
    parse_river_dataframe,
    read_river_csv,
    remove_daily_climatology,
    robust_mad_scale,
    summarize_station_coverage,
)


MOCK_CSV = """station_id,station_name,river_name,country_code,time,discharge_m3s,quality_flag
A,Station A,River A,AA,2000-01-01,10,good
A,Station A,River A,AA,2000-01-02,12,good
A,Station A,River A,AA,2001-01-01,14,good
B,Station B,River B,BB,2000-01-01,20,good
"""


def test_read_and_parse_river_csv() -> None:
    raw = read_river_csv(StringIO(MOCK_CSV))
    out = parse_river_dataframe(raw)

    assert out.shape[0] == 4
    assert list(out.columns) == [
        "station_id",
        "station_name",
        "river_name",
        "country_code",
        "time",
        "discharge_m3s",
        "quality_flag",
    ]
    assert out.loc[0, "station_id"] == "A"


def test_filter_and_build_discharge_series() -> None:
    parsed = parse_river_dataframe(read_river_csv(StringIO(MOCK_CSV)))
    filtered = filter_river_records(parsed, station_id="A", start_date="2000-01-01", end_date="2000-12-31")

    out = build_discharge_series(filtered, rule="D")

    assert filtered.shape[0] == 2
    assert out["discharge_m3s"].iloc[0] == 10
    assert out["value"].iloc[1] == 12


def test_remove_daily_climatology() -> None:
    dates = pd.to_datetime(["2000-01-01", "2001-01-01", "2000-01-02", "2001-01-02"])
    df = pd.DataFrame({"time": dates, "value": [10.0, 14.0, 20.0, 24.0]})

    out = remove_daily_climatology(df)

    assert "seasonal_climatology" in out.columns
    np.testing.assert_allclose(out["seasonal_climatology"], [12.0, 12.0, 22.0, 22.0])


def test_robust_mad_scale_and_coverage() -> None:
    parsed = parse_river_dataframe(read_river_csv(StringIO(MOCK_CSV)))
    scaled = robust_mad_scale(pd.DataFrame({"value": [1.0, 2.0, 3.0]}))
    coverage = summarize_station_coverage(parsed)

    assert abs(float(scaled["value"].median())) < 1e-12
    assert set(["station_id", "n_records", "missing_daily_fraction"]).issubset(coverage.columns)
    assert coverage.loc[0, "station_id"] == "A"
