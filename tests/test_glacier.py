from io import StringIO

import numpy as np

from edb.data.glacier import (
    build_velocity_anomaly,
    filter_glacier_velocity,
    parse_glacier_velocity_dataframe,
    read_glacier_velocity_csv,
    summarize_glacier_points,
)


MOCK_CSV = """glacier_id,point_id,time,velocity_myr,velocity_error_myr,source
G1,P1,2020-01-01,100,5,demo
G1,P1,2020-03-01,120,6,demo
G1,P1,2020-06-01,90,5,demo
G2,P1,2020-01-01,50,3,demo
"""


def test_parse_glacier_velocity_dataframe() -> None:
    raw = read_glacier_velocity_csv(StringIO(MOCK_CSV))
    out = parse_glacier_velocity_dataframe(raw)

    assert out.shape[0] == 4
    assert out.loc[0, "glacier_id"] == "G1"
    assert out.loc[0, "point_id"] == "P1"


def test_filter_build_anomaly_and_summarize() -> None:
    parsed = parse_glacier_velocity_dataframe(read_glacier_velocity_csv(StringIO(MOCK_CSV)))
    filtered = filter_glacier_velocity(parsed, glacier_id="G1", point_id="P1")
    anomaly = build_velocity_anomaly(filtered)
    summary = summarize_glacier_points(parsed)

    assert filtered.shape[0] == 3
    assert "value" in anomaly.columns
    assert np.isfinite(anomaly["value"]).all()
    assert summary.loc[0, "glacier_id"] == "G1"
