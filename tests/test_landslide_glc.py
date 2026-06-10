from io import StringIO

import numpy as np

from edb.data.landslide_glc import (
    build_landslide_count_series,
    filter_glc_records,
    parse_glc_dataframe,
    read_glc_csv,
    summarize_glc_reporting,
)


MOCK_CSV = """source_name,source_link,event_id,event_date,event_time,event_title,event_description,location_description,location_accuracy,landslide_category,landslide_trigger,landslide_size,landslide_setting,fatality_count,injury_count,storm_name,photo_link,notes,event_import_source,event_import_id,country_name,country_code,admin_division_name,admin_division_population,gazeteer_closest_point,gazeteer_distance,submitted_date,created_date,last_edited_date,longitude,latitude
AGU,https://example.org/1,684,08/01/2008 12:00:00 AM,,Title 1,Description,Location,unknown,landslide,rain,large,mine,11,,,,,glc,684,China,CN,Shaanxi,0,Jingyang,41.0,04/01/2014 12:00:00 AM,11/20/2017 03:17:00 PM,02/15/2018 03:51:00 PM,107.45,32.5625
Oregonian,https://example.org/2,956,01/02/2009 02:00:00 AM,,Title 2,Description,Location,5km,mudslide,downpour,small,unknown,0,,,,,glc,956,United States,US,Oregon,36619,Lake Oswego,0.6,04/01/2014 12:00:00 AM,11/20/2017 03:17:00 PM,02/15/2018 03:51:00 PM,-122.663,45.42
CBS News,https://example.org/3,973,01/19/2007 12:00:00 AM,,Title 3,Description,Location,10km,landslide,downpour,large,unknown,10,,,,,glc,973,Peru,PE,Junin,14708,San Ramon,0.9,04/01/2014 12:00:00 AM,11/20/2017 03:17:00 PM,02/15/2018 03:51:00 PM,-75.3587,-11.1295
"""


def test_read_and_parse_glc_csv() -> None:
    raw = read_glc_csv(StringIO(MOCK_CSV))
    out = parse_glc_dataframe(raw)

    assert out.shape[0] == 3
    assert out.loc[0, "event_id"] == 973
    assert out.loc[0, "event_year"] == 2007
    assert out.loc[1, "country_code"] == "CN"
    assert np.isclose(out.loc[1, "longitude"], 107.45)


def test_filter_glc_records_by_start_country_trigger() -> None:
    parsed = parse_glc_dataframe(read_glc_csv(StringIO(MOCK_CSV)))

    out = filter_glc_records(parsed, start_date="2008-01-01", country_code="CN", landslide_trigger="rain")

    assert out.shape[0] == 1
    assert out.loc[0, "event_id"] == 684


def test_build_landslide_count_series_monthly() -> None:
    parsed = parse_glc_dataframe(read_glc_csv(StringIO(MOCK_CSV)))

    out = build_landslide_count_series(parsed, rule="MS")

    assert list(out.columns) == ["time", "landslide_count", "value"]
    assert out["landslide_count"].sum() == 3


def test_summarize_glc_reporting() -> None:
    parsed = parse_glc_dataframe(read_glc_csv(StringIO(MOCK_CSV)))

    out = summarize_glc_reporting(parsed)

    assert set(out.columns) == {"event_year", "n_events", "n_countries", "n_triggers"}
    assert out["n_events"].sum() == 3
