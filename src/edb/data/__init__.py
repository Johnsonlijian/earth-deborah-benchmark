"""Data schemas and loaders for geophysical activity series."""

from edb.data.schema import GeoActivitySeries, to_standard_dataframe
from edb.data.landslide_glc import (
    GLC_LEGACY_CSV_URL,
    build_landslide_count_series,
    filter_glc_records,
    parse_glc_dataframe,
    read_glc_csv,
    summarize_glc_reporting,
)
from edb.data.grdc import (
    build_discharge_series,
    filter_river_records,
    parse_river_dataframe,
    read_river_csv,
    remove_daily_climatology,
    robust_mad_scale,
    summarize_station_coverage,
)
from edb.data.nwis import (
    CFS_TO_M3S,
    NWIS_DAILY_VALUES_URL,
    build_nwis_daily_values_url,
    fetch_nwis_daily_values,
    parse_nwis_daily_values_json,
)
from edb.data.itslive import (
    ITS_LIVE_STAC_URL,
    fetch_itslive_collections,
    fetch_stac_json,
    search_itslive_items,
    summarize_stac_items,
)
from edb.data.glacier import (
    build_velocity_anomaly,
    filter_glacier_velocity,
    parse_glacier_velocity_dataframe,
    read_glacier_velocity_csv,
    summarize_glacier_points,
)
from edb.data.usgs import (
    build_comcat_url,
    build_energy_rate_series,
    build_event_rate_series,
    earthquake_energy_joule,
    fetch_comcat_events,
    fetch_comcat_events_paginated,
    merge_comcat_geojson_pages,
    parse_comcat_geojson,
)

__all__ = [
    "GeoActivitySeries",
    "CFS_TO_M3S",
    "GLC_LEGACY_CSV_URL",
    "ITS_LIVE_STAC_URL",
    "NWIS_DAILY_VALUES_URL",
    "build_comcat_url",
    "build_energy_rate_series",
    "build_event_rate_series",
    "build_nwis_daily_values_url",
    "build_landslide_count_series",
    "build_discharge_series",
    "build_velocity_anomaly",
    "earthquake_energy_joule",
    "filter_glc_records",
    "filter_river_records",
    "filter_glacier_velocity",
    "fetch_comcat_events",
    "fetch_comcat_events_paginated",
    "fetch_itslive_collections",
    "fetch_nwis_daily_values",
    "fetch_stac_json",
    "merge_comcat_geojson_pages",
    "parse_glc_dataframe",
    "parse_comcat_geojson",
    "parse_nwis_daily_values_json",
    "parse_river_dataframe",
    "parse_glacier_velocity_dataframe",
    "read_river_csv",
    "read_glacier_velocity_csv",
    "remove_daily_climatology",
    "read_glc_csv",
    "robust_mad_scale",
    "search_itslive_items",
    "summarize_station_coverage",
    "summarize_stac_items",
    "summarize_glacier_points",
    "summarize_glc_reporting",
    "to_standard_dataframe",
]
