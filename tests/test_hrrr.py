import re

from ml_severe_weather_forecast.data.hrrr import HRRR_VARIABLES, hrrr_variable_search_string


def test_hrrr_variable_list_complete() -> None:
    # Spot check: each of these must be in the variable list (matches spec §5.1)
    expected_subset = {
        "MLCAPE",
        "MUCAPE",
        "MLCIN",
        "SRH_0_1km",
        "SRH_0_3km",
        "USHR_0_6km",
        "VSHR_0_6km",
        "MXUPHL_2_5km",
    }
    assert expected_subset <= set(HRRR_VARIABLES.keys())


def test_search_string_includes_all_vars() -> None:
    s = hrrr_variable_search_string()
    for grib_pattern in ("CAPE", "CIN", "VUCSH", "MXUPHL"):
        assert grib_pattern in s


def test_hourly_max_aggregate_is_max_only() -> None:
    """Hourly-max storm-attribute fields can only be aggregated by max."""
    for v in HRRR_VARIABLES.values():
        if v.is_hourly_max:
            assert v.aggregate == ("max",)


def test_search_string_is_valid_regex() -> None:
    """A future variable with unescaped regex metachars would break Task 18 silently."""
    re.compile(hrrr_variable_search_string())
