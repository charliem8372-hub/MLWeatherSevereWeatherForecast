import re
from pathlib import Path

import pytest

from ml_severe_weather_forecast.data.hrrr import (
    _VAR_TO_FILTER,
    HRRR_VARIABLES,
    extract_variables_to_dataset,
    hrrr_variable_search_string,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_hrrr.grib2"


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


def test_var_to_filter_covers_all_variables() -> None:
    """Drift guard: every canonical variable must have a cfgrib filter spec."""
    assert set(_VAR_TO_FILTER) == set(HRRR_VARIABLES)


@pytest.mark.skipif(not FIXTURE.exists(), reason="tiny_hrrr.grib2 fixture not present")
def test_extract_variables_returns_dataset() -> None:
    ds = extract_variables_to_dataset(FIXTURE)
    # Headline convective variables present
    for v in ("MLCAPE", "MUCAPE", "SRH_0_3km", "MXUPHL_2_5km"):
        assert v in ds.data_vars
    # Wind shear variables present (regression guard for filter-key bug)
    for v in ("USHR_0_6km", "VSHR_0_6km", "USHR_0_1km", "VSHR_0_1km"):
        assert v in ds.data_vars
    # MLCAPE and MUCAPE coexist (verifies reset_coords logic)
    assert ds["MLCAPE"].dims == ds["MUCAPE"].dims
    # 2D Lambert grid
    assert ds["latitude"].ndim == 2
    assert ds["latitude"].shape == ds["MLCAPE"].shape
