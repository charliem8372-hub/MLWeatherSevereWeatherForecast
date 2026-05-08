import numpy as np

from ml_severe_weather_forecast.config import settings
from ml_severe_weather_forecast.data.grid import build_grid, latlon_to_cell_id


def test_build_grid_has_expected_cell_count() -> None:
    grid = build_grid()
    # Expected: roughly 6500 cells over CONUS at 50 km. Lower bound 5000, upper 8000.
    assert 5000 < grid.n_cells < 8000


def test_grid_has_unique_cell_ids() -> None:
    grid = build_grid()
    assert len(set(grid.cell_ids)) == grid.n_cells


def test_grid_centers_are_lat_lon() -> None:
    grid = build_grid()
    assert np.all((grid.lats >= settings.grid_lat_min) & (grid.lats <= settings.grid_lat_max))
    assert np.all((grid.lons >= settings.grid_lon_min) & (grid.lons <= settings.grid_lon_max))


def test_known_city_resolves_to_a_cell() -> None:
    grid = build_grid()
    # Norman, OK — center of tornado alley, definitely inside CONUS
    cell_id = latlon_to_cell_id(35.222, -97.439, grid)
    assert cell_id is not None
    assert cell_id.startswith("c_")


def test_offshore_point_returns_none() -> None:
    grid = build_grid()
    # Middle of Atlantic
    assert latlon_to_cell_id(35.0, -50.0, grid) is None


def test_cell_center_roundtrip() -> None:
    grid = build_grid()
    # Pick the cell containing Norman, OK; lookup its center; that center should map to itself
    cell_id = latlon_to_cell_id(35.222, -97.439, grid)
    assert cell_id is not None
    idx = list(grid.cell_ids).index(cell_id)
    same_cell = latlon_to_cell_id(grid.lats[idx], grid.lons[idx], grid)
    assert same_cell == cell_id
