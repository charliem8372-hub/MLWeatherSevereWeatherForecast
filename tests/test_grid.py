import numpy as np

from ml_severe_weather_forecast.data.grid import build_grid


def test_build_grid_has_expected_cell_count() -> None:
    grid = build_grid()
    # Expected: roughly 6500 cells over CONUS at 50 km. Lower bound 5000, upper 8000.
    assert 5000 < grid.n_cells < 8000


def test_grid_has_unique_cell_ids() -> None:
    grid = build_grid()
    assert len(set(grid.cell_ids)) == grid.n_cells


def test_grid_centers_are_lat_lon() -> None:
    grid = build_grid()
    assert np.all((grid.lats >= 23) & (grid.lats <= 51))
    assert np.all((grid.lons >= -126) & (grid.lons <= -64))
