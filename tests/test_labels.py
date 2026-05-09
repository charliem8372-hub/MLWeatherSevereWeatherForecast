from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from ml_severe_weather_forecast.data.grid import build_grid, latlon_to_cell_id
from ml_severe_weather_forecast.labels import build_year_labels, label_cycle


def _reports_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True)
    return df


def test_known_tornado_labels_its_cell() -> None:
    grid = build_grid()
    cycle = datetime(2023, 5, 15, 12, tzinfo=UTC)
    norman = (35.222, -97.439)
    reports = _reports_df(
        [
            {
                "event_time_utc": cycle + timedelta(hours=6),
                "lat": norman[0],
                "lon": norman[1],
                "hazard": "tor",
                "magnitude": 2,
            }
        ]
    )
    labels = label_cycle(grid, cycle_init_utc=cycle, reports=reports, hazards=("tor",))
    expected_cell = latlon_to_cell_id(*norman, grid)
    assert labels.loc[labels["cell_id"] == expected_cell, "tor"].iloc[0] == 1


def test_distant_cells_are_zero() -> None:
    grid = build_grid()
    cycle = datetime(2023, 5, 15, 12, tzinfo=UTC)
    reports = _reports_df(
        [
            {
                "event_time_utc": cycle + timedelta(hours=6),
                "lat": 35.222,
                "lon": -97.439,
                "hazard": "tor",
                "magnitude": 2,
            }
        ]
    )
    labels = label_cycle(grid, cycle_init_utc=cycle, reports=reports, hazards=("tor",))
    seattle_cell = latlon_to_cell_id(47.6, -122.3, grid)
    assert labels.loc[labels["cell_id"] == seattle_cell, "tor"].iloc[0] == 0


def test_out_of_window_reports_excluded() -> None:
    grid = build_grid()
    cycle = datetime(2023, 5, 15, 12, tzinfo=UTC)
    reports = _reports_df(
        [
            {
                "event_time_utc": cycle - timedelta(hours=1),
                "lat": 35.222,
                "lon": -97.439,
                "hazard": "tor",
                "magnitude": 2,
            },
            {
                "event_time_utc": cycle + timedelta(hours=25),
                "lat": 35.222,
                "lon": -97.439,
                "hazard": "tor",
                "magnitude": 2,
            },
        ]
    )
    labels = label_cycle(grid, cycle_init_utc=cycle, reports=reports, hazards=("tor",))
    assert labels["tor"].sum() == 0


def test_labels_include_event_count_and_max_magnitude() -> None:
    grid = build_grid()
    cycle = datetime(2023, 5, 15, 12, tzinfo=UTC)
    reports = _reports_df(
        [
            {
                "event_time_utc": cycle + timedelta(hours=2),
                "lat": 35.220,
                "lon": -97.440,
                "hazard": "tor",
                "magnitude": 1,
            },
            {
                "event_time_utc": cycle + timedelta(hours=3),
                "lat": 35.222,
                "lon": -97.441,
                "hazard": "tor",
                "magnitude": 3,
            },
        ]
    )
    labels = label_cycle(grid, cycle_init_utc=cycle, reports=reports, hazards=("tor",))
    cell = latlon_to_cell_id(35.222, -97.439, grid)
    row = labels.loc[labels["cell_id"] == cell].iloc[0]
    assert row["tor"] == 1
    assert row["tor_event_count"] >= 2
    assert row["tor_max_magnitude"] == 3


def test_build_year_labels(tmp_path: Path) -> None:
    grid = build_grid()
    reports = _reports_df(
        [
            {
                "event_time_utc": datetime(2023, 5, 15, 18, tzinfo=UTC),
                "lat": 35.222,
                "lon": -97.439,
                "hazard": "tor",
                "magnitude": 2,
            }
        ]
    )
    reports_path = tmp_path / "2023.parquet"
    reports.to_parquet(reports_path, index=False)

    out = tmp_path / "labels"
    cycles = [datetime(2023, 5, 15, 12, tzinfo=UTC)]
    written = build_year_labels(
        year=2023, cycles=cycles, grid=grid, reports_path=reports_path, out_dir=out
    )
    assert written.exists()
    df = pd.read_parquet(written)
    assert df["tor"].sum() >= 1
    assert "cycle_init_utc" in df.columns
