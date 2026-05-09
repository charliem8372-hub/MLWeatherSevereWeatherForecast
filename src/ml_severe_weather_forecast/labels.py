"""Spatiotemporal join: SPC storm reports → grid cells → binary labels."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ml_severe_weather_forecast.config import settings
from ml_severe_weather_forecast.data.grid import Grid

EARTH_RADIUS_M = 6_371_000.0


def label_cycle(
    grid: Grid,
    cycle_init_utc: datetime,
    reports: pd.DataFrame,
    hazards: tuple[str, ...] = ("tor", "hail", "wind"),
    *,
    forecast_window_hours: int = 24,
    radius_km: float | None = None,
) -> pd.DataFrame:
    """Generate per-cell binary labels for one forecast cycle.

    Returns a DataFrame keyed by cell_id with columns:
      - cycle_init_utc (constant)
      - cell_id
      - {hazard} (0/1) for each hazard
      - {hazard}_event_count, {hazard}_max_magnitude
    """
    if cycle_init_utc.tzinfo is None:
        raise ValueError("cycle_init_utc must be timezone-aware (UTC)")
    if reports["event_time_utc"].dt.tz is None:
        raise ValueError("reports['event_time_utc'] must be timezone-aware (UTC)")
    if radius_km is None:
        radius_km = settings.label_radius_km
    radius_rad = (radius_km * 1000.0) / EARTH_RADIUS_M
    window_end = cycle_init_utc + timedelta(hours=forecast_window_hours)

    cell_rad = np.deg2rad(np.column_stack([grid.lats, grid.lons]))
    tree = BallTree(cell_rad, metric="haversine")

    out = pd.DataFrame({"cycle_init_utc": cycle_init_utc, "cell_id": grid.cell_ids})

    for hazard in hazards:
        sub = reports[
            (reports["hazard"] == hazard)
            & (reports["event_time_utc"] >= cycle_init_utc)
            & (reports["event_time_utc"] < window_end)
        ]
        label_col = np.zeros(grid.n_cells, dtype=np.int8)
        count_col = np.zeros(grid.n_cells, dtype=np.int32)
        magmax_col = np.full(grid.n_cells, np.nan, dtype=np.float32)
        if not sub.empty:
            r_rad = np.deg2rad(sub[["lat", "lon"]].to_numpy())
            idx_per_report = tree.query_radius(r_rad, r=radius_rad)
            mags = sub["magnitude"].to_numpy(dtype=np.float32)
            for i, hits in enumerate(idx_per_report):
                for cell_idx in hits:
                    label_col[cell_idx] = 1
                    count_col[cell_idx] += 1
                    cur = magmax_col[cell_idx]
                    if np.isnan(cur) or mags[i] > cur:
                        magmax_col[cell_idx] = mags[i]
        out[hazard] = label_col
        out[f"{hazard}_event_count"] = count_col
        out[f"{hazard}_max_magnitude"] = magmax_col

    return out
