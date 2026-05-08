"""50 km Lambert Conformal Conic grid over CONUS."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
from pyproj import CRS, Transformer

from ml_severe_weather_forecast.config import settings


@lru_cache(maxsize=1)
def lcc_crs() -> CRS:
    return CRS.from_proj4(
        f"+proj=lcc +lat_0={settings.lcc_lat_0} +lon_0={settings.lcc_lon_0} "
        f"+lat_1={settings.lcc_lat_1} +lat_2={settings.lcc_lat_2} "
        f"+x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
    )


@dataclass(frozen=True)
class Grid:
    """50 km cell-centered grid over CONUS in Lambert Conformal projection."""

    cell_ids: np.ndarray  # str, shape (N,) — "c_iii_jjj"
    i_index: np.ndarray  # int, shape (N,) — column in projected space
    j_index: np.ndarray  # int, shape (N,) — row in projected space
    x_centers: np.ndarray  # float, shape (N,) — x in meters (LCC)
    y_centers: np.ndarray  # float, shape (N,) — y in meters (LCC)
    lats: np.ndarray  # float, shape (N,)
    lons: np.ndarray  # float, shape (N,)

    @cached_property
    def n_cells(self) -> int:
        return int(self.cell_ids.size)


def build_grid() -> Grid:
    """Construct the canonical 50 km CONUS grid."""
    crs_lcc = lcc_crs()
    crs_geo = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(crs_geo, crs_lcc, always_xy=True)
    inv = Transformer.from_crs(crs_lcc, crs_geo, always_xy=True)

    # Project bbox corners to LCC to find x/y range
    corner_lons = np.array(
        [settings.grid_lon_min, settings.grid_lon_max, settings.grid_lon_min, settings.grid_lon_max]
    )
    corner_lats = np.array(
        [settings.grid_lat_min, settings.grid_lat_min, settings.grid_lat_max, settings.grid_lat_max]
    )
    xs, ys = fwd.transform(corner_lons, corner_lats)

    dx_m = settings.grid_dx_km * 1000.0
    x_min = np.floor(xs.min() / dx_m) * dx_m
    x_max = np.ceil(xs.max() / dx_m) * dx_m
    y_min = np.floor(ys.min() / dx_m) * dx_m
    y_max = np.ceil(ys.max() / dx_m) * dx_m

    nx = round((x_max - x_min) / dx_m)
    ny = round((y_max - y_min) / dx_m)
    assert nx < 1000 and ny < 1000, (
        f"cell-ID format c_iii_jjj only supports indices < 1000 "
        f"(got nx={nx}, ny={ny}); widen the format or shrink the grid"
    )

    # Cell centers (i, j) where i is column (x), j is row (y)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    x_centers = x_min + (ii + 0.5) * dx_m
    y_centers = y_min + (jj + 0.5) * dx_m

    lons, lats = inv.transform(x_centers, y_centers)

    # Mask to CONUS bbox in lat/lon
    inside = (
        (lons >= settings.grid_lon_min)
        & (lons <= settings.grid_lon_max)
        & (lats >= settings.grid_lat_min)
        & (lats <= settings.grid_lat_max)
    )

    i_index = ii[inside].astype(np.int32).ravel()
    j_index = jj[inside].astype(np.int32).ravel()
    x_flat = x_centers[inside].astype(np.float64).ravel()
    y_flat = y_centers[inside].astype(np.float64).ravel()
    lons_flat = lons[inside].astype(np.float64).ravel()
    lats_flat = lats[inside].astype(np.float64).ravel()

    cell_ids = np.array(
        [f"c_{i:03d}_{j:03d}" for i, j in zip(i_index, j_index, strict=True)],
        dtype=object,
    )

    return Grid(
        cell_ids=cell_ids,
        i_index=i_index,
        j_index=j_index,
        x_centers=x_flat,
        y_centers=y_flat,
        lats=lats_flat,
        lons=lons_flat,
    )
