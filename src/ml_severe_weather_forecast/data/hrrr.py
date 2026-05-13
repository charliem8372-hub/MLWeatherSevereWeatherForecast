"""HRRR data acquisition via herbie + GRIB2 variable extraction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import xarray as xr

from ml_severe_weather_forecast.data.grid import Grid

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class HRRRVar:
    """Spec for one HRRR variable.

    `grib_search` is a regex used by herbie to subset by GRIB2 message.
    `aggregate` is the canonical temporal aggregation when assembling features.
    """

    name: str
    grib_search: str
    aggregate: tuple[
        str, ...
    ]  # subset of {"max", "mean", "p90"}; or ("max",) for hourly-max fields
    is_hourly_max: bool = False


# Mapping from canonical variable name -> HRRR GRIB2 search pattern.
HRRR_VARIABLES: dict[str, HRRRVar] = {
    "SBCAPE": HRRRVar("SBCAPE", "CAPE:surface", ("max", "mean", "p90")),
    "MLCAPE": HRRRVar("MLCAPE", "CAPE:90-0 mb above ground", ("max", "mean", "p90")),
    "MUCAPE": HRRRVar("MUCAPE", "CAPE:180-0 mb above ground", ("max", "mean", "p90")),
    "SBCIN": HRRRVar("SBCIN", "CIN:surface", ("max", "mean")),
    "MLCIN": HRRRVar("MLCIN", "CIN:90-0 mb above ground", ("max", "mean")),
    "LFTX": HRRRVar("LFTX", "LFTX:500-1000 mb", ("min", "mean")),
    "SRH_0_1km": HRRRVar("SRH_0_1km", "HLCY:1000-0 m above ground", ("max", "mean", "p90")),
    "SRH_0_3km": HRRRVar("SRH_0_3km", "HLCY:3000-0 m above ground", ("max", "mean", "p90")),
    "USHR_0_6km": HRRRVar("USHR_0_6km", "VUCSH:0-6000 m above ground", ("max", "mean")),
    "VSHR_0_6km": HRRRVar("VSHR_0_6km", "VVCSH:0-6000 m above ground", ("max", "mean")),
    "USHR_0_1km": HRRRVar("USHR_0_1km", "VUCSH:0-1000 m above ground", ("max", "mean")),
    "VSHR_0_1km": HRRRVar("VSHR_0_1km", "VVCSH:0-1000 m above ground", ("max", "mean")),
    "PWAT": HRRRVar("PWAT", "PWAT:entire atmosphere", ("max", "mean")),
    "T2M": HRRRVar("T2M", "TMP:2 m above ground", ("max", "mean")),
    "TD2M": HRRRVar("TD2M", "DPT:2 m above ground", ("max", "mean")),
    "U10": HRRRVar("U10", "UGRD:10 m above ground", ("max", "mean")),
    "V10": HRRRVar("V10", "VGRD:10 m above ground", ("max", "mean")),
    "T_500": HRRRVar("T_500", "TMP:500 mb", ("mean",)),
    "T_700": HRRRVar("T_700", "TMP:700 mb", ("mean",)),
    "T_850": HRRRVar("T_850", "TMP:850 mb", ("mean",)),
    "HGT_500": HRRRVar("HGT_500", "HGT:500 mb", ("mean",)),
    "U_500": HRRRVar("U_500", "UGRD:500 mb", ("mean",)),
    "V_500": HRRRVar("V_500", "VGRD:500 mb", ("mean",)),
    "ABSV_500": HRRRVar("ABSV_500", "ABSV:500 mb", ("max", "mean")),
    "U_250": HRRRVar("U_250", "UGRD:250 mb", ("mean",)),
    "V_250": HRRRVar("V_250", "VGRD:250 mb", ("mean",)),
    "HLCY_LCL": HRRRVar("HLCY_LCL", "HGT:level of adiabatic condensation", ("mean",)),
    # Hourly-max storm-attribute fields (only `max` aggregation makes sense)
    "MXUPHL_2_5km": HRRRVar(
        "MXUPHL_2_5km", "MXUPHL:5000-2000 m above ground", ("max",), is_hourly_max=True
    ),
    "MAXWIND_10m": HRRRVar("MAXWIND_10m", "MAXUW:10 m above ground", ("max",), is_hourly_max=True),
    "MAXREFD_1km": HRRRVar(
        "MAXREFD_1km", "MAXREF:1000 m above ground", ("max",), is_hourly_max=True
    ),
    "MAXHAIL": HRRRVar("MAXHAIL", "HAIL:entire atmosphere", ("max",), is_hourly_max=True),
}


def hrrr_variable_search_string() -> str:
    """Return a single regex matching all variables we want from `wrfsfcf` files."""
    return "|".join(v.grib_search for v in HRRR_VARIABLES.values())


def download_cycle(
    cycle_init: datetime,
    forecast_hours: Iterable[int],
    *,
    cache_dir: Path,
) -> list[Path]:
    """Download HRRR `wrfsfcf{FF}.grib2` files for one 12z cycle, subset by variable list.

    Returns the local paths of downloaded files. Herbie handles caching by save_dir,
    so re-running on the same cycle reuses already-downloaded subsets.
    """
    from herbie.core import Herbie

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    search = hrrr_variable_search_string()
    for fxx in forecast_hours:
        h = Herbie(
            date=cycle_init,
            model="hrrr",
            product="sfc",
            fxx=int(fxx),
            save_dir=cache_dir,
        )
        path = h.download(search=search, verbose=False)
        if path is None:
            log.warning("hrrr.download.none", cycle=cycle_init.isoformat(), fxx=fxx)
            continue
        paths.append(Path(path))
    return paths


def _open_grib_subset(path: Path, filter_keys: dict[str, object]) -> xr.Dataset:
    # Lazy import: cfgrib pulls in eccodes C bindings at import time, which we want
    # to defer until a caller actually needs to read GRIB2 data.
    import cfgrib

    ds: xr.Dataset = cfgrib.open_dataset(
        str(path),
        backend_kwargs={"filter_by_keys": filter_keys, "errors": "ignore"},
    )
    return ds


_VAR_TO_FILTER: dict[str, dict[str, object]] = {
    "SBCAPE": {"shortName": "cape", "typeOfLevel": "surface"},
    "MLCAPE": {"shortName": "cape", "typeOfLevel": "pressureFromGroundLayer", "topLevel": 9000},
    "MUCAPE": {"shortName": "cape", "typeOfLevel": "pressureFromGroundLayer", "topLevel": 18000},
    "SBCIN": {"shortName": "cin", "typeOfLevel": "surface"},
    "MLCIN": {"shortName": "cin", "typeOfLevel": "pressureFromGroundLayer", "topLevel": 9000},
    "LFTX": {"shortName": "lftx"},
    "SRH_0_1km": {"shortName": "hlcy", "topLevel": 1000, "bottomLevel": 0},
    "SRH_0_3km": {"shortName": "hlcy", "topLevel": 3000, "bottomLevel": 0},
    "USHR_0_6km": {"shortName": "vucsh", "topLevel": 0, "bottomLevel": 6000},
    "VSHR_0_6km": {"shortName": "vvcsh", "topLevel": 0, "bottomLevel": 6000},
    "USHR_0_1km": {"shortName": "vucsh", "topLevel": 0, "bottomLevel": 1000},
    "VSHR_0_1km": {"shortName": "vvcsh", "topLevel": 0, "bottomLevel": 1000},
    "PWAT": {"shortName": "pwat"},
    "T2M": {"shortName": "2t"},
    "TD2M": {"shortName": "2d"},
    "U10": {"shortName": "10u"},
    "V10": {"shortName": "10v"},
    "T_500": {"shortName": "t", "level": 500, "typeOfLevel": "isobaricInhPa"},
    "T_700": {"shortName": "t", "level": 700, "typeOfLevel": "isobaricInhPa"},
    "T_850": {"shortName": "t", "level": 850, "typeOfLevel": "isobaricInhPa"},
    "HGT_500": {"shortName": "gh", "level": 500, "typeOfLevel": "isobaricInhPa"},
    "U_500": {"shortName": "u", "level": 500, "typeOfLevel": "isobaricInhPa"},
    "V_500": {"shortName": "v", "level": 500, "typeOfLevel": "isobaricInhPa"},
    # NOTE: not in wrfsfcf06; move to wrfprsf in a follow-up
    "ABSV_500": {"shortName": "absv", "level": 500, "typeOfLevel": "isobaricInhPa"},
    "U_250": {"shortName": "u", "level": 250, "typeOfLevel": "isobaricInhPa"},
    "V_250": {"shortName": "v", "level": 250, "typeOfLevel": "isobaricInhPa"},
    "HLCY_LCL": {"shortName": "gh", "typeOfLevel": "adiabaticCondensation"},
    # MXUPHL is in NCEP's local GRIB2 table (not in the standard eccodes table), so
    # cfgrib decodes shortName as "unknown". Identify it by parameterCategory/Number.
    "MXUPHL_2_5km": {
        "parameterCategory": 7,
        "parameterNumber": 199,
        "topLevel": 5000,
        "bottomLevel": 2000,
    },
    # MAXWIND and MAXREFD are also NCEP-local codes; identify by category/number.
    "MAXWIND_10m": {
        "parameterCategory": 2,
        "parameterNumber": 222,
        "typeOfLevel": "heightAboveGround",
        "level": 10,
    },
    "MAXREFD_1km": {
        "parameterCategory": 16,
        "parameterNumber": 198,
        "typeOfLevel": "heightAboveGround",
        "level": 1000,
    },
    "MAXHAIL": {"shortName": "hail"},
}


def extract_variables_to_dataset(grib_path: Path) -> xr.Dataset:
    """Load all configured variables from one GRIB2 file into a single xarray Dataset.

    Variables that fail to load (e.g., not present in this product) are skipped with a warning.
    """
    out_vars: dict[str, xr.DataArray] = {}
    ref_lat: xr.DataArray | None = None
    ref_lon: xr.DataArray | None = None
    for name, filt in _VAR_TO_FILTER.items():
        try:
            ds = _open_grib_subset(grib_path, filt)
        except Exception as exc:
            log.warning("hrrr.extract.skip", var=name, error=str(exc))
            continue
        if not ds.data_vars:
            log.warning("hrrr.extract.empty", var=name, filt=filt)
            continue
        # Pick the first data var (cfgrib sometimes returns the underlying GRIB shortName)
        da = next(iter(ds.data_vars.values()))
        # Drop non-grid scalar coords (e.g., "pressureFromGroundLayer"=9000 vs =18000)
        # so MLCAPE and MUCAPE can coexist in the same Dataset.
        drop_coords = [c for c in da.coords if c not in ("latitude", "longitude")]
        da = da.reset_coords(drop_coords, drop=True)
        out_vars[name] = da.rename(name)
        if ref_lat is None:
            ref_lat = ds["latitude"].reset_coords(
                [c for c in ds["latitude"].coords if c not in ("latitude", "longitude")],
                drop=True,
            )
            ref_lon = ds["longitude"].reset_coords(
                [c for c in ds["longitude"].coords if c not in ("latitude", "longitude")],
                drop=True,
            )
    if not out_vars:
        raise RuntimeError(f"No variables extracted from {grib_path}")
    assert ref_lat is not None and ref_lon is not None
    expected = set(_VAR_TO_FILTER)
    got = set(out_vars)
    missing = sorted(expected - got)
    if missing:
        log.warning("hrrr.extract.missing_vars", count=len(missing), missing=missing)
    return xr.Dataset(out_vars, coords={"latitude": ref_lat, "longitude": ref_lon})


def _assign_hrrr_points_to_cells(ds: xr.Dataset, grid: Grid) -> np.ndarray:
    """For each HRRR (lat, lon) point in `ds`, find the nearest grid cell index.

    Returns an int array of shape (ny_hrrr * nx_hrrr,) — flattened, with values in [0, n_cells)
    or -1 for points outside any cell.
    """
    from sklearn.neighbors import BallTree

    lat = ds["latitude"].to_numpy().ravel()
    lon = ds["longitude"].to_numpy().ravel()
    lon = np.where(lon > 180, lon - 360, lon)
    pts_rad = np.deg2rad(np.column_stack([lat, lon]))

    cell_rad = np.deg2rad(np.column_stack([grid.lats, grid.lons]))
    tree = BallTree(cell_rad, metric="haversine")
    radius_rad = 35_000.0 / 6_371_000.0  # 35 km — covers half of 50 km cell diagonal
    dist, idx = tree.query(pts_rad, k=1)
    flat_idx: np.ndarray = idx.ravel()
    flat_idx[(dist.ravel() > radius_rad)] = -1
    return flat_idx


def regrid_to_cells(ds: xr.Dataset, grid: Grid) -> pd.DataFrame:
    """Aggregate each variable from HRRR's 3 km grid down to the 50 km cell grid.

    For each variable, we compute the cell-wise `max` and (where applicable) `mean`
    over the contained HRRR points.
    """
    cell_idx = _assign_hrrr_points_to_cells(ds, grid)
    order = np.argsort(cell_idx, kind="stable")
    sorted_cell_idx = cell_idx[order]
    valid = sorted_cell_idx >= 0
    sorted_valid_order = order[valid]
    sorted_valid_idx = sorted_cell_idx[valid]
    boundaries = np.concatenate(
        [[0], np.cumsum(np.bincount(sorted_valid_idx, minlength=grid.n_cells))]
    )

    out_columns: dict[str, np.ndarray] = {"cell_id": grid.cell_ids}

    for name, da in ds.data_vars.items():
        flat = da.to_numpy().astype(np.float32).ravel()
        spec = HRRR_VARIABLES.get(str(name))
        var_max = np.full(grid.n_cells, np.nan, dtype=np.float32)
        var_mean = np.full(grid.n_cells, np.nan, dtype=np.float32)
        for c in range(grid.n_cells):
            start, stop = boundaries[c], boundaries[c + 1]
            if stop > start:
                values = flat[sorted_valid_order[start:stop]]
                values = values[~np.isnan(values)]
                if values.size:
                    var_max[c] = values.max()
                    var_mean[c] = values.mean()
        out_columns[f"{name}_max"] = var_max
        if not (spec and spec.is_hourly_max):
            out_columns[f"{name}_mean"] = var_mean

    return pd.DataFrame(out_columns)
