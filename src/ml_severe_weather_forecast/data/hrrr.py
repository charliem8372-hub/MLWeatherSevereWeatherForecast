"""HRRR data acquisition via herbie + GRIB2 variable extraction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog
import xarray as xr

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
    "USHR_0_6km": {"shortName": "vucsh", "topLevel": 6000, "bottomLevel": 0},
    "VSHR_0_6km": {"shortName": "vvcsh", "topLevel": 6000, "bottomLevel": 0},
    "USHR_0_1km": {"shortName": "vucsh", "topLevel": 1000, "bottomLevel": 0},
    "VSHR_0_1km": {"shortName": "vvcsh", "topLevel": 1000, "bottomLevel": 0},
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
    "MAXWIND_10m": {"shortName": "maxuw"},
    "MAXREFD_1km": {"shortName": "maxref"},
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
            ref_lat = ds["latitude"]
            ref_lon = ds["longitude"]
    if not out_vars:
        raise RuntimeError(f"No variables extracted from {grib_path}")
    assert ref_lat is not None and ref_lon is not None
    return xr.Dataset(out_vars, coords={"latitude": ref_lat, "longitude": ref_lon})
