"""HRRR data acquisition via herbie + GRIB2 variable extraction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

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
