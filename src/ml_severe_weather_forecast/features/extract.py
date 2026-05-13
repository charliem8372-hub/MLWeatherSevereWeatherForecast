"""Per-cycle feature extraction: 25 forecast-hour DataFrames -> one feature table."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _stack_per_cell(frames: Sequence[pd.DataFrame], col: str) -> np.ndarray:
    """Stack one column across forecast-hour frames into a (n_cells, n_hours) array."""
    arr: np.ndarray = np.column_stack([f[col].to_numpy(dtype=np.float32) for f in frames])
    return arr


def temporal_aggregate(
    frames: Sequence[pd.DataFrame],
    instantaneous_vars: Sequence[str],
    hourly_max_vars: Sequence[str],
) -> pd.DataFrame:
    """Aggregate per-forecast-hour cell tables into a single per-cell feature table.

    For each variable in `instantaneous_vars`, both `_max` and `_mean` columns from the
    regridded frames are aggregated three ways across forecast hours: max, mean, p90.
    For each variable in `hourly_max_vars`, only `_max` is aggregated and only `max` is kept.
    """
    if not frames:
        raise ValueError("frames must be non-empty")
    for i, f in enumerate(frames[1:], 1):
        if not f["cell_id"].equals(frames[0]["cell_id"]):
            raise ValueError(f"frames[{i}] cell_id does not match frames[0]")
    out = pd.DataFrame({"cell_id": frames[0]["cell_id"]})

    for var in instantaneous_vars:
        for stat in ("max", "mean"):
            col = f"{var}_{stat}"
            if col not in frames[0].columns:
                continue
            stacked = _stack_per_cell(frames, col)
            out[f"{col}_fhr_max"] = np.nanmax(stacked, axis=1)
            out[f"{col}_fhr_mean"] = np.nanmean(stacked, axis=1)
            out[f"{col}_fhr_p90"] = np.nanpercentile(stacked, 90, axis=1).astype(np.float32)

    for var in hourly_max_vars:
        col = f"{var}_max"
        if col not in frames[0].columns:
            continue
        stacked = _stack_per_cell(frames, col)
        out[f"{col}_fhr_max"] = np.nanmax(stacked, axis=1)

    return out
