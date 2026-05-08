"""SPC storm-report ingestion and normalization."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

SPC_BASE = "https://www.spc.noaa.gov/wcm"
HAZARD_TO_FILE = {"tor": "{year}_torn.csv", "hail": "{year}_hail.csv", "wind": "{year}_wind.csv"}
HAZARD_VALID = ("tor", "hail", "wind")


def download_spc_year(year: int, hazard: str, dest_dir: Path, *, force: bool = False) -> Path:
    """Download SPC severe-weather database CSV for one year/hazard."""
    if hazard not in HAZARD_VALID:
        raise ValueError(f"hazard must be one of {HAZARD_VALID}, got {hazard!r}")
    filename = HAZARD_TO_FILE[hazard].format(year=year)
    url = f"{SPC_BASE}/{filename}"
    dest = dest_dir / str(year) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


_TZ_OFFSET_HOURS = {3: 6, 9: 0}  # 3=CST, 9=GMT — SPC uses these in `tz`


def _row_to_utc(row: pd.Series) -> datetime:
    """Convert an SPC row's local-tz event time to UTC.

    SPC stores local time in (yr, mo, dy, time) and a timezone code in `tz`
    (3=CST, 9=GMT). The local→UTC offset is added after constructing the
    datetime; the `tzinfo=UTC` tag is incidental — the constructed value
    holds local time, not UTC.
    """
    tz = int(row["tz"])
    offset_hours = _TZ_OFFSET_HOURS.get(tz, 6)
    naive_local = datetime(
        int(row["yr"]),
        int(row["mo"]),
        int(row["dy"]),
        int(str(row["time"]).zfill(6)[:2]),
        int(str(row["time"]).zfill(6)[2:4]),
        int(str(row["time"]).zfill(6)[4:6]),
        tzinfo=UTC,
    )
    return naive_local + timedelta(hours=offset_hours)


def parse_spc_csv(path: Path, hazard: str) -> pd.DataFrame:
    """Parse one SPC severe-weather DB CSV into the canonical schema."""
    if hazard not in HAZARD_VALID:
        raise ValueError(f"hazard must be one of {HAZARD_VALID}, got {hazard!r}")
    df = pd.read_csv(path, comment="#")
    if df.empty:
        return pd.DataFrame(columns=["event_time_utc", "lat", "lon", "hazard", "magnitude"]).astype(
            {"lat": float, "lon": float}
        )
    df = df.copy()
    df["event_time_utc"] = df.apply(_row_to_utc, axis=1)
    out = pd.DataFrame(
        {
            "event_time_utc": df["event_time_utc"],
            "lat": df["slat"].astype(float),
            "lon": df["slon"].astype(float),
            "hazard": hazard,
            "magnitude": df["mag"].astype(float),
        }
    )
    return out


def apply_severity_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply severe-weather severity thresholds.

    - tornado: any EF rating (mag >= -1, where -1 is unrated)
    - hail: >= 1.00 inch (mag >= 1.0)
    - wind: >= 50 kt (mag >= 50; note SPC stores wind mag in knots)
    """
    if df.empty:
        return df
    masks = {
        "tor": df["magnitude"] >= -1,
        "hail": df["magnitude"] >= 1.0,
        "wind": df["magnitude"] >= 50.0,
    }
    keep = pd.Series(False, index=df.index)
    for hazard, mask in masks.items():
        keep |= (df["hazard"] == hazard) & mask
    return df.loc[keep].reset_index(drop=True)


def dedup_reports(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster reports within 5 min x 10 mi; keep the most-severe per cluster."""
    if df.empty:
        return df
    from sklearn.neighbors import BallTree

    df = df.sort_values(["hazard", "event_time_utc"]).reset_index(drop=True)
    keep_mask = pd.Series(True, index=df.index)
    for _hazard, sub in df.groupby("hazard", sort=False):
        if len(sub) < 2:
            continue
        rad = sub[["lat", "lon"]].to_numpy() * (np.pi / 180.0)
        tree = BallTree(rad, metric="haversine")
        radius_rad = 16093.4 / 6_371_000.0  # 10 miles in radians on Earth
        idxs = tree.query_radius(rad, r=radius_rad)
        for i, neighbors in enumerate(idxs):
            if not keep_mask.iloc[sub.index[i]]:
                continue
            for n in neighbors:
                if n == i:
                    continue
                gi, gn = sub.index[i], sub.index[n]
                if not keep_mask.at[gn]:  # already dropped — skip
                    continue
                t_i = df.at[gi, "event_time_utc"]
                t_n = df.at[gn, "event_time_utc"]
                if abs((t_i - t_n).total_seconds()) > 300:
                    continue
                # within 5 min: keep the higher magnitude
                if df.at[gn, "magnitude"] > df.at[gi, "magnitude"]:
                    keep_mask.at[gi] = False
                else:
                    keep_mask.at[gn] = False
    return df.loc[keep_mask].reset_index(drop=True)


def build_reports(
    years: list[int],
    src_dir: Path,
    out_dir: Path,
    hazards: tuple[str, ...] = HAZARD_VALID,
) -> list[Path]:
    """For each year, parse all hazard CSVs, filter, dedup, write one Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for year in years:
        frames: list[pd.DataFrame] = []
        for hazard in hazards:
            csv = src_dir / str(year) / HAZARD_TO_FILE[hazard].format(year=year)
            if not csv.exists():
                continue
            frames.append(parse_spc_csv(csv, hazard=hazard))
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df = apply_severity_filters(df)
        df = dedup_reports(df)
        path = out_dir / f"{year}.parquet"
        df.to_parquet(path, index=False)
        written.append(path)
    return written
