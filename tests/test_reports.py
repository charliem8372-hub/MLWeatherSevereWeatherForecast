from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml_severe_weather_forecast.data.reports import dedup_reports, parse_spc_csv

FIXTURE = Path(__file__).parent / "fixtures" / "sample_lsr.csv"


def test_parse_spc_csv_returns_dataframe_with_canonical_schema() -> None:
    df = parse_spc_csv(FIXTURE, hazard="tor")
    assert {"event_time_utc", "lat", "lon", "hazard", "magnitude"} <= set(df.columns)
    assert (df["hazard"] == "tor").all()
    assert df["lat"].dtype == float


def test_parse_spc_csv_handles_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text(
        "om,yr,mo,dy,date,time,tz,st,stf,stn,mag,inj,fat,loss,closs,slat,slon,elat,elon,len,wid,fc\n"
    )
    df = parse_spc_csv(empty, hazard="tor")
    assert len(df) == 0


def test_dedup_does_not_chain_eliminate_through_dead_neighbors() -> None:
    """X(10) — 9mi — A(7) — 9mi — B(5): X drops A; X is 18mi from B; B should survive.

    Regression test for a bug where a live report could be dropped against
    a dead-but-higher-magnitude neighbor.
    """
    t = datetime(2023, 5, 15, 18, 0, tzinfo=UTC)
    # 1 degree of latitude ~= 69 mi, so 0.13 deg ~= 9 mi.
    df = pd.DataFrame(
        [
            {"event_time_utc": t, "lat": 35.000, "lon": -97.000, "hazard": "tor", "magnitude": 10},
            {"event_time_utc": t, "lat": 35.130, "lon": -97.000, "hazard": "tor", "magnitude": 7},
            {"event_time_utc": t, "lat": 35.260, "lon": -97.000, "hazard": "tor", "magnitude": 5},
        ]
    )
    out = dedup_reports(df)
    mags = sorted(out["magnitude"].tolist())
    assert mags == [5, 10], f"expected to keep X(10) and B(5); got {mags}"
