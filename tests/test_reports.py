from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml_severe_weather_forecast.data.reports import (
    apply_severity_filters,
    build_reports,
    dedup_reports,
    parse_spc_csv,
)

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


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_severity_filter_drops_subsevere_hail() -> None:
    df = _make_df(
        [
            {
                "event_time_utc": datetime(2023, 5, 15, tzinfo=UTC),
                "lat": 35.0,
                "lon": -97.0,
                "hazard": "hail",
                "magnitude": 0.75,
            },
            {
                "event_time_utc": datetime(2023, 5, 15, tzinfo=UTC),
                "lat": 35.0,
                "lon": -97.0,
                "hazard": "hail",
                "magnitude": 1.0,
            },
            {
                "event_time_utc": datetime(2023, 5, 15, tzinfo=UTC),
                "lat": 35.0,
                "lon": -97.0,
                "hazard": "hail",
                "magnitude": 2.5,
            },
        ]
    )
    out = apply_severity_filters(df)
    assert len(out) == 2
    assert (out["magnitude"] >= 1.0).all()


def test_severity_filter_keeps_all_tornadoes() -> None:
    df = _make_df(
        [
            {
                "event_time_utc": datetime(2023, 5, 15, tzinfo=UTC),
                "lat": 35.0,
                "lon": -97.0,
                "hazard": "tor",
                "magnitude": 0,
            },
            {
                "event_time_utc": datetime(2023, 5, 15, tzinfo=UTC),
                "lat": 35.0,
                "lon": -97.0,
                "hazard": "tor",
                "magnitude": -1,
            },
        ]
    )
    out = apply_severity_filters(df)
    assert len(out) == 2


def test_dedup_collapses_close_in_space_and_time() -> None:
    df = _make_df(
        [
            {
                "event_time_utc": datetime(2023, 5, 15, 18, 0, tzinfo=UTC),
                "lat": 35.000,
                "lon": -97.000,
                "hazard": "tor",
                "magnitude": 0,
            },
            {
                "event_time_utc": datetime(2023, 5, 15, 18, 2, tzinfo=UTC),
                "lat": 35.005,
                "lon": -97.005,
                "hazard": "tor",
                "magnitude": 2,
            },
            {
                "event_time_utc": datetime(2023, 5, 15, 23, 0, tzinfo=UTC),
                "lat": 35.000,
                "lon": -97.000,
                "hazard": "tor",
                "magnitude": 0,
            },
        ]
    )
    out = dedup_reports(df)
    assert len(out) == 2
    # The kept "early" report should be the higher-magnitude one.
    early = out[out["event_time_utc"] < datetime(2023, 5, 15, 20, tzinfo=UTC)]
    assert len(early) == 1
    assert early.iloc[0]["magnitude"] == 2


def test_build_reports_writes_combined_parquet(tmp_path: Path) -> None:
    """Given a directory with per-year/per-hazard CSVs, build_reports collapses to one Parquet/year."""
    src = tmp_path / "src"
    (src / "2023").mkdir(parents=True)
    (src / "2023" / "2023_torn.csv").write_text(FIXTURE.read_text())
    out = tmp_path / "out"
    paths = build_reports(years=[2023], src_dir=src, out_dir=out, hazards=("tor",))
    assert len(paths) == 1
    df = pd.read_parquet(paths[0])
    assert set(df.columns) >= {"event_time_utc", "lat", "lon", "hazard", "magnitude"}
