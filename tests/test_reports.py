from pathlib import Path

from ml_severe_weather_forecast.data.reports import parse_spc_csv

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
