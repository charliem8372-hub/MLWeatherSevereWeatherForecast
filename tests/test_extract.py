import pandas as pd
import pytest

from ml_severe_weather_forecast.features.extract import snapshot_anchors, temporal_aggregate


def test_temporal_aggregate_max_mean_p90() -> None:
    cells = ["c_001_001", "c_002_002"]
    frames = [
        pd.DataFrame({"cell_id": cells, "MLCAPE_max": [100.0, 50.0], "MLCAPE_mean": [80.0, 40.0]}),
        pd.DataFrame({"cell_id": cells, "MLCAPE_max": [200.0, 60.0], "MLCAPE_mean": [150.0, 50.0]}),
        pd.DataFrame({"cell_id": cells, "MLCAPE_max": [300.0, 70.0], "MLCAPE_mean": [250.0, 60.0]}),
    ]
    out = temporal_aggregate(frames, instantaneous_vars=["MLCAPE"], hourly_max_vars=[])
    row0 = out.loc[out["cell_id"] == "c_001_001"].iloc[0]
    assert row0["MLCAPE_max_fhr_max"] == 300.0
    assert row0["MLCAPE_max_fhr_mean"] == 200.0
    assert row0["MLCAPE_max_fhr_p90"] >= 250


def test_temporal_aggregate_hourly_max_keeps_only_max() -> None:
    cells = ["c_001_001"]
    frames = [
        pd.DataFrame({"cell_id": cells, "MXUPHL_2_5km_max": [10.0]}),
        pd.DataFrame({"cell_id": cells, "MXUPHL_2_5km_max": [50.0]}),
        pd.DataFrame({"cell_id": cells, "MXUPHL_2_5km_max": [25.0]}),
    ]
    out = temporal_aggregate(frames, instantaneous_vars=[], hourly_max_vars=["MXUPHL_2_5km"])
    row = out.iloc[0]
    assert row["MXUPHL_2_5km_max_fhr_max"] == 50.0
    assert "MXUPHL_2_5km_max_fhr_mean" not in out.columns


def test_temporal_aggregate_empty_frames_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        temporal_aggregate([], instantaneous_vars=[], hourly_max_vars=[])


def test_snapshot_anchors_pull_specific_forecast_hours() -> None:
    cells = ["c_001_001"]
    frames = []
    for fhr in range(0, 13):  # f00..f12
        frames.append(pd.DataFrame({"cell_id": cells, "MLCAPE_max": [float(fhr) * 100]}))
    out = snapshot_anchors(frames, fhrs=[6, 12], instantaneous_vars=["MLCAPE"])
    row = out.iloc[0]
    assert row["MLCAPE_max_f06"] == 600.0
    assert row["MLCAPE_max_f12"] == 1200.0


def test_snapshot_anchors_handles_mean_column_and_out_of_range_fhr() -> None:
    cells = ["c_001_001"]
    frames = [
        pd.DataFrame({"cell_id": cells, "MLCAPE_max": [100.0], "MLCAPE_mean": [50.0]}),
        pd.DataFrame({"cell_id": cells, "MLCAPE_max": [200.0], "MLCAPE_mean": [150.0]}),
    ]
    out = snapshot_anchors(frames, fhrs=[0, 1, 99], instantaneous_vars=["MLCAPE"])
    assert out.loc[0, "MLCAPE_max_f00"] == 100.0
    assert out.loc[0, "MLCAPE_mean_f00"] == 50.0
    assert out.loc[0, "MLCAPE_max_f01"] == 200.0
    assert "MLCAPE_max_f99" not in out.columns  # out-of-range fhr is skipped
