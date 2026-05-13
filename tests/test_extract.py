import pandas as pd
import pytest

from ml_severe_weather_forecast.features.extract import temporal_aggregate


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
