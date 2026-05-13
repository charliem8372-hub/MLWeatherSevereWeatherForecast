"""Generate tiny_hrrr.grib2 from one real HRRR cycle.

Run once locally - output is committed to the repo as a binary fixture.

Usage:
    uv run python tests/fixtures/build_tiny_hrrr.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from herbie.core import Herbie

from ml_severe_weather_forecast.data.hrrr import hrrr_variable_search_string

OUT = Path(__file__).parent / "tiny_hrrr.grib2"

if __name__ == "__main__":
    # Naive datetime: herbie's internal date validation compares against a tz-naive
    # pd.Timestamp.utcnow().tz_localize(None), so a tz-aware datetime raises TypeError.
    cycle = datetime(2023, 5, 15, 12)
    h = Herbie(
        date=cycle,
        model="hrrr",
        product="sfc",
        fxx=6,
    )
    src = h.download(search=hrrr_variable_search_string())
    print(f"Downloaded {src} ({Path(src).stat().st_size / 1e6:.1f} MB)")
    OUT.write_bytes(Path(src).read_bytes())
    print(f"Wrote {OUT}")
