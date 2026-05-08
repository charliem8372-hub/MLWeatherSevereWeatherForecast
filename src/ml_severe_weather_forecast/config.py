"""Configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

HAZARDS = ("tornado", "hail", "wind")


class Settings(BaseSettings):
    """Project-wide configuration."""

    model_config = SettingsConfigDict(env_prefix="MLSWF_", env_file=".env", extra="ignore")

    project_root: Path = Field(default_factory=Path.cwd)
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data")

    # Grid (50 km Lambert Conformal)
    grid_dx_km: float = 50.0
    grid_lon_min: float = -125.0
    grid_lon_max: float = -65.0
    grid_lat_min: float = 24.0
    grid_lat_max: float = 50.0

    # Lambert Conformal Conic (NCEP CONUS standard)
    lcc_lat_0: float = 38.5
    lcc_lon_0: float = -97.5
    lcc_lat_1: float = 38.5
    lcc_lat_2: float = 38.5

    # Coverage window
    train_year_start: int = 2022
    train_year_end: int = 2024
    season_month_start: int = 4  # April
    season_month_end: int = 7  # July

    # Climatology window
    climo_year_start: int = 2010
    climo_year_end: int = 2021

    # HRRR cycle
    hrrr_cycle_hour: int = 12  # 12z
    hrrr_forecast_hours: tuple[int, ...] = tuple(range(25))  # f00-f24

    # Label radius (Nadocast spec, ~25 miles)
    label_radius_km: float = 40.0

    @property
    def hrrr_dir(self) -> Path:
        return self.data_dir / "hrrr"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def outlooks_dir(self) -> Path:
        return self.data_dir / "spc_outlooks"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"


settings = Settings()
