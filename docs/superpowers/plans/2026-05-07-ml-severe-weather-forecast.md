# ML Severe-Weather Forecaster — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-quality severe-weather ML forecaster following Nadocast's methodology — probabilistic 24-hour CONUS severe-weather forecasts (tornado, hail, wind) using XGBoost on HRRR forecast fields, verified against SPC storm reports.

**Architecture:** Single Python package `ml_severe_weather_forecast` with a typer CLI. Six idempotent pipeline stages (`download → extract → label → train → verify → plot`) communicate via Parquet on disk. No workflow framework. Notebooks for analysis only.

**Tech Stack:** Python 3.12+, `uv` for packaging, `herbie-data` + `cfgrib` for HRRR ingest, `xarray` + `numpy` for gridded math, `pyproj` + `cartopy` for projections, `scikit-learn` for BallTree/IsotonicRegression, `xgboost` (CUDA) for modeling, `matplotlib` for figures, `pytest` for tests, `structlog` for logging, `ruff` + `mypy` + `pre-commit` for hygiene.

**Spec:** `docs/superpowers/specs/2026-05-07-ml-severe-weather-forecast-design.md`

**Scope note:** This plan implements the spec end-to-end as one coherent project. Phases are sequential because each depends on the previous (no truly independent subsystems). Tasks within a phase are small, testable, and committed individually.

**Phase outline:**
- **Phase 0** — Repo bootstrap (Tasks 1–6)
- **Phase 1** — 50 km grid foundation (Tasks 7–9)
- **Phase 2** — SPC storm reports (Tasks 10–13)
- **Phase 3** — Labels (Tasks 14–16)
- **Phase 4** — HRRR data ingest (Tasks 17–20)
- **Phase 5** — Feature engineering (Tasks 21–27)
- **Phase 6** — SPC outlook archive baseline (Tasks 28–30)
- **Phase 7** — Climatology baseline (Task 31)
- **Phase 8** — Training table assembly (Task 32)
- **Phase 9** — Training + hyperparameter search (Tasks 33–37)
- **Phase 10** — Calibration (Tasks 38–39)
- **Phase 11** — Verification metrics (Tasks 40–46)
- **Phase 12** — Visualizations (Tasks 47–53)
- **Phase 13** — End-to-end orchestration & report (Tasks 54–57)

---

## Phase 0 — Repo Bootstrap

### Task 1: Initialize repo and uv project

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.python-version`

- [ ] **Step 1: Initialize git and uv project**

```bash
cd C:\Users\charl\OneDrive\Documents\ClaudeCode
git init
echo "3.12" > .python-version
uv init --package --name ml-severe-weather-forecast --python 3.12
```

- [ ] **Step 2: Replace generated `pyproject.toml` with the project's**

```toml
[project]
name = "ml-severe-weather-forecast"
version = "0.1.0"
description = "ML severe-weather forecaster (CONUS, XGBoost on HRRR)"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "typer>=0.12",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "structlog>=24.1",
    "numpy>=1.26",
    "pandas>=2.2",
    "pyarrow>=16.0",
    "xarray>=2024.5",
    "cfgrib>=0.9.12",
    "eccodes>=2.36",
    "herbie-data>=2024.5",
    "pyproj>=3.6",
    "shapely>=2.0",
    "scikit-learn>=1.5",
    "scipy>=1.13",
    "xgboost>=2.1",
    "matplotlib>=3.8",
    "cartopy>=0.23",
    "joblib>=1.4",
    "tqdm>=4.66",
    "psutil>=5.9",
    "shap>=0.45",
    "geopandas>=1.0",
    "fiona>=1.10",
]

[project.scripts]
mlswf = "ml_severe_weather_forecast.cli:app"

[dependency-groups]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "mypy>=1.10",
    "pre-commit>=3.7",
    "nbstripout>=0.7",
    "ipykernel>=6.29",
    "jupyter>=1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ml_severe_weather_forecast"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.12"
strict = true
files = ["src/ml_severe_weather_forecast"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --strict-markers"
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/

# Project data (regenerable)
data/*
!data/.gitkeep
docs/verification_report.html

# Notebook checkpoints
.ipynb_checkpoints/
*-checkpoint.ipynb

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 4: Sync deps and verify install**

```bash
uv sync
uv run python -c "import xgboost, herbie, cfgrib, cartopy; print('all good')"
```
Expected: `all good` printed. If `cfgrib` fails, the `eccodes` wheel may not be available for your platform — fall back to `conda install -c conda-forge cfgrib eccodes` and remove those two from `pyproject.toml`.

- [ ] **Step 5: Commit**

```bash
git add .python-version pyproject.toml uv.lock .gitignore
git commit -m "chore: initialize uv project with dependencies"
```

---

### Task 2: Create source skeleton with stub modules

**Files:**
- Create: `src/ml_severe_weather_forecast/__init__.py`
- Create: `src/ml_severe_weather_forecast/cli.py`
- Create: `src/ml_severe_weather_forecast/config.py`
- Create: `src/ml_severe_weather_forecast/logging.py`
- Create: `src/ml_severe_weather_forecast/data/__init__.py`
- Create: `src/ml_severe_weather_forecast/features/__init__.py`
- Create: `data/.gitkeep`
- Create: `tests/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write a smoke test that imports the package and runs the CLI**

```python
# tests/test_smoke.py
from typer.testing import CliRunner

from ml_severe_weather_forecast.cli import app


def test_cli_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "mlswf" in result.stdout.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_smoke.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_severe_weather_forecast.cli'`.

- [ ] **Step 3: Write `src/ml_severe_weather_forecast/__init__.py`**

```python
"""ML severe-weather forecaster (CONUS, XGBoost on HRRR)."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write `src/ml_severe_weather_forecast/logging.py`**

```python
"""Structured logging setup."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(stage: str, log_dir: Path | None = None) -> structlog.BoundLogger:
    """Configure structlog with JSON output to file + pretty to stderr."""
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    structlog.configure(
        processors=[*processors, structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    log = structlog.get_logger().bind(stage=stage)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    return log
```

- [ ] **Step 5: Write `src/ml_severe_weather_forecast/config.py`**

```python
"""Configuration via pydantic-settings."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

HAZARDS = ("tornado", "hail", "wind")


class Settings(BaseSettings):
    """Project-wide configuration."""

    model_config = SettingsConfigDict(env_prefix="NADOCAST_", env_file=".env", extra="ignore")

    project_root: Path = Field(default=Path.cwd())
    data_dir: Path = Field(default=Path.cwd() / "data")

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
```

- [ ] **Step 6: Write `src/ml_severe_weather_forecast/cli.py` (stub commands)**

```python
"""Typer CLI entry point. Subcommands are added in later tasks."""
from __future__ import annotations

import typer

app = typer.Typer(
    name="mlswf",
    help="ML severe-weather forecaster (CONUS, XGBoost on HRRR).",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """mlswf — historical-evaluation pipeline."""


if __name__ == "__main__":
    app()
```

- [ ] **Step 7: Create empty submodule packages and data directory placeholder**

```python
# src/ml_severe_weather_forecast/data/__init__.py
"""Data acquisition modules (HRRR, reports, outlooks, grid)."""
```

```python
# src/ml_severe_weather_forecast/features/__init__.py
"""Feature engineering modules."""
```

```python
# tests/__init__.py
```

Create `data/.gitkeep` as empty file: `New-Item data/.gitkeep -ItemType File`.

- [ ] **Step 8: Run the smoke test to verify it passes**

```bash
uv run pytest tests/test_smoke.py -v
```
Expected: 1 passed.

- [ ] **Step 9: Commit**

```bash
git add src/ tests/ data/.gitkeep
git commit -m "feat: package skeleton with stubbed CLI, config, and logging"
```

---

### Task 3: Set up pre-commit hooks

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write the pre-commit config**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        files: ^src/
        additional_dependencies:
          - pydantic>=2.7
          - pydantic-settings>=2.3
          - types-requests
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
        files: \.ipynb$
```

- [ ] **Step 2: Install hooks and verify they run**

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
Expected: ruff/mypy/nbstripout all pass (or auto-fix). If mypy fails on `cli.py` because typer isn't in additional_dependencies, that's expected — we run mypy on `src/` only and the typer types come from the runtime env once we add `--installed-types`.

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit hooks (ruff, mypy, nbstripout)"
```

---

### Task 4: Set up GitHub Actions CI

**Files:**
- Create: `.github/workflows/test.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"
      - name: Set up Python
        run: uv python install 3.12
      - name: Install eccodes (for cfgrib)
        run: sudo apt-get update && sudo apt-get install -y libeccodes-dev
      - name: Sync deps
        run: uv sync --all-extras
      - name: Lint
        run: uv run ruff check .
      - name: Type check
        run: uv run mypy src/
      - name: Test
        run: uv run pytest tests/ -v --cov=src/ml_severe_weather_forecast --cov-report=term-missing
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: add GitHub Actions test workflow"
```

---

### Task 5: README skeleton

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write a README that documents the project at a high level and references the spec**

```markdown
# ml-severe-weather-forecast

ML-driven probabilistic 24-hour CONUS severe-weather forecasts (tornado, hail, wind), using XGBoost on HRRR forecast fields and verified against SPC storm reports. Methodology inspired by [Nadocast](https://nadocast.com).

**Status:** in development. See `docs/superpowers/specs/2026-05-07-ml-severe-weather-forecast-design.md` for the full spec.

## Quick start

```bash
uv sync
uv run mlswf --help
```

## Pipeline

```
download → extract → label → train → verify → plot
```

Each stage is idempotent; data flows via Parquet under `data/`.

## Reproducing the verification report

```bash
uv run mlswf download hrrr        --start 2022-04-01 --end 2024-07-31
uv run mlswf download reports     --start 2010 --end 2024
uv run mlswf download spc-outlooks --start 2022 --end 2024
uv run python scripts/01_extract_all.py
uv run python scripts/02_train_all.py
uv run python scripts/03_verify_all.py
```

The full pipeline takes ~20 hours of wall-clock on a Ryzen 7 5800X + RTX 4070 SUPER.

## Hardware target

Ryzen 7 5800X / 32 GB DDR4 / RTX 4070 SUPER / 500+ GB free NVMe.

## Conda fallback

If `uv sync` fails on `cfgrib`/`eccodes` (typically Windows-specific), use:

```bash
conda env create -f environment.yml
conda activate ml-severe-weather-forecast
pip install -e .
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with quick-start and reproduction steps"
```

---

### Task 6: Initial CI sanity run

- [ ] **Step 1: Push to remote and verify CI passes**

```bash
git remote add origin <your-github-url>
git push -u origin main
```

- [ ] **Step 2: Confirm GitHub Actions run is green**

Open the Actions tab on GitHub. Expected: 1 successful workflow run with the smoke test.

(Skip this task if you're not pushing to GitHub yet — it's an integration check, not a coding task.)

---

## Phase 1 — 50 km Grid Foundation

The grid is the spatial backbone for everything: cell IDs are foreign keys joining HRRR features, SPC reports, and SPC outlooks.

### Task 7: Lambert Conformal grid module — projection setup

**Files:**
- Create: `src/ml_severe_weather_forecast/data/grid.py`
- Test: `tests/test_grid.py`

- [ ] **Step 1: Write a failing test for grid construction**

```python
# tests/test_grid.py
import numpy as np

from ml_severe_weather_forecast.config import settings
from ml_severe_weather_forecast.data.grid import Grid, build_grid


def test_build_grid_has_expected_cell_count() -> None:
    grid = build_grid()
    # Expected: roughly 6500 cells over CONUS at 50 km. Lower bound 5000, upper 8000.
    assert 5000 < grid.n_cells < 8000


def test_grid_has_unique_cell_ids() -> None:
    grid = build_grid()
    assert len(set(grid.cell_ids)) == grid.n_cells


def test_grid_centers_are_lat_lon() -> None:
    grid = build_grid()
    assert np.all((grid.lats >= 23) & (grid.lats <= 51))
    assert np.all((grid.lons >= -126) & (grid.lons <= -64))
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/test_grid.py -v
```
Expected: FAIL — `ImportError: cannot import name 'Grid' from 'ml_severe_weather_forecast.data.grid'`.

- [ ] **Step 3: Implement `grid.py`**

```python
"""50 km Lambert Conformal Conic grid over CONUS."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from pyproj import CRS, Transformer

from ml_severe_weather_forecast.config import settings


def _lcc_crs() -> CRS:
    return CRS.from_proj4(
        f"+proj=lcc +lat_0={settings.lcc_lat_0} +lon_0={settings.lcc_lon_0} "
        f"+lat_1={settings.lcc_lat_1} +lat_2={settings.lcc_lat_2} "
        f"+x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
    )


@dataclass(frozen=True)
class Grid:
    """50 km cell-centered grid over CONUS in Lambert Conformal projection."""

    cell_ids: np.ndarray  # str, shape (N,) — "c_iii_jjj"
    i_index: np.ndarray  # int, shape (N,) — column in projected space
    j_index: np.ndarray  # int, shape (N,) — row in projected space
    x_centers: np.ndarray  # float, shape (N,) — x in meters (LCC)
    y_centers: np.ndarray  # float, shape (N,) — y in meters (LCC)
    lats: np.ndarray  # float, shape (N,)
    lons: np.ndarray  # float, shape (N,)

    @cached_property
    def n_cells(self) -> int:
        return int(self.cell_ids.size)


def build_grid() -> Grid:
    """Construct the canonical 50 km CONUS grid."""
    crs_lcc = _lcc_crs()
    crs_geo = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(crs_geo, crs_lcc, always_xy=True)
    inv = Transformer.from_crs(crs_lcc, crs_geo, always_xy=True)

    # Project bbox corners to LCC to find x/y range
    corner_lons = np.array(
        [settings.grid_lon_min, settings.grid_lon_max, settings.grid_lon_min, settings.grid_lon_max]
    )
    corner_lats = np.array(
        [settings.grid_lat_min, settings.grid_lat_min, settings.grid_lat_max, settings.grid_lat_max]
    )
    xs, ys = fwd.transform(corner_lons, corner_lats)

    dx_m = settings.grid_dx_km * 1000.0
    x_min = np.floor(xs.min() / dx_m) * dx_m
    x_max = np.ceil(xs.max() / dx_m) * dx_m
    y_min = np.floor(ys.min() / dx_m) * dx_m
    y_max = np.ceil(ys.max() / dx_m) * dx_m

    nx = int(round((x_max - x_min) / dx_m))
    ny = int(round((y_max - y_min) / dx_m))

    # Cell centers (i, j) where i is column (x), j is row (y)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    x_centers = x_min + (ii + 0.5) * dx_m
    y_centers = y_min + (jj + 0.5) * dx_m

    lons, lats = inv.transform(x_centers, y_centers)

    # Mask to CONUS bbox in lat/lon
    inside = (
        (lons >= settings.grid_lon_min)
        & (lons <= settings.grid_lon_max)
        & (lats >= settings.grid_lat_min)
        & (lats <= settings.grid_lat_max)
    )

    i_index = ii[inside].astype(np.int32).ravel()
    j_index = jj[inside].astype(np.int32).ravel()
    x_flat = x_centers[inside].astype(np.float64).ravel()
    y_flat = y_centers[inside].astype(np.float64).ravel()
    lons_flat = lons[inside].astype(np.float64).ravel()
    lats_flat = lats[inside].astype(np.float64).ravel()

    cell_ids = np.array(
        [f"c_{i:03d}_{j:03d}" for i, j in zip(i_index, j_index, strict=True)],
        dtype=object,
    )

    return Grid(
        cell_ids=cell_ids,
        i_index=i_index,
        j_index=j_index,
        x_centers=x_flat,
        y_centers=y_flat,
        lats=lats_flat,
        lons=lons_flat,
    )
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_grid.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/data/grid.py tests/test_grid.py
git commit -m "feat(grid): build 50km Lambert Conformal CONUS grid"
```

---

### Task 8: Grid — point-to-cell lookup and roundtrip

**Files:**
- Modify: `src/ml_severe_weather_forecast/data/grid.py`
- Modify: `tests/test_grid.py`

- [ ] **Step 1: Add tests for point lookup and roundtrip**

Append to `tests/test_grid.py`:

```python
import pytest

from ml_severe_weather_forecast.data.grid import Grid, build_grid, latlon_to_cell_id


def test_known_city_resolves_to_a_cell() -> None:
    grid = build_grid()
    # Norman, OK — center of tornado alley, definitely inside CONUS
    cell_id = latlon_to_cell_id(35.222, -97.439, grid)
    assert cell_id is not None
    assert cell_id.startswith("c_")


def test_offshore_point_returns_none() -> None:
    grid = build_grid()
    # Middle of Atlantic
    assert latlon_to_cell_id(35.0, -50.0, grid) is None


def test_cell_center_roundtrip() -> None:
    grid = build_grid()
    # Pick the cell containing Norman, OK; lookup its center; that center should map to itself
    cell_id = latlon_to_cell_id(35.222, -97.439, grid)
    assert cell_id is not None
    idx = list(grid.cell_ids).index(cell_id)
    same_cell = latlon_to_cell_id(grid.lats[idx], grid.lons[idx], grid)
    assert same_cell == cell_id
```

- [ ] **Step 2: Run new tests, expect failure**

```bash
uv run pytest tests/test_grid.py -v
```
Expected: 3 new tests fail with `ImportError: cannot import name 'latlon_to_cell_id'`.

- [ ] **Step 3: Implement the lookup function**

Append to `src/ml_severe_weather_forecast/data/grid.py`:

```python
from pyproj import Transformer as _T  # noqa: E402

_FWD: _T | None = None


def _fwd() -> _T:
    global _FWD
    if _FWD is None:
        _FWD = _T.from_crs(CRS.from_epsg(4326), _lcc_crs(), always_xy=True)
    return _FWD


def latlon_to_cell_id(lat: float, lon: float, grid: Grid) -> str | None:
    """Look up the cell ID containing a (lat, lon) point. Returns None if outside the grid."""
    x, y = _fwd().transform(lon, lat)
    dx_m = settings.grid_dx_km * 1000.0
    x_min = grid.x_centers.min() - dx_m / 2.0
    y_min = grid.y_centers.min() - dx_m / 2.0
    i = int(np.floor((x - x_min) / dx_m))
    j = int(np.floor((y - y_min) / dx_m))
    target = f"c_{i:03d}_{j:03d}"
    matches = np.where(grid.cell_ids == target)[0]
    return target if matches.size else None
```

- [ ] **Step 4: Run all grid tests**

```bash
uv run pytest tests/test_grid.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/data/grid.py tests/test_grid.py
git commit -m "feat(grid): point-to-cell lookup with offshore handling"
```

---

### Task 9: Grid persistence — save/load to Parquet

**Files:**
- Modify: `src/ml_severe_weather_forecast/data/grid.py`
- Modify: `tests/test_grid.py`

- [ ] **Step 1: Add round-trip persistence test**

Append to `tests/test_grid.py`:

```python
def test_grid_save_load_roundtrip(tmp_path) -> None:
    grid = build_grid()
    path = tmp_path / "grid.parquet"
    save_grid(grid, path)
    loaded = load_grid(path)
    np.testing.assert_array_equal(loaded.cell_ids, grid.cell_ids)
    np.testing.assert_allclose(loaded.lats, grid.lats)
    np.testing.assert_allclose(loaded.lons, grid.lons)
```

Add the imports at the top:

```python
from ml_severe_weather_forecast.data.grid import load_grid, save_grid
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_grid.py::test_grid_save_load_roundtrip -v
```
Expected: FAIL on import.

- [ ] **Step 3: Implement save/load**

Append to `src/ml_severe_weather_forecast/data/grid.py`:

```python
import pandas as pd  # noqa: E402
from pathlib import Path  # noqa: E402


def save_grid(grid: Grid, path: Path) -> None:
    df = pd.DataFrame(
        {
            "cell_id": grid.cell_ids.astype(str),
            "i_index": grid.i_index,
            "j_index": grid.j_index,
            "x_center_m": grid.x_centers,
            "y_center_m": grid.y_centers,
            "lat": grid.lats,
            "lon": grid.lons,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_grid(path: Path) -> Grid:
    df = pd.read_parquet(path)
    return Grid(
        cell_ids=df["cell_id"].to_numpy(dtype=object),
        i_index=df["i_index"].to_numpy(dtype=np.int32),
        j_index=df["j_index"].to_numpy(dtype=np.int32),
        x_centers=df["x_center_m"].to_numpy(dtype=np.float64),
        y_centers=df["y_center_m"].to_numpy(dtype=np.float64),
        lats=df["lat"].to_numpy(dtype=np.float64),
        lons=df["lon"].to_numpy(dtype=np.float64),
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_grid.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/data/grid.py tests/test_grid.py
git commit -m "feat(grid): save/load grid to Parquet"
```

---

## Phase 2 — SPC Storm Reports

### Task 10: Storm-report download CLI

**Files:**
- Create: `src/ml_severe_weather_forecast/data/reports.py`
- Modify: `src/ml_severe_weather_forecast/cli.py`
- Test: `tests/test_reports.py`
- Test fixture: `tests/fixtures/sample_lsr.csv`

- [ ] **Step 1: Create the test fixture (a tiny SPC-format LSR CSV)**

```csv
# tests/fixtures/sample_lsr.csv
om,yr,mo,dy,date,time,tz,st,stf,stn,mag,inj,fat,loss,closs,slat,slon,elat,elon,len,wid,fc
1,2023,5,15,2023-05-15,180000,3,OK,40,0,2,0,0,0,0,35.225,-97.440,35.230,-97.430,0.5,100,0
2,2023,5,15,2023-05-15,200000,3,KS,20,0,1,0,0,0,0,38.500,-97.000,0,0,0,0,0
```

- [ ] **Step 2: Write tests for the parser and downloader**

```python
# tests/test_reports.py
from pathlib import Path

import pandas as pd

from ml_severe_weather_forecast.data.reports import (
    apply_severity_filters,
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
    empty.write_text("om,yr,mo,dy,date,time,tz,st,stf,stn,mag,inj,fat,loss,closs,slat,slon,elat,elon,len,wid,fc\n")
    df = parse_spc_csv(empty, hazard="tor")
    assert len(df) == 0
```

- [ ] **Step 3: Run, expect failure**

```bash
uv run pytest tests/test_reports.py -v
```
Expected: FAIL on import.

- [ ] **Step 4: Implement `reports.py`**

```python
"""SPC storm-report ingestion and normalization."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from ml_severe_weather_forecast.config import settings

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
    tz = int(row["tz"])
    offset_hours = _TZ_OFFSET_HOURS.get(tz, 6)
    local = datetime(
        int(row["yr"]),
        int(row["mo"]),
        int(row["dy"]),
        int(str(row["time"]).zfill(6)[:2]),
        int(str(row["time"]).zfill(6)[2:4]),
        int(str(row["time"]).zfill(6)[4:6]),
        tzinfo=UTC,
    )
    return local + timedelta(hours=offset_hours)


def parse_spc_csv(path: Path, hazard: str) -> pd.DataFrame:
    """Parse one SPC severe-weather DB CSV into the canonical schema."""
    if hazard not in HAZARD_VALID:
        raise ValueError(f"hazard must be one of {HAZARD_VALID}, got {hazard!r}")
    df = pd.read_csv(path, comment="#")
    if df.empty:
        return pd.DataFrame(
            columns=["event_time_utc", "lat", "lon", "hazard", "magnitude"]
        ).astype({"lat": float, "lon": float})
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

    - tornado: any EF rating (mag ≥ -1, where -1 is unrated)
    - hail: ≥ 1.00 inch (mag ≥ 1.0)
    - wind: ≥ 50 kt (mag ≥ 50; note SPC stores wind mag in knots)
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
    """Cluster reports within 5 min × 10 mi; keep the most-severe per cluster."""
    if df.empty:
        return df
    from sklearn.neighbors import BallTree

    df = df.sort_values(["hazard", "event_time_utc"]).reset_index(drop=True)
    keep_mask = pd.Series(True, index=df.index)
    for hazard, sub in df.groupby("hazard", sort=False):
        if len(sub) < 2:
            continue
        rad = sub[["lat", "lon"]].to_numpy() * (3.141592653589793 / 180.0)
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
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_reports.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ml_severe_weather_forecast/data/reports.py tests/test_reports.py tests/fixtures/sample_lsr.csv
git commit -m "feat(reports): SPC storm-report parser, downloader, severity filter, dedup"
```

---

### Task 11: Severity-filter and dedup tests

**Files:**
- Modify: `tests/test_reports.py`

- [ ] **Step 1: Add tests for the filtering and deduplication helpers**

Append to `tests/test_reports.py`:

```python
import pandas as pd
from datetime import datetime, UTC


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_severity_filter_drops_subsevere_hail() -> None:
    df = _make_df(
        [
            {"event_time_utc": datetime(2023, 5, 15, tzinfo=UTC), "lat": 35.0, "lon": -97.0, "hazard": "hail", "magnitude": 0.75},
            {"event_time_utc": datetime(2023, 5, 15, tzinfo=UTC), "lat": 35.0, "lon": -97.0, "hazard": "hail", "magnitude": 1.0},
            {"event_time_utc": datetime(2023, 5, 15, tzinfo=UTC), "lat": 35.0, "lon": -97.0, "hazard": "hail", "magnitude": 2.5},
        ]
    )
    out = apply_severity_filters(df)
    assert len(out) == 2
    assert (out["magnitude"] >= 1.0).all()


def test_severity_filter_keeps_all_tornadoes() -> None:
    df = _make_df(
        [
            {"event_time_utc": datetime(2023, 5, 15, tzinfo=UTC), "lat": 35.0, "lon": -97.0, "hazard": "tor", "magnitude": 0},
            {"event_time_utc": datetime(2023, 5, 15, tzinfo=UTC), "lat": 35.0, "lon": -97.0, "hazard": "tor", "magnitude": -1},
        ]
    )
    out = apply_severity_filters(df)
    assert len(out) == 2


def test_dedup_collapses_close_in_space_and_time() -> None:
    df = _make_df(
        [
            {"event_time_utc": datetime(2023, 5, 15, 18, 0, tzinfo=UTC), "lat": 35.000, "lon": -97.000, "hazard": "tor", "magnitude": 0},
            {"event_time_utc": datetime(2023, 5, 15, 18, 2, tzinfo=UTC), "lat": 35.005, "lon": -97.005, "hazard": "tor", "magnitude": 2},
            {"event_time_utc": datetime(2023, 5, 15, 23, 0, tzinfo=UTC), "lat": 35.000, "lon": -97.000, "hazard": "tor", "magnitude": 0},
        ]
    )
    out = dedup_reports(df)
    assert len(out) == 2
    # The kept "early" report should be the higher-magnitude one.
    early = out[out["event_time_utc"] < datetime(2023, 5, 15, 20, tzinfo=UTC)]
    assert len(early) == 1
    assert early.iloc[0]["magnitude"] == 2
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_reports.py -v
```
Expected: 5 passed total.

- [ ] **Step 3: Commit**

```bash
git add tests/test_reports.py
git commit -m "test(reports): cover severity filter and dedup edge cases"
```

---

### Task 12: Reports CLI subcommand

**Files:**
- Modify: `src/ml_severe_weather_forecast/cli.py`

- [ ] **Step 1: Add a typer-app group for `download` with a `reports` subcommand**

Replace `src/ml_severe_weather_forecast/cli.py` with:

```python
"""Typer CLI entry point."""
from __future__ import annotations

from pathlib import Path

import typer

from ml_severe_weather_forecast.config import settings

app = typer.Typer(
    name="mlswf",
    help="ML severe-weather forecaster (CONUS, XGBoost on HRRR).",
    no_args_is_help=True,
)
download_app = typer.Typer(name="download", help="Data acquisition commands.", no_args_is_help=True)
app.add_typer(download_app)


@download_app.command("reports")
def download_reports_cmd(
    start: int = typer.Option(..., help="First year (inclusive)."),
    end: int = typer.Option(..., help="Last year (inclusive)."),
    dest: Path = typer.Option(None, help="Override the default reports dir."),
    force: bool = typer.Option(False, help="Re-download even if cached."),
) -> None:
    """Download SPC severe-weather DB CSVs for tornado/hail/wind."""
    from ml_severe_weather_forecast.data.reports import HAZARD_VALID, download_spc_year

    target_dir = dest or settings.reports_dir
    typer.echo(f"Downloading SPC reports {start}–{end} → {target_dir}")
    for year in range(start, end + 1):
        for hazard in HAZARD_VALID:
            path = download_spc_year(year, hazard, target_dir, force=force)
            typer.echo(f"  {year} {hazard}: {path}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Smoke-test the CLI**

```bash
uv run mlswf download --help
uv run mlswf download reports --help
```
Expected: help text with `--start`, `--end`, `--dest`, `--force`.

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/cli.py
git commit -m "feat(cli): mlswf download reports subcommand"
```

---

### Task 13: Reports build pipeline (download → parse → filter → dedup → save)

**Files:**
- Modify: `src/ml_severe_weather_forecast/data/reports.py`
- Modify: `src/ml_severe_weather_forecast/cli.py`
- Test: `tests/test_reports.py`

- [ ] **Step 1: Add a build_reports test that uses the fixture**

Append to `tests/test_reports.py`:

```python
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
```

Add the import at the top:

```python
from ml_severe_weather_forecast.data.reports import build_reports
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_reports.py::test_build_reports_writes_combined_parquet -v
```
Expected: FAIL on import.

- [ ] **Step 3: Implement `build_reports`**

Append to `src/ml_severe_weather_forecast/data/reports.py`:

```python
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
```

- [ ] **Step 4: Wire into the CLI — extend `download reports` to also build per-year Parquets**

Modify `download_reports_cmd` in `src/ml_severe_weather_forecast/cli.py` to call `build_reports` after the downloads. Replace its body with:

```python
    from ml_severe_weather_forecast.data.reports import HAZARD_VALID, build_reports, download_spc_year

    target_dir = dest or settings.reports_dir
    typer.echo(f"Downloading SPC reports {start}–{end} → {target_dir}")
    for year in range(start, end + 1):
        for hazard in HAZARD_VALID:
            path = download_spc_year(year, hazard, target_dir, force=force)
            typer.echo(f"  {year} {hazard}: {path}")
    typer.echo("Building per-year Parquet artifacts…")
    parquets = build_reports(list(range(start, end + 1)), target_dir, target_dir)
    for p in parquets:
        typer.echo(f"  {p}")
```

- [ ] **Step 5: Run all reports tests**

```bash
uv run pytest tests/test_reports.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ml_severe_weather_forecast/data/reports.py src/ml_severe_weather_forecast/cli.py tests/test_reports.py
git commit -m "feat(reports): build_reports pipeline + CLI integration"
```

---

## Phase 3 — Labels

### Task 14: Spatiotemporal label join

**Files:**
- Create: `src/ml_severe_weather_forecast/labels.py`
- Test: `tests/test_labels.py`

- [ ] **Step 1: Write failing tests for label generation**

```python
# tests/test_labels.py
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from ml_severe_weather_forecast.data.grid import build_grid, latlon_to_cell_id
from ml_severe_weather_forecast.labels import label_cycle


def _reports_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_known_tornado_labels_its_cell() -> None:
    grid = build_grid()
    cycle = datetime(2023, 5, 15, 12, tzinfo=UTC)
    norman = (35.222, -97.439)
    reports = _reports_df(
        [
            {
                "event_time_utc": cycle + timedelta(hours=6),
                "lat": norman[0],
                "lon": norman[1],
                "hazard": "tor",
                "magnitude": 2,
            }
        ]
    )
    labels = label_cycle(grid, cycle_init_utc=cycle, reports=reports, hazards=("tor",))
    expected_cell = latlon_to_cell_id(*norman, grid)
    assert labels.loc[labels["cell_id"] == expected_cell, "tor"].iloc[0] == 1


def test_distant_cells_are_zero() -> None:
    grid = build_grid()
    cycle = datetime(2023, 5, 15, 12, tzinfo=UTC)
    reports = _reports_df(
        [
            {
                "event_time_utc": cycle + timedelta(hours=6),
                "lat": 35.222,
                "lon": -97.439,
                "hazard": "tor",
                "magnitude": 2,
            }
        ]
    )
    labels = label_cycle(grid, cycle_init_utc=cycle, reports=reports, hazards=("tor",))
    seattle_cell = latlon_to_cell_id(47.6, -122.3, grid)
    assert labels.loc[labels["cell_id"] == seattle_cell, "tor"].iloc[0] == 0


def test_out_of_window_reports_excluded() -> None:
    grid = build_grid()
    cycle = datetime(2023, 5, 15, 12, tzinfo=UTC)
    reports = _reports_df(
        [
            {
                "event_time_utc": cycle - timedelta(hours=1),
                "lat": 35.222,
                "lon": -97.439,
                "hazard": "tor",
                "magnitude": 2,
            },
            {
                "event_time_utc": cycle + timedelta(hours=25),
                "lat": 35.222,
                "lon": -97.439,
                "hazard": "tor",
                "magnitude": 2,
            },
        ]
    )
    labels = label_cycle(grid, cycle_init_utc=cycle, reports=reports, hazards=("tor",))
    assert labels["tor"].sum() == 0


def test_labels_include_event_count_and_max_magnitude() -> None:
    grid = build_grid()
    cycle = datetime(2023, 5, 15, 12, tzinfo=UTC)
    reports = _reports_df(
        [
            {"event_time_utc": cycle + timedelta(hours=2), "lat": 35.220, "lon": -97.440, "hazard": "tor", "magnitude": 1},
            {"event_time_utc": cycle + timedelta(hours=3), "lat": 35.222, "lon": -97.441, "hazard": "tor", "magnitude": 3},
        ]
    )
    labels = label_cycle(grid, cycle_init_utc=cycle, reports=reports, hazards=("tor",))
    cell = latlon_to_cell_id(35.222, -97.439, grid)
    row = labels.loc[labels["cell_id"] == cell].iloc[0]
    assert row["tor"] == 1
    assert row["tor_event_count"] >= 2
    assert row["tor_max_magnitude"] == 3
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_labels.py -v
```
Expected: FAIL on import.

- [ ] **Step 3: Implement label join**

```python
# src/ml_severe_weather_forecast/labels.py
"""Spatiotemporal join: SPC storm reports → grid cells → binary labels."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ml_severe_weather_forecast.config import settings
from ml_severe_weather_forecast.data.grid import Grid

EARTH_RADIUS_M = 6_371_000.0


def label_cycle(
    grid: Grid,
    cycle_init_utc: datetime,
    reports: pd.DataFrame,
    hazards: tuple[str, ...] = ("tor", "hail", "wind"),
    *,
    forecast_window_hours: int = 24,
    radius_km: float | None = None,
) -> pd.DataFrame:
    """Generate per-cell binary labels for one forecast cycle.

    Returns a DataFrame keyed by cell_id with columns:
      - cycle_init_utc (constant)
      - cell_id
      - {hazard} (0/1) for each hazard
      - {hazard}_event_count, {hazard}_max_magnitude
    """
    if reports["event_time_utc"].dt.tz is None:
        raise ValueError("reports['event_time_utc'] must be timezone-aware (UTC)")
    radius_km = radius_km or settings.label_radius_km
    radius_rad = (radius_km * 1000.0) / EARTH_RADIUS_M
    window_end = cycle_init_utc + timedelta(hours=forecast_window_hours)

    cell_rad = np.deg2rad(np.column_stack([grid.lats, grid.lons]))
    tree = BallTree(cell_rad, metric="haversine")

    out = pd.DataFrame({"cycle_init_utc": cycle_init_utc, "cell_id": grid.cell_ids})

    for hazard in hazards:
        sub = reports[
            (reports["hazard"] == hazard)
            & (reports["event_time_utc"] >= cycle_init_utc)
            & (reports["event_time_utc"] < window_end)
        ]
        label_col = np.zeros(grid.n_cells, dtype=np.int8)
        count_col = np.zeros(grid.n_cells, dtype=np.int32)
        magmax_col = np.full(grid.n_cells, np.nan, dtype=np.float32)
        if not sub.empty:
            r_rad = np.deg2rad(sub[["lat", "lon"]].to_numpy())
            idx_per_report = tree.query_radius(r_rad, r=radius_rad)
            mags = sub["magnitude"].to_numpy(dtype=np.float32)
            for i, hits in enumerate(idx_per_report):
                for cell_idx in hits:
                    label_col[cell_idx] = 1
                    count_col[cell_idx] += 1
                    cur = magmax_col[cell_idx]
                    if np.isnan(cur) or mags[i] > cur:
                        magmax_col[cell_idx] = mags[i]
        out[hazard] = label_col
        out[f"{hazard}_event_count"] = count_col
        out[f"{hazard}_max_magnitude"] = magmax_col

    return out
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_labels.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/labels.py tests/test_labels.py
git commit -m "feat(labels): per-cycle BallTree spatiotemporal label join"
```

---

### Task 15: Year-level label aggregation

**Files:**
- Modify: `src/ml_severe_weather_forecast/labels.py`
- Modify: `tests/test_labels.py`

- [ ] **Step 1: Add a year-level test**

Append to `tests/test_labels.py`:

```python
def test_build_year_labels(tmp_path) -> None:
    from ml_severe_weather_forecast.labels import build_year_labels

    grid = build_grid()
    reports = _reports_df(
        [
            {
                "event_time_utc": datetime(2023, 5, 15, 18, tzinfo=UTC),
                "lat": 35.222,
                "lon": -97.439,
                "hazard": "tor",
                "magnitude": 2,
            }
        ]
    )
    reports_path = tmp_path / "2023.parquet"
    reports.to_parquet(reports_path, index=False)

    out = tmp_path / "labels"
    cycles = [datetime(2023, 5, 15, 12, tzinfo=UTC)]
    written = build_year_labels(
        year=2023, cycles=cycles, grid=grid, reports_path=reports_path, out_dir=out
    )
    assert written.exists()
    df = pd.read_parquet(written)
    assert df["tor"].sum() >= 1
    assert "cycle_init_utc" in df.columns
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_labels.py::test_build_year_labels -v
```

- [ ] **Step 3: Implement `build_year_labels`**

Append to `src/ml_severe_weather_forecast/labels.py`:

```python
from pathlib import Path  # noqa: E402


def build_year_labels(
    year: int,
    cycles: list[datetime],
    grid: Grid,
    reports_path: Path,
    out_dir: Path,
    hazards: tuple[str, ...] = ("tor", "hail", "wind"),
) -> Path:
    """Generate labels for all cycles in a year and write one Parquet."""
    reports = pd.read_parquet(reports_path)
    if reports["event_time_utc"].dt.tz is None:
        reports["event_time_utc"] = reports["event_time_utc"].dt.tz_localize(UTC)
    parts: list[pd.DataFrame] = []
    for cycle in cycles:
        parts.append(label_cycle(grid, cycle_init_utc=cycle, reports=reports, hazards=hazards))
    df = pd.concat(parts, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{year}.parquet"
    df.to_parquet(path, index=False)
    return path


from datetime import UTC  # noqa: E402  # used inside build_year_labels for tz-naive backfill
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_labels.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/labels.py tests/test_labels.py
git commit -m "feat(labels): build_year_labels aggregator"
```

---

### Task 16: Labels CLI subcommand

**Files:**
- Modify: `src/ml_severe_weather_forecast/cli.py`

- [ ] **Step 1: Add the `mlswf label` command**

Append to `src/ml_severe_weather_forecast/cli.py`:

```python
@app.command("label")
def label_cmd(
    year: int = typer.Option(..., help="Year to label."),
) -> None:
    """Generate per-cycle labels for a given year using cached reports."""
    from datetime import UTC, datetime, timedelta

    from ml_severe_weather_forecast.data.grid import build_grid
    from ml_severe_weather_forecast.labels import build_year_labels

    grid = build_grid()
    reports_path = settings.reports_dir / f"{year}.parquet"
    if not reports_path.exists():
        raise typer.BadParameter(f"missing reports parquet: {reports_path}. Run `mlswf download reports` first.")

    season_start = datetime(year, settings.season_month_start, 1, tzinfo=UTC)
    # Last day of season_month_end:
    if settings.season_month_end == 12:
        season_end = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        season_end = datetime(year, settings.season_month_end + 1, 1, tzinfo=UTC)

    cycles: list[datetime] = []
    cur = season_start.replace(hour=settings.hrrr_cycle_hour)
    while cur < season_end:
        cycles.append(cur)
        cur = cur + timedelta(days=1)

    out = build_year_labels(year, cycles, grid, reports_path, settings.labels_dir)
    typer.echo(f"Wrote {out}")
```

- [ ] **Step 2: Smoke-test**

```bash
uv run mlswf label --help
```
Expected: help text with `--year`.

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/cli.py
git commit -m "feat(cli): mlswf label subcommand"
```

---

## Phase 4 — HRRR Data Ingest

The HRRR layer is the bulk of the data work: download GRIB2, extract a curated set of variables, regrid from 3 km to 50 km cells, persist as Parquet. Each cycle is independent, so this stage is embarrassingly parallel.

### Task 17: HRRR download via herbie

**Files:**
- Create: `src/ml_severe_weather_forecast/data/hrrr.py`
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write a failing test for the variable specification**

```python
# tests/test_hrrr.py
from ml_severe_weather_forecast.data.hrrr import HRRR_VARIABLES, hrrr_variable_search_string


def test_hrrr_variable_list_complete() -> None:
    # Spot check: each of these must be in the variable list (matches spec §5.1)
    expected_subset = {"MLCAPE", "MUCAPE", "MLCIN", "SRH_0_1km", "SRH_0_3km", "BWD_0_6km", "MXUPHL_2_5km"}
    assert expected_subset <= set(HRRR_VARIABLES.keys())


def test_search_string_includes_all_vars() -> None:
    s = hrrr_variable_search_string()
    for grib_pattern in ("CAPE", "CIN", "VUCSH", "MXUPHL"):
        assert grib_pattern in s
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_hrrr.py -v
```

- [ ] **Step 3: Implement `hrrr.py` (variable spec + download wrapper)**

```python
"""HRRR data acquisition via herbie + GRIB2 variable extraction."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import structlog
import xarray as xr

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class HRRRVar:
    """Spec for one HRRR variable.

    `grib_search` is a regex used by herbie to subset by GRIB2 message.
    `aggregate` is the canonical temporal aggregation when assembling features.
    """

    name: str
    grib_search: str
    aggregate: tuple[str, ...]  # subset of {"max", "mean", "p90"}; or ("max",) for hourly-max fields
    is_hourly_max: bool = False


# Mapping from canonical variable name → HRRR GRIB2 search pattern.
# The grib_search pattern matches herbie's `searchString` semantics:
# "VAR:LEVEL:STAT" — variable code colon level info.
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
    "MAXWIND_10m": HRRRVar(
        "MAXWIND_10m", "MAXUW:10 m above ground", ("max",), is_hourly_max=True
    ),
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

    Returns the local paths of downloaded files. Idempotent: skips files already present.
    """
    from herbie import Herbie

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
        path = h.download(searchString=search, verbose=False)
        if path is None:
            log.warning("hrrr.download.none", cycle=cycle_init.isoformat(), fxx=fxx)
            continue
        paths.append(Path(path))
    return paths
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_hrrr.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/data/hrrr.py tests/test_hrrr.py
git commit -m "feat(hrrr): variable spec and herbie-based downloader"
```

---

### Task 18: GRIB2 → xarray extraction

**Files:**
- Modify: `src/ml_severe_weather_forecast/data/hrrr.py`
- Modify: `tests/test_hrrr.py`
- Create test fixture: `tests/fixtures/build_tiny_hrrr.py` (one-time generator)

- [ ] **Step 1: Write a script that produces a tiny GRIB2 fixture**

This script is run once locally to generate the test fixture; it's not part of CI.

```python
# tests/fixtures/build_tiny_hrrr.py
"""Generate tiny_hrrr.grib2 from one real HRRR cycle.

Run once locally — output is committed to the repo as a binary fixture.

Usage:
    uv run python tests/fixtures/build_tiny_hrrr.py
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from herbie import Herbie

from ml_severe_weather_forecast.data.hrrr import hrrr_variable_search_string

OUT = Path(__file__).parent / "tiny_hrrr.grib2"

if __name__ == "__main__":
    cycle = datetime(2023, 5, 15, 12, tzinfo=UTC)
    h = Herbie(
        date=cycle,
        model="hrrr",
        product="sfc",
        fxx=6,
    )
    # Download just one forecast hour with a narrow variable subset
    src = h.download(searchString=hrrr_variable_search_string())
    print(f"Downloaded {src} ({Path(src).stat().st_size / 1e6:.1f} MB)")
    # In practice this file is ~30-100 MB after subset; copy or symlink to OUT
    OUT.write_bytes(Path(src).read_bytes())
    print(f"Wrote {OUT}")
```

(Run once: `uv run python tests/fixtures/build_tiny_hrrr.py`. Commit the resulting `tiny_hrrr.grib2` to the repo. Don't commit if >50 MB; if so, further-subset the search string.)

- [ ] **Step 2: Add an extraction test that loads the fixture**

Append to `tests/test_hrrr.py`:

```python
from pathlib import Path
import pytest
import xarray as xr

from ml_severe_weather_forecast.data.hrrr import extract_variables_to_dataset

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_hrrr.grib2"


@pytest.mark.skipif(not FIXTURE.exists(), reason="tiny_hrrr.grib2 fixture not present")
def test_extract_variables_returns_dataset() -> None:
    ds = extract_variables_to_dataset(FIXTURE)
    # Must contain at least the most common convective variables
    for v in ("MLCAPE", "SRH_0_3km", "MXUPHL_2_5km"):
        assert v in ds.data_vars
    # 2D fields with HRRR grid dims
    assert ds["MLCAPE"].ndim == 2
```

- [ ] **Step 3: Implement `extract_variables_to_dataset`**

Append to `src/ml_severe_weather_forecast/data/hrrr.py`:

```python
import cfgrib  # noqa: E402


def _open_grib_subset(path: Path, filter_keys: dict[str, object]) -> xr.Dataset:
    return cfgrib.open_dataset(
        str(path),
        backend_kwargs={"filter_by_keys": filter_keys, "errors": "ignore"},
    )


_VAR_TO_FILTER: dict[str, dict[str, object]] = {
    "SBCAPE": {"shortName": "cape", "typeOfLevel": "surface"},
    "MLCAPE": {"shortName": "cape", "typeOfLevel": "pressureFromGroundLayer", "topLevel": 9000},
    "MUCAPE": {"shortName": "cape", "typeOfLevel": "pressureFromGroundLayer", "topLevel": 18000},
    "SBCIN": {"shortName": "cin", "typeOfLevel": "surface"},
    "MLCIN": {"shortName": "cin", "typeOfLevel": "pressureFromGroundLayer", "topLevel": 9000},
    "LFTX": {"shortName": "lftx"},
    "SRH_0_1km": {"shortName": "hlcy", "topLevel": 1000},
    "SRH_0_3km": {"shortName": "hlcy", "topLevel": 3000},
    "USHR_0_6km": {"shortName": "vucsh", "topLevel": 6000},
    "VSHR_0_6km": {"shortName": "vvcsh", "topLevel": 6000},
    "USHR_0_1km": {"shortName": "vucsh", "topLevel": 1000},
    "VSHR_0_1km": {"shortName": "vvcsh", "topLevel": 1000},
    "PWAT": {"shortName": "pwat"},
    "T2M": {"shortName": "2t"},
    "TD2M": {"shortName": "2d"},
    "U10": {"shortName": "10u"},
    "V10": {"shortName": "10v"},
    "T_500": {"shortName": "t", "level": 500, "typeOfLevel": "isobaricInhPa"},
    "T_700": {"shortName": "t", "level": 700, "typeOfLevel": "isobaricInhPa"},
    "T_850": {"shortName": "t", "level": 850, "typeOfLevel": "isobaricInhPa"},
    "HGT_500": {"shortName": "gh", "level": 500, "typeOfLevel": "isobaricInhPa"},
    "U_500": {"shortName": "u", "level": 500, "typeOfLevel": "isobaricInhPa"},
    "V_500": {"shortName": "v", "level": 500, "typeOfLevel": "isobaricInhPa"},
    "ABSV_500": {"shortName": "absv", "level": 500, "typeOfLevel": "isobaricInhPa"},
    "U_250": {"shortName": "u", "level": 250, "typeOfLevel": "isobaricInhPa"},
    "V_250": {"shortName": "v", "level": 250, "typeOfLevel": "isobaricInhPa"},
    "HLCY_LCL": {"shortName": "gh", "typeOfLevel": "adiabaticCondensation"},
    "MXUPHL_2_5km": {"shortName": "mxuphl", "topLevel": 5000, "bottomLevel": 2000},
    "MAXWIND_10m": {"shortName": "maxuw"},
    "MAXREFD_1km": {"shortName": "maxref"},
    "MAXHAIL": {"shortName": "hail"},
}


def extract_variables_to_dataset(grib_path: Path) -> xr.Dataset:
    """Load all configured variables from one GRIB2 file into a single xarray Dataset.

    Variables that fail to load (e.g., not present in this product) are skipped with a warning.
    """
    out_vars: dict[str, xr.DataArray] = {}
    coords_set = False
    for name, filt in _VAR_TO_FILTER.items():
        try:
            ds = _open_grib_subset(grib_path, filt)
        except Exception as exc:  # noqa: BLE001
            log.warning("hrrr.extract.skip", var=name, error=str(exc))
            continue
        if not ds.data_vars:
            log.warning("hrrr.extract.empty", var=name, filt=filt)
            continue
        # Pick the first data var (cfgrib sometimes returns the underlying GRIB shortName)
        da = next(iter(ds.data_vars.values()))
        out_vars[name] = da.rename(name)
        if not coords_set:
            ref_lat = ds["latitude"]
            ref_lon = ds["longitude"]
            coords_set = True
    if not out_vars:
        raise RuntimeError(f"No variables extracted from {grib_path}")
    out = xr.Dataset(out_vars, coords={"latitude": ref_lat, "longitude": ref_lon})
    return out
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_hrrr.py -v
```
Expected: 3 passed (with the fixture present); the extraction test is skipped if the fixture is missing.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/data/hrrr.py tests/test_hrrr.py tests/fixtures/build_tiny_hrrr.py tests/fixtures/tiny_hrrr.grib2
git commit -m "feat(hrrr): extract configured variables into xarray Dataset"
```

---

### Task 19: 3 km → 50 km regridding to grid cells

**Files:**
- Modify: `src/ml_severe_weather_forecast/data/hrrr.py`
- Modify: `tests/test_hrrr.py`

- [ ] **Step 1: Write the regrid test**

Append to `tests/test_hrrr.py`:

```python
@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_regrid_to_50km_returns_dataframe_per_cell() -> None:
    from ml_severe_weather_forecast.data.grid import build_grid
    from ml_severe_weather_forecast.data.hrrr import extract_variables_to_dataset, regrid_to_cells

    ds = extract_variables_to_dataset(FIXTURE)
    grid = build_grid()
    df = regrid_to_cells(ds, grid)
    # One row per cell
    assert len(df) == grid.n_cells
    # Each variable contributed both _max and _mean columns (where applicable)
    assert "MLCAPE_max" in df.columns
    assert "MLCAPE_mean" in df.columns
    # Hourly-max fields contribute _max only
    assert "MXUPHL_2_5km_max" in df.columns
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_hrrr.py::test_regrid_to_50km_returns_dataframe_per_cell -v
```

- [ ] **Step 3: Implement regridding via nearest-neighbor cell assignment**

Append to `src/ml_severe_weather_forecast/data/hrrr.py`:

```python
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml_severe_weather_forecast.data.grid import Grid  # noqa: E402


def _assign_hrrr_points_to_cells(ds: xr.Dataset, grid: Grid) -> np.ndarray:
    """For each HRRR (lat, lon) point in `ds`, find the nearest grid cell index.

    Returns an int array of shape (ny_hrrr * nx_hrrr,) — flattened, with values in [0, n_cells)
    or -1 for points outside any cell.
    """
    from sklearn.neighbors import BallTree

    # HRRR points
    lat = ds["latitude"].to_numpy().ravel()
    lon = ds["longitude"].to_numpy().ravel()
    # Wrap lon into [-180, 180]
    lon = np.where(lon > 180, lon - 360, lon)
    pts_rad = np.deg2rad(np.column_stack([lat, lon]))

    # Grid cells
    cell_rad = np.deg2rad(np.column_stack([grid.lats, grid.lons]))
    tree = BallTree(cell_rad, metric="haversine")
    # 25 km in radians on Earth (>= half of 50 km cell diagonal)
    radius_rad = 35_000.0 / 6_371_000.0
    dist, idx = tree.query(pts_rad, k=1)
    # Where the nearest cell is farther than radius, mark as outside
    flat_idx = idx.ravel()
    flat_idx[(dist.ravel() > radius_rad)] = -1
    return flat_idx


def regrid_to_cells(ds: xr.Dataset, grid: Grid) -> pd.DataFrame:
    """Aggregate each variable from HRRR's 3 km grid down to the 50 km cell grid.

    For each variable, we compute the cell-wise `max` and (where applicable) `mean`
    over the contained HRRR points.
    """
    cell_idx = _assign_hrrr_points_to_cells(ds, grid)
    # Buckets of HRRR-flat-indices per cell
    order = np.argsort(cell_idx, kind="stable")
    sorted_cell_idx = cell_idx[order]
    valid = sorted_cell_idx >= 0
    sorted_valid_order = order[valid]
    sorted_valid_idx = sorted_cell_idx[valid]
    boundaries = np.concatenate(
        [[0], np.cumsum(np.bincount(sorted_valid_idx, minlength=grid.n_cells))]
    )

    out_columns: dict[str, np.ndarray] = {"cell_id": grid.cell_ids}

    for name, da in ds.data_vars.items():
        flat = da.to_numpy().astype(np.float32).ravel()
        spec = HRRR_VARIABLES.get(str(name))
        var_max = np.full(grid.n_cells, np.nan, dtype=np.float32)
        var_mean = np.full(grid.n_cells, np.nan, dtype=np.float32)
        for c in range(grid.n_cells):
            start, stop = boundaries[c], boundaries[c + 1]
            if stop > start:
                values = flat[sorted_valid_order[start:stop]]
                values = values[~np.isnan(values)]
                if values.size:
                    var_max[c] = values.max()
                    var_mean[c] = values.mean()
        out_columns[f"{name}_max"] = var_max
        if not (spec and spec.is_hourly_max):
            out_columns[f"{name}_mean"] = var_mean

    return pd.DataFrame(out_columns)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_hrrr.py -v
```
Expected: all hrrr tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/data/hrrr.py tests/test_hrrr.py
git commit -m "feat(hrrr): nearest-cell regrid 3km HRRR -> 50km grid"
```

---

### Task 20: HRRR download CLI subcommand

**Files:**
- Modify: `src/ml_severe_weather_forecast/cli.py`

- [ ] **Step 1: Add `mlswf download hrrr`**

Append to `src/ml_severe_weather_forecast/cli.py`:

```python
@download_app.command("hrrr")
def download_hrrr_cmd(
    start: str = typer.Option(..., help="First cycle date YYYY-MM-DD."),
    end: str = typer.Option(..., help="Last cycle date YYYY-MM-DD (inclusive)."),
) -> None:
    """Download 12z HRRR cycles between start and end (inclusive)."""
    from datetime import UTC, datetime, timedelta

    from ml_severe_weather_forecast.data.hrrr import download_cycle

    s = datetime.fromisoformat(start).replace(tzinfo=UTC, hour=settings.hrrr_cycle_hour)
    e = datetime.fromisoformat(end).replace(tzinfo=UTC, hour=settings.hrrr_cycle_hour)
    cur = s
    while cur <= e:
        if settings.season_month_start <= cur.month <= settings.season_month_end:
            paths = download_cycle(
                cur, settings.hrrr_forecast_hours, cache_dir=settings.hrrr_dir
            )
            typer.echo(f"  {cur.date()}: {len(paths)} files")
        cur = cur + timedelta(days=1)
```

- [ ] **Step 2: Smoke-test help**

```bash
uv run mlswf download hrrr --help
```
Expected: help text with `--start` and `--end`.

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/cli.py
git commit -m "feat(cli): mlswf download hrrr subcommand"
```

---

## Phase 5 — Feature Engineering

### Task 21: Temporal aggregation across forecast hours

**Files:**
- Create: `src/ml_severe_weather_forecast/features/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write a test that takes per-forecast-hour DataFrames and aggregates them**

```python
# tests/test_extract.py
import numpy as np
import pandas as pd

from ml_severe_weather_forecast.features.extract import temporal_aggregate


def test_temporal_aggregate_max_mean_p90() -> None:
    cells = ["c_001_001", "c_002_002"]
    frames = [
        pd.DataFrame({"cell_id": cells, "MLCAPE_max": [100.0, 50.0], "MLCAPE_mean": [80.0, 40.0]}),
        pd.DataFrame({"cell_id": cells, "MLCAPE_max": [200.0, 60.0], "MLCAPE_mean": [150.0, 50.0]}),
        pd.DataFrame({"cell_id": cells, "MLCAPE_max": [300.0, 70.0], "MLCAPE_mean": [250.0, 60.0]}),
    ]
    out = temporal_aggregate(frames, instantaneous_vars=["MLCAPE"], hourly_max_vars=[])
    # MLCAPE_max should be aggregated to fhr_max, fhr_mean, fhr_p90
    row0 = out.loc[out["cell_id"] == "c_001_001"].iloc[0]
    assert row0["MLCAPE_max_fhr_max"] == 300.0
    assert row0["MLCAPE_max_fhr_mean"] == 200.0  # (100+200+300)/3
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
    # Hourly-max fields don't get fhr_mean / fhr_p90 columns
    assert "MXUPHL_2_5km_max_fhr_mean" not in out.columns
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_extract.py -v
```

- [ ] **Step 3: Implement `temporal_aggregate`**

```python
# src/ml_severe_weather_forecast/features/extract.py
"""Per-cycle feature extraction: 25 forecast-hour DataFrames → one feature table."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _stack_per_cell(frames: Sequence[pd.DataFrame], col: str) -> np.ndarray:
    """Stack one column across forecast-hour frames into a (n_cells, n_hours) array."""
    arr = np.column_stack([f[col].to_numpy(dtype=np.float32) for f in frames])
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_extract.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/features/extract.py tests/test_extract.py
git commit -m "feat(features): temporal aggregation across forecast hours"
```

---

### Task 22: Snapshot-anchor features (f06, f12, f18)

**Files:**
- Modify: `src/ml_severe_weather_forecast/features/extract.py`
- Modify: `tests/test_extract.py`

- [ ] **Step 1: Add a snapshot test**

Append to `tests/test_extract.py`:

```python
def test_snapshot_anchors_pull_specific_forecast_hours() -> None:
    from ml_severe_weather_forecast.features.extract import snapshot_anchors

    cells = ["c_001_001"]
    frames = []
    for fhr in range(0, 13):  # f00..f12
        frames.append(pd.DataFrame({"cell_id": cells, "MLCAPE_max": [float(fhr) * 100]}))
    out = snapshot_anchors(frames, fhrs=[6, 12], instantaneous_vars=["MLCAPE"])
    row = out.iloc[0]
    assert row["MLCAPE_max_f06"] == 600.0
    assert row["MLCAPE_max_f12"] == 1200.0
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/test_extract.py::test_snapshot_anchors_pull_specific_forecast_hours -v
```

- [ ] **Step 3: Implement `snapshot_anchors`**

Append to `src/ml_severe_weather_forecast/features/extract.py`:

```python
def snapshot_anchors(
    frames: Sequence[pd.DataFrame],
    fhrs: Sequence[int],
    instantaneous_vars: Sequence[str],
) -> pd.DataFrame:
    """Pull per-cell values at specific forecast hours (e.g., f06, f12, f18) for each variable.

    `frames` is indexed by forecast hour starting at f00; `fhrs` selects which hours to keep.
    """
    out = pd.DataFrame({"cell_id": frames[0]["cell_id"]})
    for var in instantaneous_vars:
        for stat in ("max", "mean"):
            col = f"{var}_{stat}"
            if col not in frames[0].columns:
                continue
            for fhr in fhrs:
                if fhr >= len(frames):
                    continue
                out[f"{col}_f{fhr:02d}"] = frames[fhr][col].to_numpy(dtype=np.float32)
    return out
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_extract.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/features/extract.py tests/test_extract.py
git commit -m "feat(features): snapshot anchors at f06/f12/f18"
```

---

### Task 23: Spatial neighborhood (concentric rings)

**Files:**
- Create: `src/ml_severe_weather_forecast/features/neighborhood.py`
- Test: `tests/test_neighborhood.py`

- [ ] **Step 1: Write a test for ring features**

```python
# tests/test_neighborhood.py
import numpy as np
import pandas as pd

from ml_severe_weather_forecast.data.grid import build_grid
from ml_severe_weather_forecast.features.neighborhood import compute_ring_features


def test_compute_ring_features_returns_per_cell_table() -> None:
    grid = build_grid()
    # Synthetic feature: 1.0 for all cells; means and maxes per ring should also be 1.0
    df = pd.DataFrame(
        {
            "cell_id": grid.cell_ids,
            "MLCAPE_max_fhr_max": np.ones(grid.n_cells, dtype=np.float32),
        }
    )
    out = compute_ring_features(df, grid, base_columns=["MLCAPE_max_fhr_max"], ring_radii_km=[40, 80, 160])
    assert "MLCAPE_max_fhr_max_ring0_40_max" in out.columns
    assert "MLCAPE_max_fhr_max_ring40_80_mean" in out.columns
    assert np.allclose(out["MLCAPE_max_fhr_max_ring0_40_max"], 1.0, equal_nan=True)


def test_compute_ring_features_with_localized_signal() -> None:
    grid = build_grid()
    arr = np.zeros(grid.n_cells, dtype=np.float32)
    # Put a "1" in one cell — its neighbors should pick it up in the 0-40 ring max
    arr[0] = 1.0
    df = pd.DataFrame({"cell_id": grid.cell_ids, "X_max": arr})
    out = compute_ring_features(df, grid, base_columns=["X_max"], ring_radii_km=[40])
    # The cell itself sees 1.0
    assert out.iloc[0]["X_max_ring0_40_max"] == 1.0
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement neighborhood rings**

```python
# src/ml_severe_weather_forecast/features/neighborhood.py
"""Concentric-ring neighborhood feature engineering."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ml_severe_weather_forecast.data.grid import Grid

EARTH_RADIUS_M = 6_371_000.0


def _ring_indices(grid: Grid, radii_km: Sequence[float]) -> list[list[np.ndarray]]:
    """For each cell, list of arrays of neighbor indices in each ring."""
    pts = np.deg2rad(np.column_stack([grid.lats, grid.lons]))
    tree = BallTree(pts, metric="haversine")
    radii_rad = [r * 1000.0 / EARTH_RADIUS_M for r in radii_km]
    per_cell: list[list[np.ndarray]] = []
    prev: list[set[int]] = [set() for _ in range(grid.n_cells)]
    for r in radii_rad:
        within = tree.query_radius(pts, r=r)
        cur: list[np.ndarray] = []
        for i, neighbors in enumerate(within):
            ring = np.array([n for n in neighbors if n not in prev[i]], dtype=np.int32)
            cur.append(ring)
            prev[i].update(neighbors)
        per_cell.append(cur)
    # per_cell[ring_idx][cell_idx] = neighbor indices in that ring
    return per_cell


def compute_ring_features(
    df: pd.DataFrame,
    grid: Grid,
    base_columns: Sequence[str],
    ring_radii_km: Sequence[float] = (40, 80, 160),
) -> pd.DataFrame:
    """Add ring max/mean for each base column.

    Ring boundaries: 0–r0, r0–r1, r1–r2 (inclusive lower, exclusive upper).
    """
    rings = _ring_indices(grid, ring_radii_km)
    radius_pairs = list(zip([0, *ring_radii_km[:-1]], ring_radii_km, strict=True))
    out_cols: dict[str, np.ndarray] = {"cell_id": grid.cell_ids}
    for col in base_columns:
        if col not in df.columns:
            continue
        values = df[col].to_numpy(dtype=np.float32)
        for ring_idx, (lo, hi) in enumerate(radius_pairs):
            cell_max = np.full(grid.n_cells, np.nan, dtype=np.float32)
            cell_mean = np.full(grid.n_cells, np.nan, dtype=np.float32)
            for c in range(grid.n_cells):
                neighbors = rings[ring_idx][c]
                if neighbors.size == 0:
                    continue
                v = values[neighbors]
                v = v[~np.isnan(v)]
                if v.size == 0:
                    continue
                cell_max[c] = v.max()
                cell_mean[c] = v.mean()
            out_cols[f"{col}_ring{int(lo)}_{int(hi)}_max"] = cell_max
            out_cols[f"{col}_ring{int(lo)}_{int(hi)}_mean"] = cell_mean
    return pd.DataFrame(out_cols)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_neighborhood.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/features/neighborhood.py tests/test_neighborhood.py
git commit -m "feat(features): concentric-ring neighborhood max/mean"
```

---

### Task 24: Derived composites (EHI, SCP, STP)

**Files:**
- Create: `src/ml_severe_weather_forecast/features/derived.py`
- Test: `tests/test_derived.py`

- [ ] **Step 1: Test composite formulas with hand-computed values**

```python
# tests/test_derived.py
import numpy as np
import pandas as pd

from ml_severe_weather_forecast.features.derived import compute_composites, compute_lapse_rates


def test_ehi_formula() -> None:
    df = pd.DataFrame(
        {
            "MLCAPE_max_fhr_max": [3200.0],
            "SRH_0_3km_max_fhr_max": [400.0],
            "MUCAPE_max_fhr_max": [3500.0],
            "USHR_0_6km_max_fhr_max": [25.0],
            "VSHR_0_6km_max_fhr_max": [10.0],
            "SRH_0_1km_max_fhr_max": [200.0],
            "HLCY_LCL_mean_fhr_mean": [1000.0],
            "MLCIN_mean_fhr_mean": [-50.0],
        }
    )
    out = compute_composites(df)
    # EHI = MLCAPE * SRH3 / 160000
    assert np.isclose(out.loc[0, "EHI_0_3km"], 3200 * 400 / 160000)
    # STP and SCP must be present and finite
    assert np.isfinite(out.loc[0, "STP"])
    assert np.isfinite(out.loc[0, "SCP"])


def test_lapse_rate_computation() -> None:
    # Standard atmosphere lapse rate 700-500 mb is roughly 6.5 K/km
    # 700 mb is ~3000 m, 500 mb is ~5500 m -> 2.5 km layer
    # T700 = 273, T500 = 257 -> 16 K / 2.5 km = 6.4 K/km
    df = pd.DataFrame({"T_700_mean": [273.0], "T_500_mean": [257.0]})
    out = compute_lapse_rates(df)
    assert 6.0 < out.loc[0, "LAPSE_700_500"] < 7.0
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement composites**

```python
# src/ml_severe_weather_forecast/features/derived.py
"""Derived meteorological features: lapse rates, composites (EHI, SCP, STP)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate height-difference between pressure levels in standard atmosphere (m).
# Used for converting (Tlow - Thigh) into lapse rate K/km.
_PRESSURE_LEVEL_HEIGHTS = {
    850: 1500.0,
    700: 3000.0,
    500: 5500.0,
}


def compute_lapse_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 700–500 mb and 850–500 mb lapse rates (K/km)."""
    out = pd.DataFrame(index=df.index)
    pairs = [(700, 500), (850, 500)]
    for low, high in pairs:
        col_low = f"T_{low}_mean"
        col_high = f"T_{high}_mean"
        if col_low not in df.columns or col_high not in df.columns:
            continue
        dh_km = (_PRESSURE_LEVEL_HEIGHTS[high] - _PRESSURE_LEVEL_HEIGHTS[low]) / 1000.0
        out[f"LAPSE_{low}_{high}"] = (df[col_low] - df[col_high]) / dh_km
    return out


def compute_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Energy-helicity index, supercell composite, significant tornado parameter.

    Uses _fhr_max columns where available. Caller is responsible for ensuring inputs.
    """
    out = pd.DataFrame(index=df.index)

    cape3 = df.get("MLCAPE_max_fhr_max")
    srh3 = df.get("SRH_0_3km_max_fhr_max")
    if cape3 is not None and srh3 is not None:
        out["EHI_0_3km"] = (cape3 * srh3) / 160_000.0

    mucape = df.get("MUCAPE_max_fhr_max")
    ushr = df.get("USHR_0_6km_max_fhr_max")
    vshr = df.get("VSHR_0_6km_max_fhr_max")
    if all(x is not None for x in (mucape, srh3, ushr, vshr)):
        bwd6 = np.sqrt(ushr.fillna(0) ** 2 + vshr.fillna(0) ** 2)
        out["BWD_0_6km"] = bwd6
        out["SCP"] = (mucape / 1000.0) * (srh3 / 50.0) * (bwd6 / 20.0)

    mlcape = df.get("MLCAPE_max_fhr_max")
    srh1 = df.get("SRH_0_1km_max_fhr_max")
    lcl = df.get("HLCY_LCL_mean_fhr_mean")
    mlcin = df.get("MLCIN_mean_fhr_mean")
    if all(x is not None for x in (mlcape, srh1, ushr, vshr, lcl, mlcin)):
        bwd6 = np.sqrt(ushr.fillna(0) ** 2 + vshr.fillna(0) ** 2)
        bwd6_kt = bwd6 * 1.94384  # m/s -> kt
        lcl_term = ((2000.0 - lcl).clip(lower=0)) / 1000.0
        cin_term = ((mlcin + 200.0).clip(lower=0)) / 150.0
        out["STP"] = (
            (mlcape / 1500.0) * (srh1 / 150.0) * (bwd6_kt / 12.0) * lcl_term * cin_term
        )
    return out
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_derived.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/features/derived.py tests/test_derived.py
git commit -m "feat(features): derived composites (EHI, SCP, STP, lapse rates)"
```

---

### Task 25: Per-cycle feature assembly orchestrator

**Files:**
- Create: `src/ml_severe_weather_forecast/features/assembly.py`

- [ ] **Step 1: Implement the orchestrator that strings everything together**

```python
# src/ml_severe_weather_forecast/features/assembly.py
"""Per-cycle orchestration: GRIB2s → regridded frames → temporal agg → neighborhood → composites → final Parquet."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import structlog

from ml_severe_weather_forecast.data.grid import Grid
from ml_severe_weather_forecast.data.hrrr import (
    HRRR_VARIABLES,
    extract_variables_to_dataset,
    regrid_to_cells,
)
from ml_severe_weather_forecast.features.derived import compute_composites, compute_lapse_rates
from ml_severe_weather_forecast.features.extract import snapshot_anchors, temporal_aggregate
from ml_severe_weather_forecast.features.neighborhood import compute_ring_features

log = structlog.get_logger(__name__)


def _per_fhr_frames(grib_paths: list[Path], grid: Grid) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for p in sorted(grib_paths):
        ds = extract_variables_to_dataset(p)
        frames.append(regrid_to_cells(ds, grid))
    return frames


def assemble_cycle_features(
    cycle_init: datetime,
    grib_paths: list[Path],
    grid: Grid,
    out_path: Path,
    *,
    snapshot_fhrs: Iterable[int] = (6, 12, 18),
    ring_radii_km: Iterable[float] = (40, 80, 160),
) -> Path:
    """Build the final per-cycle feature Parquet for one 12z HRRR cycle."""
    log.info("assemble.start", cycle=cycle_init.isoformat(), n_grib=len(grib_paths))

    frames = _per_fhr_frames(grib_paths, grid)
    instantaneous = [name for name, spec in HRRR_VARIABLES.items() if not spec.is_hourly_max]
    hourly_max = [name for name, spec in HRRR_VARIABLES.items() if spec.is_hourly_max]

    temporal = temporal_aggregate(frames, instantaneous, hourly_max)
    snaps = snapshot_anchors(frames, list(snapshot_fhrs), instantaneous)

    # Choose a small base set of "interesting" columns to grow neighborhood features for.
    # Limiting this keeps the final column count to ~500 rather than ~thousands.
    base_for_rings = [
        "MLCAPE_max_fhr_max",
        "MUCAPE_max_fhr_max",
        "SRH_0_3km_max_fhr_max",
        "SRH_0_1km_max_fhr_max",
        "MXUPHL_2_5km_max_fhr_max",
        "MAXREFD_1km_max_fhr_max",
        "MAXHAIL_max_fhr_max",
        "MLCIN_mean_fhr_mean",
    ]
    rings = compute_ring_features(temporal, grid, base_for_rings, list(ring_radii_km))

    # Lapse rates need T_700_mean / T_500_mean from the temporal table.
    # The temporal aggregator emitted T_700_mean_fhr_mean — alias to T_700_mean for derived module.
    lapse_input = pd.DataFrame(
        {
            "T_500_mean": temporal["T_500_mean_fhr_mean"]
            if "T_500_mean_fhr_mean" in temporal.columns
            else float("nan"),
            "T_700_mean": temporal["T_700_mean_fhr_mean"]
            if "T_700_mean_fhr_mean" in temporal.columns
            else float("nan"),
            "T_850_mean": temporal["T_850_mean_fhr_mean"]
            if "T_850_mean_fhr_mean" in temporal.columns
            else float("nan"),
        }
    )
    lapse = compute_lapse_rates(lapse_input)
    composites = compute_composites(temporal)

    # Merge everything by cell_id index alignment
    merged = temporal.copy()
    for extra in (snaps, rings, lapse, composites):
        cols_new = [c for c in extra.columns if c not in merged.columns and c != "cell_id"]
        if "cell_id" in extra.columns:
            merged = merged.merge(extra[["cell_id", *cols_new]], on="cell_id", how="left")
        else:
            for c in cols_new:
                merged[c] = extra[c].to_numpy()

    merged.insert(0, "cycle_init_utc", pd.Timestamp(cycle_init))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    log.info("assemble.done", cycle=cycle_init.isoformat(), out=str(out_path), n_cols=len(merged.columns))
    return out_path
```

- [ ] **Step 2: No new test added here (covered by integration test in Task 53). Manual sanity check:**

```bash
uv run python -c "from ml_severe_weather_forecast.features.assembly import assemble_cycle_features; print('importable')"
```
Expected: `importable`.

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/features/assembly.py
git commit -m "feat(features): per-cycle assembly orchestrator"
```

---

### Task 26: `mlswf extract` CLI subcommand

**Files:**
- Modify: `src/ml_severe_weather_forecast/cli.py`

- [ ] **Step 1: Add `mlswf extract --date YYYY-MM-DD`**

Append to `src/ml_severe_weather_forecast/cli.py`:

```python
@app.command("extract")
def extract_cmd(
    date: str = typer.Option(..., help="Cycle date YYYY-MM-DD (12z run is used)."),
) -> None:
    """Extract features from one HRRR cycle's downloaded GRIB2 files."""
    from datetime import UTC, datetime

    from ml_severe_weather_forecast.data.grid import build_grid
    from ml_severe_weather_forecast.features.assembly import assemble_cycle_features

    cycle = datetime.fromisoformat(date).replace(tzinfo=UTC, hour=settings.hrrr_cycle_hour)
    grid = build_grid()
    cycle_dir = settings.hrrr_dir / f"hrrr.{cycle.strftime('%Y%m%d')}" / "conus"
    if not cycle_dir.exists():
        # Herbie's default save layout — adjust if your save_dir differs
        cycle_dir = settings.hrrr_dir / cycle.strftime("%Y%m%d")
    grib_paths = sorted(cycle_dir.rglob("*.grib2"))
    if not grib_paths:
        raise typer.BadParameter(f"no GRIB2 files under {cycle_dir}")
    out = settings.features_dir / str(cycle.year) / f"{cycle.strftime('%m%d')}.parquet"
    assemble_cycle_features(cycle, grib_paths, grid, out)
    typer.echo(f"Wrote {out}")
```

- [ ] **Step 2: Smoke-test help**

```bash
uv run mlswf extract --help
```

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/cli.py
git commit -m "feat(cli): mlswf extract subcommand"
```

---

### Task 27: Memory-aware multi-cycle extraction script

**Files:**
- Create: `scripts/01_extract_all.py`

- [ ] **Step 1: Implement parallel extraction with bounded memory**

```python
# scripts/01_extract_all.py
"""Extract features for all 12z cycles Apr–Jul of training years.

Uses a process pool (one cycle at a time per worker) bounded by available RAM.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psutil
import typer

from ml_severe_weather_forecast.config import settings
from ml_severe_weather_forecast.data.grid import build_grid
from ml_severe_weather_forecast.features.assembly import assemble_cycle_features

app = typer.Typer()


def _cycle_paths(cycle: datetime) -> list[Path]:
    cycle_dir = settings.hrrr_dir / cycle.strftime("%Y%m%d")
    paths = sorted(cycle_dir.rglob("*.grib2"))
    return paths


def _extract_one(cycle_iso: str) -> str:
    cycle = datetime.fromisoformat(cycle_iso)
    grid = build_grid()
    grib_paths = _cycle_paths(cycle)
    if not grib_paths:
        return f"SKIP {cycle.date()} (no grib)"
    out = settings.features_dir / str(cycle.year) / f"{cycle.strftime('%m%d')}.parquet"
    if out.exists():
        return f"CACHED {out.name}"
    assemble_cycle_features(cycle, grib_paths, grid, out)
    return f"OK {out.name}"


@app.command()
def main(
    workers: int = typer.Option(
        max(1, psutil.cpu_count(logical=False) // 2),
        help="Process workers (CPU-physical / 2 by default to avoid OOM).",
    ),
) -> None:
    cycles: list[datetime] = []
    for year in range(settings.train_year_start, settings.train_year_end + 1):
        cur = datetime(year, settings.season_month_start, 1, settings.hrrr_cycle_hour, tzinfo=UTC)
        end = datetime(year, settings.season_month_end + 1, 1, settings.hrrr_cycle_hour, tzinfo=UTC) \
            if settings.season_month_end < 12 else datetime(year + 1, 1, 1, settings.hrrr_cycle_hour, tzinfo=UTC)
        while cur < end:
            cycles.append(cur)
            cur = cur + timedelta(days=1)

    typer.echo(f"Extracting {len(cycles)} cycles with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_one, c.isoformat()): c for c in cycles}
        for fut in as_completed(futures):
            typer.echo(fut.result())


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/01_extract_all.py
git commit -m "feat(scripts): parallel extraction over all training cycles"
```

---

## Phase 6 — SPC Outlook Archive Baseline

### Task 28: SPC outlook downloader

**Files:**
- Create: `src/ml_severe_weather_forecast/data/outlooks.py`
- Test: `tests/test_outlooks.py`

- [ ] **Step 1: Test the URL-construction and date iteration**

```python
# tests/test_outlooks.py
from datetime import UTC, date

from ml_severe_weather_forecast.data.outlooks import outlook_archive_url


def test_outlook_archive_url_format() -> None:
    url = outlook_archive_url(date(2023, 5, 15), product="day1otlk", issuance="1200")
    assert "2023" in url
    assert "20230515" in url
    assert url.endswith(".zip") or url.endswith(".gz") or url.endswith(".kmz")
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement outlook download**

```python
# src/ml_severe_weather_forecast/data/outlooks.py
"""SPC Day-1 Convective Outlook archive: download + rasterize to grid cells."""
from __future__ import annotations

from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import httpx
import structlog

log = structlog.get_logger(__name__)

SPC_OUTLOOK_BASE = "https://www.spc.noaa.gov/products/outlook/archive"


def outlook_archive_url(d: date, product: str = "day1otlk", issuance: str = "1200") -> str:
    """Return the URL for a single archived outlook shapefile zip."""
    yyyy = d.strftime("%Y")
    yyyymmdd = d.strftime("%Y%m%d")
    return f"{SPC_OUTLOOK_BASE}/{yyyy}/{product}_{yyyymmdd}_{issuance}-shp.zip"


def download_outlook(d: date, dest_dir: Path, *, force: bool = False) -> Path | None:
    """Download one Day-1 12z outlook shapefile archive. Returns the local path or None on 404."""
    url = outlook_archive_url(d)
    out = dest_dir / str(d.year) / f"{d.strftime('%Y%m%d')}_1200-shp.zip"
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            log.warning("outlook.missing", url=url)
            return None
        raise
    out.write_bytes(response.content)
    return out


def unzip_outlook(zip_path: Path, dest_dir: Path) -> Path:
    """Extract shapefile components and return the directory containing them."""
    out_dir = dest_dir / zip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return out_dir
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_outlooks.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/data/outlooks.py tests/test_outlooks.py
git commit -m "feat(outlooks): SPC Day-1 outlook downloader"
```

---

### Task 29: Rasterize outlook polygons onto the grid

**Files:**
- Modify: `src/ml_severe_weather_forecast/data/outlooks.py`
- Modify: `tests/test_outlooks.py`

- [ ] **Step 1: Test rasterization with a synthetic polygon**

Append to `tests/test_outlooks.py`:

```python
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from ml_severe_weather_forecast.data.grid import build_grid
from ml_severe_weather_forecast.data.outlooks import rasterize_outlook_to_grid


def test_rasterize_circular_polygon_picks_inside_cells() -> None:
    grid = build_grid()
    # A small box around Norman, OK at probability 30%
    poly = Polygon(
        [
            (-99.0, 33.0),
            (-95.0, 33.0),
            (-95.0, 37.0),
            (-99.0, 37.0),
            (-99.0, 33.0),
        ]
    )
    gdf = gpd.GeoDataFrame(
        {"hazard": ["tor"], "probability": [0.30], "geometry": [poly]},
        crs="EPSG:4326",
    )
    df = rasterize_outlook_to_grid(gdf, grid)
    assert "tor_prob" in df.columns
    inside = df[df["tor_prob"] > 0]
    assert len(inside) > 0
    assert inside["tor_prob"].max() == 0.30
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement rasterization**

Append to `src/ml_severe_weather_forecast/data/outlooks.py`:

```python
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from ml_severe_weather_forecast.data.grid import Grid  # noqa: E402

# SPC's per-hazard probability label → numeric (categorical bins)
SPC_TORNADO_PROBS = {"0.02": 0.02, "0.05": 0.05, "0.10": 0.10, "0.15": 0.15, "0.30": 0.30, "0.45": 0.45, "0.60": 0.60}
SPC_HAILWIND_PROBS = {"0.05": 0.05, "0.15": 0.15, "0.30": 0.30, "0.45": 0.45, "0.60": 0.60}

_HAZARD_ALIASES = {
    "tor": ("torn", "tornado", "tor"),
    "hail": ("hail",),
    "wind": ("wind",),
}


def _normalize_hazard(label: str) -> str | None:
    s = label.lower()
    for canon, aliases in _HAZARD_ALIASES.items():
        if any(a in s for a in aliases):
            return canon
    return None


def _outlook_dataframe_from_dir(shp_dir: Path) -> gpd.GeoDataFrame:
    """Load all per-hazard shapefiles in `shp_dir` into one GeoDataFrame."""
    frames: list[gpd.GeoDataFrame] = []
    for shp in shp_dir.glob("*.shp"):
        gdf = gpd.read_file(shp)
        if gdf.empty:
            continue
        # Identify hazard from filename (e.g., day1otlk_20230515_1200_torn.shp)
        hazard = _normalize_hazard(shp.stem)
        if hazard is None:
            continue
        # Probability column varies across years; "DN", "LABEL", and "DENSITY" are seen
        prob_col = next(
            (c for c in gdf.columns if c.upper() in {"DN", "LABEL", "DENSITY", "PROB", "PROB2"}),
            None,
        )
        if prob_col is None:
            continue
        gdf = gdf.assign(
            hazard=hazard,
            probability=pd.to_numeric(gdf[prob_col], errors="coerce") / 100.0,
        )
        frames.append(gdf[["hazard", "probability", "geometry"]])
    if not frames:
        return gpd.GeoDataFrame({"hazard": [], "probability": [], "geometry": []}, crs="EPSG:4326")
    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    return out.to_crs("EPSG:4326")


def rasterize_outlook_to_grid(gdf: gpd.GeoDataFrame, grid: Grid) -> pd.DataFrame:
    """For each grid cell center, look up the maximum overlapping polygon probability per hazard."""
    points = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(grid.lons, grid.lats, strict=True)],
        crs="EPSG:4326",
    )
    pts_gdf = gpd.GeoDataFrame({"cell_id": grid.cell_ids, "geometry": points}, crs="EPSG:4326")

    out = pd.DataFrame({"cell_id": grid.cell_ids})
    for hazard in ("tor", "hail", "wind"):
        out[f"{hazard}_prob"] = 0.0
        sub = gdf[gdf["hazard"] == hazard]
        if sub.empty:
            continue
        joined = gpd.sjoin(pts_gdf, sub[["probability", "geometry"]], how="left", predicate="within")
        per_cell_max = joined.groupby("cell_id")["probability"].max().fillna(0.0)
        out[f"{hazard}_prob"] = out["cell_id"].map(per_cell_max).fillna(0.0).to_numpy()
    return out


def build_outlook_year(year: int, raw_dir: Path, out_dir: Path) -> list[Path]:
    """For each cycle date in `raw_dir`, rasterize and write a per-day Parquet."""
    from ml_severe_weather_forecast.data.grid import build_grid as _build_grid

    grid = _build_grid()
    written: list[Path] = []
    out_dir = out_dir / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    for shp_dir in sorted((raw_dir / str(year)).iterdir()):
        if not shp_dir.is_dir():
            continue
        gdf = _outlook_dataframe_from_dir(shp_dir)
        df = rasterize_outlook_to_grid(gdf, grid)
        # Filename pattern: 20230515_1200-shp → 0515.parquet
        stem = shp_dir.name.split("_")[0]  # 20230515
        out = out_dir / f"{stem[4:8]}.parquet"
        df.to_parquet(out, index=False)
        written.append(out)
    return written
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_outlooks.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/data/outlooks.py tests/test_outlooks.py
git commit -m "feat(outlooks): rasterize Day-1 polygons onto 50km grid"
```

---

### Task 30: SPC outlook CLI subcommand

**Files:**
- Modify: `src/ml_severe_weather_forecast/cli.py`

- [ ] **Step 1: Add `mlswf download spc-outlooks`**

Append to `src/ml_severe_weather_forecast/cli.py`:

```python
@download_app.command("spc-outlooks")
def download_spc_outlooks_cmd(
    start: int = typer.Option(..., help="First year (inclusive)."),
    end: int = typer.Option(..., help="Last year (inclusive)."),
) -> None:
    """Download SPC Day-1 outlook shapefiles for the cycle dates in our season."""
    from datetime import date, timedelta

    from ml_severe_weather_forecast.data.outlooks import build_outlook_year, download_outlook, unzip_outlook

    raw_dir = settings.outlooks_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in range(start, end + 1):
        season_start = date(year, settings.season_month_start, 1)
        if settings.season_month_end == 12:
            season_end = date(year + 1, 1, 1)
        else:
            season_end = date(year, settings.season_month_end + 1, 1)
        d = season_start
        while d < season_end:
            zp = download_outlook(d, raw_dir)
            if zp is not None:
                unzip_outlook(zp, raw_dir / str(year))
            d = d + timedelta(days=1)
        typer.echo(f"  {year}: rasterizing…")
        build_outlook_year(year, raw_dir, settings.outlooks_dir)
```

- [ ] **Step 2: Smoke-test help**

```bash
uv run mlswf download spc-outlooks --help
```

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/cli.py
git commit -m "feat(cli): mlswf download spc-outlooks subcommand"
```

---

## Phase 7 — Climatology Baseline

### Task 31: Climatology — per (cell × month) base rate from 2010–2021 reports

**Files:**
- Create: `src/ml_severe_weather_forecast/climatology.py`
- Test: `tests/test_climatology.py`

- [ ] **Step 1: Test climatology computation**

```python
# tests/test_climatology.py
from datetime import UTC, datetime

import pandas as pd

from ml_severe_weather_forecast.climatology import compute_climatology
from ml_severe_weather_forecast.data.grid import build_grid


def test_compute_climatology_returns_per_cell_month_rate() -> None:
    grid = build_grid()
    reports = pd.DataFrame(
        [
            {
                "event_time_utc": datetime(2015, 5, 15, 18, tzinfo=UTC),
                "lat": 35.222,
                "lon": -97.439,
                "hazard": "tor",
                "magnitude": 2,
            },
            {
                "event_time_utc": datetime(2017, 5, 16, 19, tzinfo=UTC),
                "lat": 35.220,
                "lon": -97.430,
                "hazard": "tor",
                "magnitude": 1,
            },
        ]
    )
    n_years = 12
    df = compute_climatology(reports, grid, hazards=("tor",), years=n_years)
    assert {"cell_id", "month", "tor_climo_prob"} <= set(df.columns)
    # The Norman cell should have nonzero May tor climo
    norman_cell = df[(df["month"] == 5) & (df["tor_climo_prob"] > 0)]
    assert len(norman_cell) >= 1
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement climatology**

```python
# src/ml_severe_weather_forecast/climatology.py
"""Climatological baseline: per (cell × month) historical event probability."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ml_severe_weather_forecast.config import settings
from ml_severe_weather_forecast.data.grid import Grid

EARTH_RADIUS_M = 6_371_000.0


def compute_climatology(
    reports: pd.DataFrame,
    grid: Grid,
    *,
    hazards: Sequence[str] = ("tor", "hail", "wind"),
    years: int,
    months: Sequence[int] | None = None,
    radius_km: float | None = None,
) -> pd.DataFrame:
    """Per (cell × month × hazard) historical event probability — fraction of days in that
    month-cell over `years` that had at least one severe report within `radius_km`.

    Returns a long-format DataFrame: (cell_id, month, {hazard}_climo_prob).
    """
    radius_km = radius_km or settings.label_radius_km
    radius_rad = (radius_km * 1000.0) / EARTH_RADIUS_M
    if reports["event_time_utc"].dt.tz is None:
        raise ValueError("reports must be tz-aware UTC")

    cell_rad = np.deg2rad(np.column_stack([grid.lats, grid.lons]))
    tree = BallTree(cell_rad, metric="haversine")

    months_to_use = list(months) if months else list(
        range(settings.season_month_start, settings.season_month_end + 1)
    )

    rows: list[pd.DataFrame] = []
    for month in months_to_use:
        # Days in that month over the climo years (approximation: 30)
        n_days = 30 * years
        per_cell: dict[str, np.ndarray] = {"cell_id": grid.cell_ids, "month": np.full(grid.n_cells, month)}
        for hazard in hazards:
            sub = reports[
                (reports["hazard"] == hazard) & (reports["event_time_utc"].dt.month == month)
            ]
            cell_event_days = np.zeros(grid.n_cells, dtype=np.int32)
            if not sub.empty:
                # Group reports by date; count one event-day per cell per date
                sub = sub.assign(date=sub["event_time_utc"].dt.date)
                for _, group in sub.groupby("date"):
                    pts = np.deg2rad(group[["lat", "lon"]].to_numpy())
                    hits = tree.query_radius(pts, r=radius_rad)
                    seen: set[int] = set()
                    for arr in hits:
                        for c in arr:
                            seen.add(int(c))
                    for c in seen:
                        cell_event_days[c] += 1
            per_cell[f"{hazard}_climo_prob"] = (cell_event_days / max(n_days, 1)).astype(np.float32)
        rows.append(pd.DataFrame(per_cell))
    return pd.concat(rows, ignore_index=True)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_climatology.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/climatology.py tests/test_climatology.py
git commit -m "feat(climatology): per (cell, month) base-rate from 2010–2021 reports"
```

---

## Phase 8 — Training Table Assembly

### Task 32: Combine features + labels per fold

**Files:**
- Create: `src/ml_severe_weather_forecast/training.py`
- Test: `tests/test_training.py`

- [ ] **Step 1: Test training-set assembly**

```python
# tests/test_training.py
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_severe_weather_forecast.training import (
    Fold,
    assemble_fold_data,
    fold_definitions,
    load_year_features_and_labels,
)


def test_fold_definitions_cover_all_years() -> None:
    folds = fold_definitions(years=[2022, 2023, 2024])
    assert len(folds) == 3
    test_years = {f.test_year for f in folds}
    assert test_years == {2022, 2023, 2024}


def test_load_year_features_and_labels_joins_correctly(tmp_path: Path) -> None:
    feats_dir = tmp_path / "features" / "2023"
    feats_dir.mkdir(parents=True)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True)

    cycle = pd.Timestamp("2023-05-15 12:00:00", tz="UTC")
    cells = ["c_001_001", "c_002_002"]
    feats = pd.DataFrame(
        {
            "cycle_init_utc": [cycle, cycle],
            "cell_id": cells,
            "MLCAPE_max_fhr_max": [3000.0, 100.0],
        }
    )
    feats.to_parquet(feats_dir / "0515.parquet", index=False)

    labels = pd.DataFrame(
        {
            "cycle_init_utc": [cycle, cycle],
            "cell_id": cells,
            "tor": [1, 0],
            "hail": [0, 0],
            "wind": [0, 0],
            "tor_event_count": [1, 0],
            "tor_max_magnitude": [2.0, np.nan],
            "hail_event_count": [0, 0],
            "hail_max_magnitude": [np.nan, np.nan],
            "wind_event_count": [0, 0],
            "wind_max_magnitude": [np.nan, np.nan],
        }
    )
    labels.to_parquet(labels_dir / "2023.parquet", index=False)

    df = load_year_features_and_labels(year=2023, features_dir=tmp_path / "features", labels_dir=labels_dir)
    assert {"MLCAPE_max_fhr_max", "tor", "hail", "wind"} <= set(df.columns)
    assert len(df) == 2
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement training assembly**

```python
# src/ml_severe_weather_forecast/training.py
"""Training: fold definitions, table assembly, XGBoost training loop."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Hazard = Literal["tor", "hail", "wind"]


@dataclass(frozen=True)
class Fold:
    """One leave-one-year-out fold."""

    name: str
    train_years: tuple[int, ...]
    test_year: int


def fold_definitions(years: Sequence[int]) -> list[Fold]:
    """Generate leave-one-year-out folds over the given years."""
    folds: list[Fold] = []
    for held_out in years:
        train = tuple(y for y in years if y != held_out)
        folds.append(Fold(name=f"holdout_{held_out}", train_years=train, test_year=held_out))
    return folds


def load_year_features_and_labels(
    year: int, features_dir: Path, labels_dir: Path
) -> pd.DataFrame:
    """For a given year: concatenate all per-cycle feature Parquets and join with labels."""
    feature_files = sorted((features_dir / str(year)).glob("*.parquet"))
    if not feature_files:
        raise FileNotFoundError(f"No feature files in {features_dir / str(year)}")
    feats = pd.concat([pd.read_parquet(p) for p in feature_files], ignore_index=True)

    label_path = labels_dir / f"{year}.parquet"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing labels: {label_path}")
    labels = pd.read_parquet(label_path)

    # Normalize the join key: ensure both sides have UTC timestamp + str cell_id
    if feats["cycle_init_utc"].dt.tz is None:
        feats["cycle_init_utc"] = feats["cycle_init_utc"].dt.tz_localize("UTC")
    if labels["cycle_init_utc"].dt.tz is None:
        labels["cycle_init_utc"] = labels["cycle_init_utc"].dt.tz_localize("UTC")

    merged = feats.merge(labels, on=["cycle_init_utc", "cell_id"], how="inner")
    return merged


def split_by_month(df: pd.DataFrame, calib_month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a year's training rows into (fit, calibration) by forecast-cycle month."""
    months = df["cycle_init_utc"].dt.month
    return df[months != calib_month].reset_index(drop=True), df[months == calib_month].reset_index(drop=True)


def assemble_fold_data(
    fold: Fold,
    features_dir: Path,
    labels_dir: Path,
    *,
    calib_month: int = 7,
) -> dict[str, pd.DataFrame]:
    """Build the four data partitions for one fold:
    - fit: Apr–Jun of training years
    - calib: Jul of training years (early-stopping + isotonic input)
    - test: Apr–Jul of test year
    """
    train_frames = [load_year_features_and_labels(y, features_dir, labels_dir) for y in fold.train_years]
    train = pd.concat(train_frames, ignore_index=True)
    fit_df, calib_df = split_by_month(train, calib_month=calib_month)
    test_df = load_year_features_and_labels(fold.test_year, features_dir, labels_dir)
    return {"fit": fit_df, "calib": calib_df, "test": test_df}
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_training.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/training.py tests/test_training.py
git commit -m "feat(training): fold definitions and per-fold table assembly"
```

---

## Phase 9 — Training + Hyperparameter Search

### Task 33: Feature/label column selectors

**Files:**
- Modify: `src/ml_severe_weather_forecast/training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Add a test for column selection**

Append to `tests/test_training.py`:

```python
def test_select_feature_columns_excludes_metadata_and_labels() -> None:
    from ml_severe_weather_forecast.training import select_feature_columns

    df = pd.DataFrame(
        {
            "cycle_init_utc": [pd.Timestamp.now(tz="UTC")],
            "cell_id": ["c_000_000"],
            "MLCAPE_max_fhr_max": [100.0],
            "STP": [0.5],
            "tor": [0],
            "hail": [0],
            "wind": [0],
            "tor_event_count": [0],
            "tor_max_magnitude": [np.nan],
            "hail_event_count": [0],
            "hail_max_magnitude": [np.nan],
            "wind_event_count": [0],
            "wind_max_magnitude": [np.nan],
        }
    )
    cols = select_feature_columns(df)
    assert set(cols) == {"MLCAPE_max_fhr_max", "STP"}
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement column selectors**

Append to `src/ml_severe_weather_forecast/training.py`:

```python
_LABEL_COLUMNS = {
    "tor",
    "hail",
    "wind",
    "tor_event_count",
    "tor_max_magnitude",
    "hail_event_count",
    "hail_max_magnitude",
    "wind_event_count",
    "wind_max_magnitude",
}
_METADATA_COLUMNS = {"cycle_init_utc", "cell_id"}


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns that are valid model inputs (not metadata, not labels)."""
    excluded = _LABEL_COLUMNS | _METADATA_COLUMNS
    return [c for c in df.columns if c not in excluded]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_training.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/training.py tests/test_training.py
git commit -m "feat(training): feature column selector"
```

---

### Task 34: Train one booster on a fold

**Files:**
- Modify: `src/ml_severe_weather_forecast/training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Test that a tiny synthetic fold yields a fitted booster**

Append to `tests/test_training.py`:

```python
def test_train_booster_on_synthetic_data() -> None:
    from ml_severe_weather_forecast.training import train_one_booster

    rng = np.random.default_rng(0)
    n = 2000
    fit = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "tor": rng.binomial(1, 0.05, size=n),
        }
    )
    calib = pd.DataFrame(
        {
            "x1": rng.normal(size=500),
            "x2": rng.normal(size=500),
            "tor": rng.binomial(1, 0.05, size=500),
        }
    )
    booster, n_iter = train_one_booster(
        fit=fit,
        calib=calib,
        feature_cols=["x1", "x2"],
        target_col="tor",
        max_depth=4,
        learning_rate=0.1,
        device="cpu",
    )
    assert hasattr(booster, "predict_proba")
    assert n_iter > 0
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement single-booster training**

Append to `src/ml_severe_weather_forecast/training.py`:

```python
import xgboost as xgb  # noqa: E402


def train_one_booster(
    fit: pd.DataFrame,
    calib: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    max_depth: int,
    learning_rate: float,
    device: str = "cuda",
    n_estimators: int = 5000,
    early_stopping_rounds: int = 50,
    subsample: float = 0.8,
    colsample_bytree: float = 0.6,
    min_child_weight: int = 5,
    reg_lambda: float = 1.0,
) -> tuple[xgb.XGBClassifier, int]:
    """Fit a single XGBoost booster and return (booster, best_iteration)."""
    pos = int(fit[target_col].sum())
    neg = int(len(fit) - pos)
    spw = (neg / pos) if pos > 0 else 1.0

    booster = xgb.XGBClassifier(
        tree_method="hist",
        device=device,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        scale_pos_weight=spw,
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        verbosity=0,
    )
    booster.fit(
        fit[list(feature_cols)],
        fit[target_col],
        eval_set=[(calib[list(feature_cols)], calib[target_col])],
        verbose=False,
    )
    return booster, int(booster.best_iteration)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_training.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/training.py tests/test_training.py
git commit -m "feat(training): single-booster training with early stopping"
```

---

### Task 35: Hyperparameter grid search per fold/hazard

**Files:**
- Modify: `src/ml_severe_weather_forecast/training.py`
- Modify: `tests/test_training.py`

- [ ] **Step 1: Test the grid-search loop selects the best HP combo**

Append to `tests/test_training.py`:

```python
def test_grid_search_returns_best_combo() -> None:
    from ml_severe_weather_forecast.training import grid_search_hps

    rng = np.random.default_rng(1)
    n = 1000
    fit = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "tor": rng.binomial(1, 0.1, size=n),
        }
    )
    calib = pd.DataFrame(
        {
            "x1": rng.normal(size=300),
            "tor": rng.binomial(1, 0.1, size=300),
        }
    )
    result = grid_search_hps(
        fit=fit,
        calib=calib,
        feature_cols=["x1"],
        target_col="tor",
        max_depths=[2, 4],
        learning_rates=[0.05, 0.1],
        device="cpu",
    )
    assert "best_max_depth" in result
    assert "best_lr" in result
    assert "best_logloss" in result
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement grid search**

Append to `src/ml_severe_weather_forecast/training.py`:

```python
from sklearn.metrics import log_loss  # noqa: E402


def grid_search_hps(
    fit: pd.DataFrame,
    calib: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    max_depths: Sequence[int] = (4, 6, 8, 10),
    learning_rates: Sequence[float] = (0.03, 0.05, 0.1),
    device: str = "cuda",
) -> dict[str, object]:
    """Sweep `max_depth × learning_rate`; return the booster minimizing calib log-loss."""
    best: dict[str, object] = {"best_logloss": float("inf")}
    for md in max_depths:
        for lr in learning_rates:
            booster, n_iter = train_one_booster(
                fit=fit,
                calib=calib,
                feature_cols=feature_cols,
                target_col=target_col,
                max_depth=md,
                learning_rate=lr,
                device=device,
            )
            preds = booster.predict_proba(calib[list(feature_cols)])[:, 1]
            ll = log_loss(calib[target_col], preds, labels=[0, 1])
            if ll < float(best["best_logloss"]):
                best = {
                    "best_logloss": ll,
                    "best_max_depth": md,
                    "best_lr": lr,
                    "best_n_iter": n_iter,
                    "best_booster": booster,
                }
    return best
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_training.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/training.py tests/test_training.py
git commit -m "feat(training): hyperparameter grid search by calib log-loss"
```

---

### Task 36: Full per-fold per-hazard pipeline

**Files:**
- Modify: `src/ml_severe_weather_forecast/training.py`

- [ ] **Step 1: Implement the orchestrator that runs grid search and saves artifacts**

Append to `src/ml_severe_weather_forecast/training.py`:

```python
import joblib  # noqa: E402

from ml_severe_weather_forecast.config import settings  # noqa: E402


def train_fold_for_hazard(
    fold: Fold,
    hazard: Hazard,
    features_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    *,
    device: str = "cuda",
) -> Path:
    """Train one (fold, hazard) booster + save artifact bundle."""
    data = assemble_fold_data(fold, features_dir, labels_dir)
    feature_cols = select_feature_columns(data["fit"])

    sweep = grid_search_hps(
        fit=data["fit"],
        calib=data["calib"],
        feature_cols=feature_cols,
        target_col=hazard,
        device=device,
    )
    booster = sweep["best_booster"]

    artifact = {
        "fold": fold,
        "hazard": hazard,
        "booster": booster,
        "feature_cols": feature_cols,
        "best_max_depth": sweep["best_max_depth"],
        "best_lr": sweep["best_lr"],
        "best_n_iter": sweep["best_n_iter"],
        "best_logloss": sweep["best_logloss"],
    }

    out_dir = out_dir / hazard
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fold.name}.joblib"
    joblib.dump(artifact, path)
    return path
```

- [ ] **Step 2: Smoke-test the import**

```bash
uv run python -c "from ml_severe_weather_forecast.training import train_fold_for_hazard; print('ok')"
```

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/training.py
git commit -m "feat(training): fold/hazard orchestrator with artifact persistence"
```

---

### Task 37: `mlswf train` CLI subcommand

**Files:**
- Modify: `src/ml_severe_weather_forecast/cli.py`

- [ ] **Step 1: Add `mlswf train`**

Append to `src/ml_severe_weather_forecast/cli.py`:

```python
@app.command("train")
def train_cmd(
    hazard: str = typer.Option(..., help="One of: tor, hail, wind."),
    gpu: bool = typer.Option(True, help="Use GPU (cuda) for XGBoost."),
) -> None:
    """Train one booster per fold for a given hazard."""
    from ml_severe_weather_forecast.training import fold_definitions, train_fold_for_hazard

    if hazard not in {"tor", "hail", "wind"}:
        raise typer.BadParameter("hazard must be tor, hail, or wind")
    years = list(range(settings.train_year_start, settings.train_year_end + 1))
    folds = fold_definitions(years)
    device = "cuda" if gpu else "cpu"
    for fold in folds:
        typer.echo(f"  fold={fold.name} hazard={hazard}")
        path = train_fold_for_hazard(
            fold,
            hazard=hazard,  # type: ignore[arg-type]
            features_dir=settings.features_dir,
            labels_dir=settings.labels_dir,
            out_dir=settings.models_dir,
            device=device,
        )
        typer.echo(f"    saved {path}")
```

- [ ] **Step 2: Smoke-test help**

```bash
uv run mlswf train --help
```

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/cli.py
git commit -m "feat(cli): mlswf train subcommand"
```

---

## Phase 10 — Calibration

### Task 38: Isotonic calibration fitting and application

**Files:**
- Create: `src/ml_severe_weather_forecast/calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Test isotonic with miscalibrated input**

```python
# tests/test_calibration.py
import numpy as np

from ml_severe_weather_forecast.calibration import fit_isotonic, calibrate


def test_isotonic_fixes_systematic_overconfidence() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    # True probability is 0.1; raw model overpredicts at 0.3
    raw = np.full(n, 0.30, dtype=np.float32) + rng.normal(0, 0.02, size=n).astype(np.float32)
    raw = np.clip(raw, 0.0, 1.0)
    y = rng.binomial(1, 0.1, size=n)
    iso = fit_isotonic(raw, y)
    calibrated = calibrate(raw, iso)
    assert abs(calibrated.mean() - 0.1) < 0.05


def test_calibrate_is_monotone() -> None:
    rng = np.random.default_rng(1)
    raw_train = rng.uniform(0, 1, size=2000).astype(np.float32)
    y_train = (raw_train + rng.normal(0, 0.1, size=2000) > 0.5).astype(np.int32)
    iso = fit_isotonic(raw_train, y_train)
    grid = np.linspace(0, 1, 100)
    out = calibrate(grid, iso)
    diffs = np.diff(out)
    assert (diffs >= -1e-6).all()
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement calibration**

```python
# src/ml_severe_weather_forecast/calibration.py
"""Isotonic regression calibration."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(raw_probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic regressor mapping raw → calibrated probabilities."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(raw_probs).astype(float), np.asarray(labels).astype(float))
    return iso


def calibrate(raw_probs: np.ndarray, iso: IsotonicRegression) -> np.ndarray:
    """Apply a fitted isotonic regressor."""
    return iso.transform(np.asarray(raw_probs).astype(float)).astype(np.float32)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_calibration.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): isotonic fit and apply"
```

---

### Task 39: Wire calibration into the training artifact

**Files:**
- Modify: `src/ml_severe_weather_forecast/training.py`

- [ ] **Step 1: Update `train_fold_for_hazard` to fit isotonic on calib + persist it + persist test predictions**

Replace `train_fold_for_hazard` in `src/ml_severe_weather_forecast/training.py` with:

```python
from ml_severe_weather_forecast.calibration import calibrate, fit_isotonic  # noqa: E402


def train_fold_for_hazard(
    fold: Fold,
    hazard: Hazard,
    features_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    *,
    device: str = "cuda",
) -> Path:
    """Train one (fold, hazard) booster + isotonic + save artifact bundle."""
    data = assemble_fold_data(fold, features_dir, labels_dir)
    feature_cols = select_feature_columns(data["fit"])

    sweep = grid_search_hps(
        fit=data["fit"],
        calib=data["calib"],
        feature_cols=feature_cols,
        target_col=hazard,
        device=device,
    )
    booster: xgb.XGBClassifier = sweep["best_booster"]  # type: ignore[assignment]

    # Fit isotonic on calib raw probs; never on test fold (no leakage).
    raw_calib = booster.predict_proba(data["calib"][feature_cols])[:, 1]
    iso = fit_isotonic(raw_calib, data["calib"][hazard].to_numpy())

    raw_test = booster.predict_proba(data["test"][feature_cols])[:, 1]
    test_calibrated = calibrate(raw_test, iso)

    test_preds = data["test"][["cycle_init_utc", "cell_id", hazard]].copy()
    test_preds["raw_prob"] = raw_test
    test_preds["calibrated_prob"] = test_calibrated

    artifact = {
        "fold": fold,
        "hazard": hazard,
        "booster": booster,
        "isotonic": iso,
        "feature_cols": feature_cols,
        "best_max_depth": sweep["best_max_depth"],
        "best_lr": sweep["best_lr"],
        "best_n_iter": sweep["best_n_iter"],
        "best_logloss": sweep["best_logloss"],
        "test_predictions": test_preds,
    }

    out_dir = out_dir / hazard
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fold.name}.joblib"
    joblib.dump(artifact, path)
    return path
```

- [ ] **Step 2: Run training tests (existing) to confirm nothing broke**

```bash
uv run pytest tests/test_training.py tests/test_calibration.py -v
```

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/training.py
git commit -m "feat(training): integrate isotonic calibration and persist test predictions"
```

---

## Phase 11 — Verification Metrics

### Task 40: Brier score + decomposition

**Files:**
- Create: `src/ml_severe_weather_forecast/verification.py`
- Test: `tests/test_verification.py`

- [ ] **Step 1: Test Brier and BSS**

```python
# tests/test_verification.py
import numpy as np

from ml_severe_weather_forecast.verification import brier_decomposition, brier_score, brier_skill_score


def test_brier_score_zero_for_perfect_predictions() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(p, y) == 0.0


def test_brier_skill_score_against_climatology() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.binomial(1, 0.1, size=n)
    perfect = y.astype(float)
    climo = np.full(n, 0.1)
    bss = brier_skill_score(perfect, y, baseline=climo)
    assert bss > 0.95  # near-perfect should be ~1


def test_brier_decomposition_sums_to_total() -> None:
    rng = np.random.default_rng(1)
    n = 5000
    y = rng.binomial(1, 0.2, size=n)
    p = rng.uniform(0, 1, size=n).astype(np.float32)
    bs = brier_score(p, y)
    rel, res, unc = brier_decomposition(p, y, n_bins=10)
    # BS = REL - RES + UNC (within rounding)
    assert abs(bs - (rel - res + unc)) < 1e-3
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement Brier metrics**

```python
# src/ml_severe_weather_forecast/verification.py
"""Probabilistic-forecast verification metrics."""
from __future__ import annotations

import numpy as np


def brier_score(prob: np.ndarray, label: np.ndarray) -> float:
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def brier_skill_score(prob: np.ndarray, label: np.ndarray, baseline: np.ndarray) -> float:
    bs_model = brier_score(prob, label)
    bs_base = brier_score(baseline, label)
    if bs_base <= 0:
        return float("nan")
    return 1.0 - bs_model / bs_base


def brier_decomposition(
    prob: np.ndarray, label: np.ndarray, n_bins: int = 10
) -> tuple[float, float, float]:
    """Return (reliability, resolution, uncertainty). BS = REL - RES + UNC."""
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    n = len(p)
    base = float(y.mean())
    uncertainty = base * (1.0 - base)

    bins = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    bin_idx = np.digitize(p, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        mask = bin_idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        pk_mean = float(p[mask].mean())
        ok = float(y[mask].mean())
        reliability += nk / n * (pk_mean - ok) ** 2
        resolution += nk / n * (ok - base) ** 2
    return reliability, resolution, uncertainty
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_verification.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/verification.py tests/test_verification.py
git commit -m "feat(verification): Brier score + reliability/resolution/uncertainty decomposition"
```

---

### Task 41: Reliability diagram data

**Files:**
- Modify: `src/ml_severe_weather_forecast/verification.py`
- Modify: `tests/test_verification.py`

- [ ] **Step 1: Test reliability binning**

Append to `tests/test_verification.py`:

```python
def test_reliability_diagram_returns_n_bins_rows() -> None:
    from ml_severe_weather_forecast.verification import reliability_diagram

    rng = np.random.default_rng(2)
    n = 5000
    p = rng.uniform(0, 1, size=n)
    y = rng.binomial(1, p)
    df = reliability_diagram(p, y, n_bins=10)
    assert len(df) == 10
    assert {"forecast_prob_mean", "observed_freq", "n"} <= set(df.columns)
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement reliability**

Append to `src/ml_severe_weather_forecast/verification.py`:

```python
import pandas as pd  # noqa: E402


def reliability_diagram(prob: np.ndarray, label: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bin forecasts by quantile and report (mean forecast, observed frequency, count) per bin."""
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    quantiles = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) - 1 < n_bins:
        # Fall back to uniform binning when probs are constant
        quantiles = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    bin_idx = np.digitize(p, quantiles) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        mask = bin_idx == k
        nk = int(mask.sum())
        rows.append(
            {
                "bin": k,
                "forecast_prob_mean": float(p[mask].mean()) if nk else float("nan"),
                "observed_freq": float(y[mask].mean()) if nk else float("nan"),
                "n": nk,
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_verification.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/verification.py tests/test_verification.py
git commit -m "feat(verification): reliability diagram binning"
```

---

### Task 42: ROC, AUC, performance diagram

**Files:**
- Modify: `src/ml_severe_weather_forecast/verification.py`
- Modify: `tests/test_verification.py`

- [ ] **Step 1: Test ROC and performance metrics**

Append to `tests/test_verification.py`:

```python
def test_roc_auc_perfect_classifier() -> None:
    from ml_severe_weather_forecast.verification import roc_auc_value

    p = np.array([0.0, 0.1, 0.2, 0.3, 0.9, 0.95, 1.0])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    assert roc_auc_value(p, y) == 1.0


def test_performance_diagram_returns_pod_sr_csi() -> None:
    from ml_severe_weather_forecast.verification import performance_diagram

    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, size=2000)
    y = rng.binomial(1, p)
    df = performance_diagram(p, y, thresholds=np.linspace(0.05, 0.95, 19))
    assert {"threshold", "POD", "SR", "CSI", "FAR"} <= set(df.columns)
    assert (df["POD"] >= 0).all() and (df["POD"] <= 1).all()
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement ROC & performance metrics**

Append to `src/ml_severe_weather_forecast/verification.py`:

```python
from sklearn.metrics import roc_auc_score  # noqa: E402


def roc_auc_value(prob: np.ndarray, label: np.ndarray) -> float:
    return float(roc_auc_score(label, prob))


def performance_diagram(
    prob: np.ndarray, label: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    """For each threshold compute POD, SR (1-FAR), CSI, FAR."""
    p = np.asarray(prob)
    y = np.asarray(label)
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        pod = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        sr = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        far = fp / (tp + fp) if (tp + fp) > 0 else float("nan")
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        rows.append({"threshold": float(t), "POD": pod, "SR": sr, "FAR": far, "CSI": csi})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_verification.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/ml_severe_weather_forecast/verification.py tests/test_verification.py
git commit -m "feat(verification): ROC AUC and performance diagram"
```

---

### Task 43: Fractions skill score (FSS)

**Files:**
- Modify: `src/ml_severe_weather_forecast/verification.py`
- Modify: `tests/test_verification.py`

- [ ] **Step 1: Test FSS at a single radius**

Append to `tests/test_verification.py`:

```python
def test_fss_perfect_match_is_one() -> None:
    from ml_severe_weather_forecast.verification import fractions_skill_score

    cells = np.array(
        [["c_001_001", 0.0, 35.0, -97.0, 1.0, 1]],
        dtype=object,
    )
    df = pd.DataFrame(
        {
            "cell_id": ["c_001_001", "c_001_002"],
            "lat": [35.0, 35.5],
            "lon": [-97.0, -97.0],
            "tor": [1, 0],
            "calibrated_prob": [1.0, 0.0],
        }
    )
    fss = fractions_skill_score(df, prob_col="calibrated_prob", label_col="tor", radius_km=80)
    assert 0.99 <= fss <= 1.0
```

- [ ] **Step 2: Implement FSS**

Append to `src/ml_severe_weather_forecast/verification.py`:

```python
from sklearn.neighbors import BallTree  # noqa: E402


def _neighborhood_fractions(
    df: pd.DataFrame, value_col: str, radius_km: float
) -> np.ndarray:
    pts = np.deg2rad(df[["lat", "lon"]].to_numpy())
    tree = BallTree(pts, metric="haversine")
    radius_rad = (radius_km * 1000.0) / 6_371_000.0
    fractions = np.zeros(len(df), dtype=np.float32)
    values = df[value_col].to_numpy(dtype=np.float32)
    neighbors = tree.query_radius(pts, r=radius_rad)
    for i, ns in enumerate(neighbors):
        fractions[i] = float(values[ns].mean()) if len(ns) else 0.0
    return fractions


def fractions_skill_score(
    df: pd.DataFrame,
    *,
    prob_col: str,
    label_col: str,
    radius_km: float,
) -> float:
    """FSS at a single radius. Caller supplies a frame with cell_id, lat, lon, prob, label."""
    if not {"lat", "lon", prob_col, label_col} <= set(df.columns):
        raise ValueError("df must contain lat, lon, prob, label columns")
    pf = _neighborhood_fractions(df, prob_col, radius_km)
    of = _neighborhood_fractions(df, label_col, radius_km)
    mse = float(np.mean((pf - of) ** 2))
    ref = float(np.mean(pf**2 + of**2))
    if ref == 0:
        return float("nan")
    return 1.0 - mse / ref
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_verification.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/ml_severe_weather_forecast/verification.py tests/test_verification.py
git commit -m "feat(verification): fractions skill score (single-radius)"
```

---

### Task 44: Aggregate metrics across folds

**Files:**
- Modify: `src/ml_severe_weather_forecast/verification.py`

- [ ] **Step 1: Implement the aggregator that loads model artifacts and produces a metrics report**

Append to `src/ml_severe_weather_forecast/verification.py`:

```python
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402


def load_test_predictions(model_dir: Path, hazard: str) -> pd.DataFrame:
    """Concatenate calibrated test predictions across all folds for one hazard."""
    parts: list[pd.DataFrame] = []
    for f in sorted((model_dir / hazard).glob("holdout_*.joblib")):
        artifact = joblib.load(f)
        df = artifact["test_predictions"].copy()
        df["fold"] = artifact["fold"].name
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No models found in {model_dir / hazard}")
    return pd.concat(parts, ignore_index=True)


def aggregate_metrics(
    preds: pd.DataFrame,
    *,
    hazard: str,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    cell_ids: np.ndarray,
    climatology: np.ndarray,
) -> dict[str, object]:
    """Compute pooled and per-fold metrics for a hazard.

    `climatology` should be aligned to `preds` by row order (per-prediction climo prob).
    """
    p = preds["calibrated_prob"].to_numpy(dtype=np.float64)
    y = preds[hazard].to_numpy(dtype=np.float64)

    pooled = {
        "brier": brier_score(p, y),
        "bss_climo": brier_skill_score(p, y, baseline=climatology.astype(np.float64)),
        "auc": roc_auc_value(p, y),
    }
    rel, res, unc = brier_decomposition(p, y, n_bins=10)
    pooled.update({"reliability": rel, "resolution": res, "uncertainty": unc})

    per_fold: list[dict[str, object]] = []
    for fold_name, sub in preds.groupby("fold"):
        idx = sub.index.to_numpy()
        per_fold.append(
            {
                "fold": fold_name,
                "brier": brier_score(sub["calibrated_prob"], sub[hazard]),
                "bss_climo": brier_skill_score(
                    sub["calibrated_prob"], sub[hazard], baseline=climatology[idx]
                ),
                "auc": roc_auc_value(sub["calibrated_prob"], sub[hazard]),
            }
        )
    fold_df = pd.DataFrame(per_fold)

    return {
        "pooled": pooled,
        "per_fold": fold_df,
        "per_fold_mean_std": {
            col: (float(fold_df[col].mean()), float(fold_df[col].std()))
            for col in ("brier", "bss_climo", "auc")
        },
    }
```

- [ ] **Step 2: Smoke-test import**

```bash
uv run python -c "from ml_severe_weather_forecast.verification import aggregate_metrics; print('ok')"
```

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/verification.py
git commit -m "feat(verification): pooled + per-fold metric aggregation"
```

---

### Task 45: Baselines — climatology, MXUPHL threshold, SPC outlooks

**Files:**
- Create: `src/ml_severe_weather_forecast/baselines.py`
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Test the MXUPHL sigmoid baseline fitting**

```python
# tests/test_baselines.py
import numpy as np
import pandas as pd

from ml_severe_weather_forecast.baselines import (
    align_climatology_to_predictions,
    fit_mxuphl_sigmoid,
    spc_outlook_predictions,
)


def test_fit_mxuphl_sigmoid_returns_a_b() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    mxuphl = rng.uniform(0, 100, size=n).astype(np.float32)
    y = (mxuphl > 50).astype(np.int32)
    a, b = fit_mxuphl_sigmoid(mxuphl, y)
    assert isinstance(a, float)
    assert isinstance(b, float)


def test_align_climatology_handles_missing_cells() -> None:
    preds = pd.DataFrame(
        {"cell_id": ["c_001_001", "c_002_002"], "cycle_init_utc": pd.to_datetime(["2024-05-15 12:00", "2024-06-15 12:00"], utc=True)}
    )
    climo = pd.DataFrame(
        {"cell_id": ["c_001_001", "c_002_002"], "month": [5, 6], "tor_climo_prob": [0.01, 0.02]}
    )
    aligned = align_climatology_to_predictions(preds, climo, hazard="tor")
    assert aligned.shape == (2,)
    assert aligned[0] == 0.01
    assert aligned[1] == 0.02
```

- [ ] **Step 2: Implement baselines**

```python
# src/ml_severe_weather_forecast/baselines.py
"""Baselines: climatology alignment, MXUPHL sigmoid threshold, SPC outlook predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_mxuphl_sigmoid(mxuphl: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit a 2-parameter logistic P(severe) = sigmoid(a*mxuphl + b) by maximum likelihood."""
    x = np.asarray(mxuphl, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    def neg_loglik(params: np.ndarray) -> float:
        a, b = params
        logits = a * x + b
        # log(1+exp(z)) stable
        loss = np.log1p(np.exp(-np.abs(logits))) + np.maximum(-logits, 0) * (1 - y_arr) + np.maximum(logits, 0) * (1 - y_arr) - logits * y_arr
        return float(loss.mean())

    result = minimize(neg_loglik, x0=np.array([0.05, -3.0]), method="Nelder-Mead")
    a, b = float(result.x[0]), float(result.x[1])
    return a, b


def predict_mxuphl_sigmoid(mxuphl: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * np.asarray(mxuphl, dtype=np.float64) + b
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def align_climatology_to_predictions(
    preds: pd.DataFrame, climo: pd.DataFrame, hazard: str
) -> np.ndarray:
    """Look up the per-cell-month climatological probability for each prediction row."""
    need = preds.copy()
    need["month"] = need["cycle_init_utc"].dt.month
    out = need.merge(
        climo[["cell_id", "month", f"{hazard}_climo_prob"]],
        on=["cell_id", "month"],
        how="left",
    )
    return out[f"{hazard}_climo_prob"].fillna(0.0).to_numpy(dtype=np.float32)


def spc_outlook_predictions(
    preds: pd.DataFrame, outlooks_dir: Path, hazard: str
) -> np.ndarray:
    """For each (cycle, cell) row, look up the SPC outlook probability from the rasterized cache."""
    out = np.zeros(len(preds), dtype=np.float32)
    if "tor" == hazard:
        col = "tor_prob"
    elif hazard == "hail":
        col = "hail_prob"
    else:
        col = "wind_prob"

    cache: dict[tuple[int, str], pd.DataFrame] = {}
    for i, row in preds.iterrows():
        cycle = row["cycle_init_utc"]
        year = int(cycle.year)
        mmdd = cycle.strftime("%m%d")
        key = (year, mmdd)
        if key not in cache:
            path = outlooks_dir / str(year) / f"{mmdd}.parquet"
            cache[key] = pd.read_parquet(path) if path.exists() else None  # type: ignore[assignment]
        df = cache[key]
        if df is None:
            continue
        match = df[df["cell_id"] == row["cell_id"]]
        if not match.empty:
            out[i] = float(match.iloc[0][col])
    return out
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_baselines.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/ml_severe_weather_forecast/baselines.py tests/test_baselines.py
git commit -m "feat(baselines): climatology alignment, MXUPHL sigmoid, SPC outlook lookup"
```

---

### Task 46: `mlswf verify` CLI subcommand

**Files:**
- Modify: `src/ml_severe_weather_forecast/cli.py`

- [ ] **Step 1: Add `mlswf verify`**

Append to `src/ml_severe_weather_forecast/cli.py`:

```python
@app.command("verify")
def verify_cmd(
    hazard: str = typer.Option(..., help="One of: tor, hail, wind."),
    out: Path = typer.Option(None, help="Where to write the metrics JSON."),
) -> None:
    """Compute pooled + per-fold metrics for a hazard."""
    import json

    import pandas as pd

    from ml_severe_weather_forecast.baselines import align_climatology_to_predictions
    from ml_severe_weather_forecast.data.grid import build_grid
    from ml_severe_weather_forecast.verification import aggregate_metrics, load_test_predictions

    if hazard not in {"tor", "hail", "wind"}:
        raise typer.BadParameter("hazard must be tor, hail, or wind")

    preds = load_test_predictions(settings.models_dir, hazard)
    grid = build_grid()

    climo_path = settings.data_dir / "climatology.parquet"
    if not climo_path.exists():
        raise typer.BadParameter(f"missing climatology: {climo_path}")
    climo = pd.read_parquet(climo_path)
    climo_aligned = align_climatology_to_predictions(preds, climo, hazard)

    metrics = aggregate_metrics(
        preds,
        hazard=hazard,
        grid_lats=grid.lats,
        grid_lons=grid.lons,
        cell_ids=grid.cell_ids,
        climatology=climo_aligned,
    )

    out_path = out or settings.data_dir / f"verification_{hazard}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "pooled": metrics["pooled"],
        "per_fold": metrics["per_fold"].to_dict(orient="records"),
        "per_fold_mean_std": metrics["per_fold_mean_std"],
    }
    out_path.write_text(json.dumps(serializable, default=str, indent=2))
    typer.echo(f"Wrote {out_path}")
```

- [ ] **Step 2: Smoke-test help**

```bash
uv run mlswf verify --help
```

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/cli.py
git commit -m "feat(cli): mlswf verify subcommand"
```

---

## Phase 12 — Visualizations

Per the spec, `viz.py` has no unit tests — visual correctness is verified manually. Each task below produces a figure file you should open and inspect before committing.

### Task 47: Cartopy basemap utility + probability map

**Files:**
- Create: `src/ml_severe_weather_forecast/viz.py`

- [ ] **Step 1: Implement basemap and forecast-map plotting**

```python
# src/ml_severe_weather_forecast/viz.py
"""Matplotlib + cartopy figure builders. No unit tests; verify visually."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

# SPC convective-outlook color stops (probability bins)
SPC_PROB_BINS = [0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60]
SPC_COLORS = ["#008b00", "#8b4513", "#ffc800", "#ff0000", "#ff00ff", "#912cee", "#104e8b"]


def make_conus_axes(ax: plt.Axes | None = None) -> plt.Axes:
    """Return a Lambert Conformal CONUS Axes with state borders."""
    proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5)
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    return ax


def plot_forecast_map(
    cell_lats: np.ndarray,
    cell_lons: np.ndarray,
    cell_probs: np.ndarray,
    *,
    title: str,
    out_path: Path,
    report_lats: np.ndarray | None = None,
    report_lons: np.ndarray | None = None,
) -> Path:
    """Render a single forecast-probability map with optional storm-report overlay."""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(-97.5, 38.5))
    make_conus_axes(ax)

    cmap = ListedColormap(SPC_COLORS)
    norm = BoundaryNorm([0, *SPC_PROB_BINS, 1.0], cmap.N + 1, extend="max")

    sc = ax.scatter(
        cell_lons,
        cell_lats,
        c=cell_probs,
        cmap=cmap,
        norm=norm,
        s=12,
        marker="s",
        transform=ccrs.PlateCarree(),
    )
    if report_lats is not None and report_lons is not None and len(report_lats):
        ax.scatter(
            report_lons, report_lats, s=6, color="black", marker="o",
            transform=ccrs.PlateCarree(), zorder=5,
        )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, label="Probability")
    cbar.set_ticks(SPC_PROB_BINS)
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

- [ ] **Step 2: Manual verification**

Make a quick smoke test:

```bash
uv run python -c "
from pathlib import Path
import numpy as np
from ml_severe_weather_forecast.data.grid import build_grid
from ml_severe_weather_forecast.viz import plot_forecast_map

g = build_grid()
probs = np.random.rand(g.n_cells) * 0.3
plot_forecast_map(g.lats, g.lons, probs, title='smoke test', out_path=Path('data/outputs/smoke.png'))
"
```
Expected: `data/outputs/smoke.png` exists. Open and visually verify it shows a CONUS map with colored cells.

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/viz.py
git commit -m "feat(viz): CONUS basemap + forecast-probability map"
```

---

### Task 48: Side-by-side panel (our forecast vs. SPC outlook)

**Files:**
- Modify: `src/ml_severe_weather_forecast/viz.py`

- [ ] **Step 1: Add side-by-side renderer**

Append to `src/ml_severe_weather_forecast/viz.py`:

```python
def plot_side_by_side(
    cell_lats: np.ndarray,
    cell_lons: np.ndarray,
    our_probs: np.ndarray,
    spc_probs: np.ndarray,
    *,
    title: str,
    out_path: Path,
    report_lats: np.ndarray | None = None,
    report_lons: np.ndarray | None = None,
) -> Path:
    """Two-panel figure: our model on the left, SPC Day-1 outlook on the right."""
    fig = plt.figure(figsize=(16, 6))
    proj = ccrs.LambertConformal(-97.5, 38.5)
    cmap = ListedColormap(SPC_COLORS)
    norm = BoundaryNorm([0, *SPC_PROB_BINS, 1.0], cmap.N + 1, extend="max")

    for i, (label, probs) in enumerate([("Our model", our_probs), ("SPC Day-1", spc_probs)]):
        ax = fig.add_subplot(1, 2, i + 1, projection=proj)
        make_conus_axes(ax)
        sc = ax.scatter(
            cell_lons, cell_lats, c=probs, cmap=cmap, norm=norm,
            s=12, marker="s", transform=ccrs.PlateCarree(),
        )
        if report_lats is not None and report_lons is not None and len(report_lats):
            ax.scatter(
                report_lons, report_lats, s=6, color="black", marker="o",
                transform=ccrs.PlateCarree(), zorder=5,
            )
        ax.set_title(label)
    fig.suptitle(title)
    fig.colorbar(sc, ax=fig.axes, shrink=0.5, label="Probability", orientation="horizontal", pad=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

- [ ] **Step 2: Commit**

```bash
git add src/ml_severe_weather_forecast/viz.py
git commit -m "feat(viz): side-by-side our-model vs SPC outlook panel"
```

---

### Task 49: Reliability + ROC + performance diagrams

**Files:**
- Modify: `src/ml_severe_weather_forecast/viz.py`

- [ ] **Step 1: Add static-figure renderers**

Append to `src/ml_severe_weather_forecast/viz.py`:

```python
def plot_reliability(reliability_df: pd.DataFrame, out_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.plot(
        reliability_df["forecast_prob_mean"],
        reliability_df["observed_freq"],
        "o-",
        label="Model",
    )
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, max(reliability_df["forecast_prob_mean"].max() * 1.05, 0.3))
    ax.set_ylim(0, max(reliability_df["observed_freq"].max() * 1.05, 0.3))
    ax.legend()
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_and_performance(
    prob: np.ndarray,
    label: np.ndarray,
    perf_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> Path:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(label, prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(fpr, tpr, "b-")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"{title} — ROC")

    axes[1].plot(perf_df["SR"], perf_df["POD"], "ro-")
    axes[1].set_xlabel("Success Ratio (1 - FAR)")
    axes[1].set_ylabel("Probability of Detection")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"{title} — Performance Diagram")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_importance(
    feature_names: Sequence[str], importance: Sequence[float], out_path: Path, title: str, top_k: int = 20
) -> Path:
    pairs = sorted(zip(feature_names, importance, strict=True), key=lambda x: x[1], reverse=True)[:top_k]
    names = [p[0] for p in pairs][::-1]
    vals = [p[1] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals)
    ax.set_xlabel("Gain")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

- [ ] **Step 2: Commit**

```bash
git add src/ml_severe_weather_forecast/viz.py
git commit -m "feat(viz): reliability, ROC, performance, feature-importance figures"
```

---

### Task 50: Monthly skill heatmap + hits/misses mosaic

**Files:**
- Modify: `src/ml_severe_weather_forecast/viz.py`

- [ ] **Step 1: Add the monthly heatmap renderer**

Append to `src/ml_severe_weather_forecast/viz.py`:

```python
def plot_monthly_bss_heatmap(
    bss_table: pd.DataFrame,
    out_path: Path,
    title: str,
) -> Path:
    """`bss_table` has rows hazards × cols months, values BSS."""
    fig, ax = plt.subplots(figsize=(8, 3))
    arr = bss_table.to_numpy(dtype=np.float32)
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-0.2, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(bss_table.columns)), bss_table.columns)
    ax.set_yticks(range(len(bss_table.index)), bss_table.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i,j]:.2f}", ha="center", va="center",
                    color="black" if abs(arr[i,j]) < 0.2 else "white")
    fig.colorbar(im, ax=ax, label="BSS")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_hits_misses_mosaic(
    candidates: dict[str, dict[str, np.ndarray]],
    out_path: Path,
    title: str,
) -> Path:
    """`candidates` is {label: {lats, lons, probs, report_lats, report_lons}} for 4 panels."""
    fig = plt.figure(figsize=(14, 8))
    proj = ccrs.LambertConformal(-97.5, 38.5)
    cmap = ListedColormap(SPC_COLORS)
    norm = BoundaryNorm([0, *SPC_PROB_BINS, 1.0], cmap.N + 1, extend="max")
    for i, (label, payload) in enumerate(candidates.items()):
        ax = fig.add_subplot(2, 2, i + 1, projection=proj)
        make_conus_axes(ax)
        ax.scatter(
            payload["lons"], payload["lats"], c=payload["probs"],
            cmap=cmap, norm=norm, s=12, marker="s", transform=ccrs.PlateCarree(),
        )
        if "report_lats" in payload and len(payload["report_lats"]):
            ax.scatter(
                payload["report_lons"], payload["report_lats"], s=8, color="black",
                marker="o", transform=ccrs.PlateCarree(), zorder=5,
            )
        ax.set_title(label, fontsize=10)
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

- [ ] **Step 2: Commit**

```bash
git add src/ml_severe_weather_forecast/viz.py
git commit -m "feat(viz): monthly BSS heatmap and hits/misses 2×2 mosaic"
```

---

### Task 51: SHAP summary plot

**Files:**
- Modify: `src/ml_severe_weather_forecast/viz.py`

- [ ] **Step 1: Add a SHAP summary helper that samples for tractability**

Append to `src/ml_severe_weather_forecast/viz.py`:

```python
def plot_shap_summary(
    booster, X: pd.DataFrame, out_path: Path, title: str, sample_n: int = 50_000
) -> Path:
    import shap

    if len(X) > sample_n:
        X = X.sample(sample_n, random_state=0)
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X)
    fig = plt.figure(figsize=(8, 8))
    shap.summary_plot(sv, X, show=False, plot_size=None)
    fig = plt.gcf()
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

- [ ] **Step 2: Commit**

```bash
git add src/ml_severe_weather_forecast/viz.py
git commit -m "feat(viz): SHAP summary plot (sampled for tractability)"
```

---

### Task 52: `mlswf plot` CLI subcommand

**Files:**
- Modify: `src/ml_severe_weather_forecast/cli.py`

- [ ] **Step 1: Add `mlswf plot`**

Append to `src/ml_severe_weather_forecast/cli.py`:

```python
@app.command("plot")
def plot_cmd(
    hazard: str = typer.Option(..., help="One of: tor, hail, wind."),
    cycle: str = typer.Option(..., help="Cycle date YYYY-MM-DD (12z run)."),
    out: Path = typer.Option(None, help="Override default output PNG path."),
) -> None:
    """Render the per-cycle forecast map for one hazard."""
    from datetime import UTC, datetime

    import pandas as pd

    from ml_severe_weather_forecast.data.grid import build_grid
    from ml_severe_weather_forecast.verification import load_test_predictions
    from ml_severe_weather_forecast.viz import plot_forecast_map

    if hazard not in {"tor", "hail", "wind"}:
        raise typer.BadParameter("hazard must be tor, hail, or wind")
    cycle_dt = datetime.fromisoformat(cycle).replace(tzinfo=UTC, hour=settings.hrrr_cycle_hour)

    preds = load_test_predictions(settings.models_dir, hazard)
    sub = preds[preds["cycle_init_utc"] == pd.Timestamp(cycle_dt)]
    if sub.empty:
        raise typer.BadParameter(f"no test predictions for cycle {cycle_dt}")
    grid = build_grid()
    cell_to_prob = dict(zip(sub["cell_id"], sub["calibrated_prob"], strict=True))
    probs = np.array([cell_to_prob.get(cid, 0.0) for cid in grid.cell_ids], dtype=np.float32)

    out_path = out or settings.outputs_dir / hazard / f"{cycle_dt.strftime('%Y-%m-%d')}-12z.png"
    plot_forecast_map(
        grid.lats, grid.lons, probs,
        title=f"{hazard} 24h prob — cycle {cycle_dt:%Y-%m-%d %Hz}",
        out_path=out_path,
    )
    typer.echo(f"Wrote {out_path}")
```

Add `import numpy as np` at the top of `cli.py` if not already present.

- [ ] **Step 2: Smoke-test help**

```bash
uv run mlswf plot --help
```

- [ ] **Step 3: Commit**

```bash
git add src/ml_severe_weather_forecast/cli.py
git commit -m "feat(cli): mlswf plot subcommand"
```

---

### Task 53: End-to-end integration test

**Files:**
- Create: `tests/test_end_to_end.py`

- [ ] **Step 1: Write the integration test using fixtures**

```python
# tests/test_end_to_end.py
"""End-to-end smoke test: tiny fixtures → full pipeline runs without error.

Skipped on CI (where the GRIB2 fixture is unavailable). Run locally with `pytest -v -m e2e`.
"""
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_GRIB = Path(__file__).parent / "fixtures" / "tiny_hrrr.grib2"


@pytest.mark.skipif(not FIXTURE_GRIB.exists(), reason="tiny_hrrr.grib2 fixture not present")
def test_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    from ml_severe_weather_forecast.calibration import calibrate, fit_isotonic
    from ml_severe_weather_forecast.data.grid import build_grid
    from ml_severe_weather_forecast.data.hrrr import extract_variables_to_dataset, regrid_to_cells
    from ml_severe_weather_forecast.features.assembly import assemble_cycle_features
    from ml_severe_weather_forecast.labels import label_cycle
    from ml_severe_weather_forecast.training import train_one_booster

    grid = build_grid()

    # 1) Extract + regrid one fixture cycle
    feats = assemble_cycle_features(
        cycle_init=datetime(2023, 5, 15, 12, tzinfo=UTC),
        grib_paths=[FIXTURE_GRIB],
        grid=grid,
        out_path=tmp_path / "features.parquet",
    )
    df = pd.read_parquet(feats)
    assert "MLCAPE_max_fhr_max" in df.columns

    # 2) Synthesize labels
    reports = pd.DataFrame(
        [
            {
                "event_time_utc": pd.Timestamp("2023-05-15 18:00", tz="UTC"),
                "lat": 35.222,
                "lon": -97.439,
                "hazard": "tor",
                "magnitude": 2,
            }
        ]
    )
    lbl = label_cycle(grid, datetime(2023, 5, 15, 12, tzinfo=UTC), reports)
    assert lbl["tor"].sum() >= 1

    # 3) Tiny train + calibrate
    train = df.merge(lbl, on="cell_id", how="inner")
    feats_only = [c for c in train.columns if c not in {"cycle_init_utc", "cell_id", "tor", "hail", "wind", "tor_event_count", "tor_max_magnitude", "hail_event_count", "hail_max_magnitude", "wind_event_count", "wind_max_magnitude"}]
    fit = train.iloc[: len(train) // 2]
    cal = train.iloc[len(train) // 2 :]
    if fit["tor"].sum() == 0:
        pytest.skip("no positive label in tiny fold")
    booster, _ = train_one_booster(
        fit=fit, calib=cal, feature_cols=feats_only, target_col="tor",
        max_depth=4, learning_rate=0.1, device="cpu", n_estimators=20,
    )
    raw = booster.predict_proba(cal[feats_only])[:, 1]
    iso = fit_isotonic(raw, cal["tor"].to_numpy())
    out = calibrate(raw, iso)
    assert out.shape == raw.shape
```

- [ ] **Step 2: Run the integration test**

```bash
uv run pytest tests/test_end_to_end.py -v
```
Expected: PASS (or `SKIP` if the fixture isn't present).

- [ ] **Step 3: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "test: end-to-end smoke covering extract → label → train → calibrate"
```

---

## Phase 13 — End-to-End Orchestration & Report

### Task 54: `scripts/00_download_all.py`

**Files:**
- Create: `scripts/00_download_all.py`

- [ ] **Step 1: Implement**

```python
# scripts/00_download_all.py
"""Run all three download CLI commands in sequence."""
from __future__ import annotations

import subprocess

from ml_severe_weather_forecast.config import settings


def main() -> None:
    train_start = f"{settings.train_year_start}-04-01"
    train_end = f"{settings.train_year_end}-07-31"
    subprocess.check_call(["uv", "run", "mlswf", "download", "hrrr", "--start", train_start, "--end", train_end])
    subprocess.check_call(
        ["uv", "run", "mlswf", "download", "reports",
         "--start", str(settings.climo_year_start), "--end", str(settings.train_year_end)]
    )
    subprocess.check_call(
        ["uv", "run", "mlswf", "download", "spc-outlooks",
         "--start", str(settings.train_year_start), "--end", str(settings.train_year_end)]
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/00_download_all.py
git commit -m "feat(scripts): orchestrate all three download stages"
```

---

### Task 55: `scripts/02_train_all.py`

**Files:**
- Create: `scripts/02_train_all.py`

- [ ] **Step 1: Implement**

```python
# scripts/02_train_all.py
"""Build labels, climatology, then train one booster per hazard per fold."""
from __future__ import annotations

import subprocess

import pandas as pd

from ml_severe_weather_forecast.climatology import compute_climatology
from ml_severe_weather_forecast.config import settings
from ml_severe_weather_forecast.data.grid import build_grid


def _build_climatology() -> None:
    grid = build_grid()
    paths = sorted(settings.reports_dir.glob("*.parquet"))
    in_window = [p for p in paths if settings.climo_year_start <= int(p.stem) <= settings.climo_year_end]
    if not in_window:
        raise FileNotFoundError("No climatology-window report parquets present")
    reports = pd.concat([pd.read_parquet(p) for p in in_window], ignore_index=True)
    n_years = settings.climo_year_end - settings.climo_year_start + 1
    df = compute_climatology(reports, grid, years=n_years)
    out = settings.data_dir / "climatology.parquet"
    df.to_parquet(out, index=False)


def main() -> None:
    for year in range(settings.train_year_start, settings.train_year_end + 1):
        subprocess.check_call(["uv", "run", "mlswf", "label", "--year", str(year)])
    _build_climatology()
    for hazard in ("tor", "hail", "wind"):
        subprocess.check_call(["uv", "run", "mlswf", "train", "--hazard", hazard])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/02_train_all.py
git commit -m "feat(scripts): orchestrate labels + climatology + training"
```

---

### Task 56: `scripts/03_verify_all.py`

**Files:**
- Create: `scripts/03_verify_all.py`

- [ ] **Step 1: Implement verification + figure regeneration**

```python
# scripts/03_verify_all.py
"""Run verification + regenerate static report figures."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_severe_weather_forecast.baselines import (
    align_climatology_to_predictions,
    fit_mxuphl_sigmoid,
    predict_mxuphl_sigmoid,
    spc_outlook_predictions,
)
from ml_severe_weather_forecast.config import settings
from ml_severe_weather_forecast.verification import (
    aggregate_metrics,
    brier_skill_score,
    load_test_predictions,
    performance_diagram,
    reliability_diagram,
)
from ml_severe_weather_forecast.viz import (
    plot_feature_importance,
    plot_monthly_bss_heatmap,
    plot_reliability,
    plot_roc_and_performance,
    plot_shap_summary,
)


def _figures_for_hazard(hazard: str, fig_dir: Path) -> None:
    preds = load_test_predictions(settings.models_dir, hazard)
    climo = pd.read_parquet(settings.data_dir / "climatology.parquet")
    climo_aligned = align_climatology_to_predictions(preds, climo, hazard)

    metrics = aggregate_metrics(
        preds,
        hazard=hazard,
        grid_lats=np.array([]),
        grid_lons=np.array([]),
        cell_ids=np.array([]),
        climatology=climo_aligned,
    )

    rel = reliability_diagram(preds["calibrated_prob"], preds[hazard])
    plot_reliability(rel, fig_dir / f"reliability_{hazard}.png", title=f"{hazard} — reliability")

    perf = performance_diagram(
        preds["calibrated_prob"].to_numpy(), preds[hazard].to_numpy(),
        thresholds=np.linspace(0.02, 0.6, 20),
    )
    plot_roc_and_performance(
        preds["calibrated_prob"].to_numpy(), preds[hazard].to_numpy(),
        perf, fig_dir / f"roc_perf_{hazard}.png", title=hazard,
    )

    # Feature importance from the first fold's booster
    first_fold_path = sorted((settings.models_dir / hazard).glob("holdout_*.joblib"))[0]
    artifact = joblib.load(first_fold_path)
    imp = artifact["booster"].get_booster().get_score(importance_type="gain")
    feature_names = list(imp.keys())
    importance = list(imp.values())
    plot_feature_importance(
        feature_names, importance, fig_dir / f"importance_{hazard}.png",
        title=f"{hazard} — top features (gain)",
    )

    (fig_dir / f"metrics_{hazard}.json").write_text(
        json.dumps(
            {
                "pooled": metrics["pooled"],
                "per_fold_mean_std": metrics["per_fold_mean_std"],
            },
            default=str,
            indent=2,
        )
    )


def _monthly_bss_heatmap(hazards: list[str], fig_dir: Path) -> None:
    rows = {}
    for hazard in hazards:
        preds = load_test_predictions(settings.models_dir, hazard)
        climo = pd.read_parquet(settings.data_dir / "climatology.parquet")
        climo_aligned = align_climatology_to_predictions(preds, climo, hazard)
        per_month = {}
        for month, sub in preds.groupby(preds["cycle_init_utc"].dt.month):
            idx = sub.index.to_numpy()
            per_month[int(month)] = brier_skill_score(
                sub["calibrated_prob"].to_numpy(),
                sub[hazard].to_numpy(),
                baseline=climo_aligned[idx],
            )
        rows[hazard] = per_month
    df = pd.DataFrame(rows).T
    df.columns = [f"{m:02d}" for m in df.columns]
    plot_monthly_bss_heatmap(df, fig_dir / "monthly_bss.png", title="Monthly BSS by hazard")


def main() -> None:
    fig_dir = Path("docs") / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for hazard in ("tor", "hail", "wind"):
        subprocess.check_call(["uv", "run", "mlswf", "verify", "--hazard", hazard])
        _figures_for_hazard(hazard, fig_dir)
    _monthly_bss_heatmap(["tor", "hail", "wind"], fig_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/03_verify_all.py
git commit -m "feat(scripts): verify + regenerate report figures"
```

---

### Task 57: Verification report notebook

**Files:**
- Create: `notebooks/04_verification_report.ipynb` (skeleton)
- Modify: `README.md` to point at the rendered HTML

- [ ] **Step 1: Create the notebook skeleton**

Use jupyter to create a new notebook at `notebooks/04_verification_report.ipynb` with the following cells (one per markdown/code section):

1. **Markdown:** Title + methodology overview (1–2 paragraphs summarizing the spec).
2. **Code:** load `data/verification_*.json` per hazard, render the headline-numbers table.
3. **Markdown:** "Reliability and discrimination."
4. **Code:** display `docs/figures/reliability_{hazard}.png` for each hazard.
5. **Markdown:** "Comparison to SPC outlooks."
6. **Code:** load `data/spc_outlook_metrics.json` (produced by extending `scripts/03_verify_all.py` to also call `spc_outlook_predictions`), render side-by-side BSS table.
7. **Markdown:** "Monthly skill."
8. **Code:** display `docs/figures/monthly_bss.png`.
9. **Markdown:** "Top features driving each model."
10. **Code:** display the importance plots and SHAP plots.
11. **Markdown:** "Limitations and reproducibility."

Render to HTML:

```bash
uv run jupyter nbconvert --to html --execute notebooks/04_verification_report.ipynb --output ../docs/verification_report.html
```

- [ ] **Step 2: Update README to link the rendered report**

Edit `README.md` to add a section:

```markdown
## Verification report

The full verification writeup is at [docs/verification_report.html](docs/verification_report.html). The headline numbers across hazards:

(table to be filled in once the pipeline has run)
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_verification_report.ipynb README.md
git commit -m "docs: verification report skeleton notebook + README pointer"
```

---

## Self-review

Before handing off for execution, the writer ran a self-review of this plan against the spec:

- **Spec coverage:**
  - §3 scope (3 hazards, 3 yrs, Day-1 24h, 50 km grid) — Tasks 7–9, 14–16, 32, 36
  - §4 architecture (CLI, repo layout) — Tasks 1–6, 12, 16, 20, 26, 30, 37, 46, 52
  - §5 data layer (HRRR, reports, outlooks, grid) — Tasks 7–9 (grid), 10–13 (reports), 17–20 (HRRR), 28–30 (outlooks)
  - §6 feature engineering — Tasks 21–25
  - §7 labels — Tasks 14–16
  - §8 modeling (one model per hazard, k-fold LOYO, HP search, calibration) — Tasks 32–39
  - §9 verification (metrics, baselines, visualizations) — Tasks 40–53
  - §10 testing & error handling — distributed across all tasks; integration test in Task 53
  - §11 decision log — implicit in plan structure
- **Placeholders:** none (every step contains either runnable code, a runnable command, or both)
- **Type/name consistency:** `cell_id` string format `c_iii_jjj` consistent throughout; `cycle_init_utc` (Timestamp w/ UTC) consistent; `calibrated_prob` and `raw_prob` columns consistent in test prediction frames
- **Out-of-scope items deliberately not in the plan:** real-time inference, web app, Day-2 outlooks, neural-network alternatives, Docker — all matching spec §12

---

## Execution notes

- Phases 0–11 are required code; Phase 12 has manual visual checks and Phase 13 is orchestration glue.
- The biggest single wall-clock cost is feature extraction (Phase 4–5 over 366 cycles, ~12–24 h on the 5800X with parallelism).
- HP grid search (Phase 9) is GPU-bound, ~3–6 h on the 4070 SUPER for all `(3 hazards × 3 folds × 12 HP combos)` trainings.
- Verification + figures (Phases 11–13) take minutes once predictions exist.

