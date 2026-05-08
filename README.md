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

> **Not yet implemented.** The commands below describe the eventual end-to-end pipeline. They become available as the implementation plan progresses (see `docs/superpowers/plans/2026-05-07-ml-severe-weather-forecast.md`).

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

If `uv sync` fails on `cfgrib` or `eccodes` (most likely on platforms without pre-built wheels), install those two packages from conda-forge and continue using `uv` for everything else:

```bash
conda install -c conda-forge cfgrib eccodes
uv sync
```

Then drop `cfgrib` and `eccodes` from `pyproject.toml`'s `dependencies` list so `uv` doesn't try to reinstall them.
