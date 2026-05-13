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
    dest: Path | None = typer.Option(None, help="Override the default reports dir."),  # noqa: B008
    force: bool = typer.Option(False, help="Re-download even if cached."),
) -> None:
    """Download SPC severe-weather DB CSVs for tornado/hail/wind."""
    if start > end:
        raise typer.BadParameter(f"--start ({start}) must be <= --end ({end})")
    from ml_severe_weather_forecast.data.reports import (
        HAZARD_VALID,
        build_reports,
        download_spc_year,
    )

    target_dir = dest or settings.reports_dir
    typer.echo(f"Downloading SPC reports {start}-{end} -> {target_dir}")
    for year in range(start, end + 1):
        for hazard in HAZARD_VALID:
            path = download_spc_year(year, hazard, target_dir, force=force)
            typer.echo(f"  {year} {hazard}: {path}")
    typer.echo("Building per-year Parquet artifacts...")
    parquets = build_reports(list(range(start, end + 1)), target_dir, target_dir)
    for p in parquets:
        typer.echo(f"  {p}")


@app.command("label")
def label_cmd(
    year: int = typer.Option(..., help="Year to label."),
) -> None:
    """Generate per-cycle labels for a given year using cached reports."""
    reports_path = settings.reports_dir / f"{year}.parquet"
    if not reports_path.exists():
        raise typer.BadParameter(
            f"missing reports parquet: {reports_path}. Run `mlswf download reports` first."
        )

    from datetime import UTC, datetime, timedelta

    from ml_severe_weather_forecast.data.grid import build_grid
    from ml_severe_weather_forecast.labels import build_year_labels

    grid = build_grid()
    season_start = datetime(year, settings.season_month_start, 1, tzinfo=UTC)
    # Exclusive upper bound — first moment of the month after season_month_end:
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


@download_app.command("hrrr")
def download_hrrr_cmd(
    start: str = typer.Option(..., help="First cycle date YYYY-MM-DD."),
    end: str = typer.Option(..., help="Last cycle date YYYY-MM-DD (inclusive)."),
) -> None:
    """Download 12z HRRR cycles between start and end (inclusive)."""
    from datetime import UTC, datetime, timedelta

    from ml_severe_weather_forecast.data.hrrr import download_cycle

    try:
        s = datetime.fromisoformat(start).replace(tzinfo=UTC, hour=settings.hrrr_cycle_hour)
        e = datetime.fromisoformat(end).replace(tzinfo=UTC, hour=settings.hrrr_cycle_hour)
    except ValueError as exc:
        raise typer.BadParameter(f"Dates must be YYYY-MM-DD: {exc}") from exc

    if s > e:
        raise typer.BadParameter(f"--start ({start}) must be <= --end ({end})")

    typer.echo(
        f"Downloading HRRR 12z cycles {start}..{end} "
        f"(months {settings.season_month_start}-{settings.season_month_end}) "
        f"-> {settings.hrrr_dir}"
    )
    cycles_downloaded = 0
    total_files = 0
    cur = s
    while cur <= e:
        if settings.season_month_start <= cur.month <= settings.season_month_end:
            paths = download_cycle(cur, settings.hrrr_forecast_hours, cache_dir=settings.hrrr_dir)
            typer.echo(f"  {cur.date()}: {len(paths)} files")
            cycles_downloaded += 1
            total_files += len(paths)
        cur = cur + timedelta(days=1)
    typer.echo(f"Done: {cycles_downloaded} cycles, {total_files} files")


if __name__ == "__main__":
    app()
