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
    from ml_severe_weather_forecast.data.reports import HAZARD_VALID, download_spc_year

    target_dir = dest or settings.reports_dir
    typer.echo(f"Downloading SPC reports {start}-{end} -> {target_dir}")
    for year in range(start, end + 1):
        for hazard in HAZARD_VALID:
            path = download_spc_year(year, hazard, target_dir, force=force)
            typer.echo(f"  {year} {hazard}: {path}")


if __name__ == "__main__":
    app()
