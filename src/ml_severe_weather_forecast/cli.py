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
