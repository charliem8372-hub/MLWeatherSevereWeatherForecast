"""Typer CLI entry point. Subcommands are added in later tasks."""
from __future__ import annotations

import typer

app = typer.Typer(
    name="nadocast",
    help="Nadocast-style severe-weather ML forecaster.",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """nadocast — historical-evaluation pipeline."""


if __name__ == "__main__":
    app()
