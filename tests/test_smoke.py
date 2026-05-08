"""Smoke test — package imports and CLI help work."""

from typer.testing import CliRunner

from nadocast_clone.cli import app


def test_cli_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "nadocast" in result.stdout.lower()
