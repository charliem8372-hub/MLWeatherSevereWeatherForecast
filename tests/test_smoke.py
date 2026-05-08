"""Smoke test — package imports and CLI help work."""

from typer.testing import CliRunner

from ml_severe_weather_forecast.cli import app


def test_cli_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "mlswf" in result.stdout.lower()
