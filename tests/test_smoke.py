"""Smoke test — package imports and CLI help work."""

from typer.testing import CliRunner

from ml_severe_weather_forecast.cli import app


def test_cli_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "mlswf" in result.stdout.lower()


def test_download_hrrr_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["download", "hrrr", "--help"])
    assert result.exit_code == 0
    assert "--start" in result.stdout
    assert "--end" in result.stdout


def test_download_hrrr_rejects_bad_date() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["download", "hrrr", "--start", "April 1", "--end", "2024-04-30"])
    assert result.exit_code != 0
    assert (
        "YYYY-MM-DD" in result.stdout
        or "YYYY-MM-DD" in result.stderr
        or "YYYY-MM-DD" in str(result.exception)
    )


def test_download_hrrr_rejects_reversed_range() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["download", "hrrr", "--start", "2024-04-30", "--end", "2024-04-01"]
    )
    assert result.exit_code != 0
