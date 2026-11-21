"""CLI smoke tests for Poseidon-CLI."""

from typer.testing import CliRunner

from poseidon.cli.app import app

runner = CliRunner()


def test_cli_help_displays_branding() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Poseidon-CLI" in result.output
    assert "chat" in result.output
    assert "run" in result.output


def test_cli_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Poseidon-CLI" in result.output


def test_cli_models_command() -> None:
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "nvidia/nemotron-3.5-super-vl:free" in result.output
