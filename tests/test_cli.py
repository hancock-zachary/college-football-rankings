from typer.testing import CliRunner

from cfb_bias.cli import app


def test_cli_help_lists_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("init-db", "pull", "ratings", "coverage"):
        assert cmd in result.output
