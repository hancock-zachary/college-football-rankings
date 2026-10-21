from typer.testing import CliRunner

from cfb_bias.cli import app


def test_cli_help_lists_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("init-db", "pull", "ratings", "coverage", "enrollment"):
        assert cmd in result.output


def test_enrollment_without_key_exits_1(monkeypatch):
    from dataclasses import replace

    from cfb_bias import cli

    original = cli.load_settings
    monkeypatch.setattr(cli, "load_settings", lambda: replace(original(), scorecard_api_key=None))
    result = CliRunner().invoke(app, ["enrollment"])
    assert result.exit_code == 1
    assert "SCORECARD_API_KEY" in result.output


def test_institutions_csv_resolved_from_repo_not_cwd(tmp_path, monkeypatch):
    from pathlib import Path

    from cfb_bias import cli

    monkeypatch.chdir(tmp_path)
    assert cli.INSTITUTIONS_CSV.is_absolute()
    assert cli.INSTITUTIONS_CSV.is_file()
    repo = Path(__file__).resolve().parents[1]
    assert cli.INSTITUTIONS_CSV == repo / "data" / "reference" / "team_institutions.csv"
