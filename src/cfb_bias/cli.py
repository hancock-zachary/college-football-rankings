from pathlib import Path

import pandas as pd
import typer

from cfb_bias.analysis.brand import build_team_brand
from cfb_bias.cfbd import CfbdClient, CfbdError
from cfb_bias.config import load_settings
from cfb_bias.coverage import coverage_table, write_coverage_md
from cfb_bias.db import get_engine, init_schema, read_sql, upsert
from cfb_bias.ingest.runner import ALL_SOURCES, parse_only, pull_all
from cfb_bias.ratings.inputs import POOLED_NON_FBS
from cfb_bias.ratings.weekly import weekly_ratings

app = typer.Typer(help="College football poll-bias analysis.", no_args_is_help=True)


@app.command("init-db")
def init_db() -> None:
    """Create the database schema."""
    settings = load_settings()
    init_schema(get_engine(settings.database_url))
    typer.echo("Database schema initialized.")


@app.command()
def pull(
    season: int = typer.Option(..., "--season", help="Season year to pull."),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass the on-disk cache."),
    only: str | None = typer.Option(
        None, "--only", help=f"Comma-separated subset of: {','.join(ALL_SOURCES)} (default all)."
    ),
) -> None:
    """Pull data from the CFBD API into Postgres."""
    try:
        sources = parse_only(only)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)
    settings = load_settings()
    client = CfbdClient(settings.cfbd_api_key, settings.cache_dir)
    try:
        written = pull_all(
            client, get_engine(settings.database_url), season, sources, refresh=refresh
        )
    except CfbdError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)
    detail = ", ".join(f"{name}={n}" for name, n in written.items())
    typer.echo(f"Season {season}: wrote {sum(written.values())} rows ({detail}).")


@app.command()
def ratings(
    season: int = typer.Option(..., "--season", help="Season year to rate."),
    bootstrap: int = typer.Option(500, "--bootstrap", help="Bootstrap resamples per rating."),
) -> None:
    """Compute poll-free weekly team ratings into ratings_weekly."""
    engine = get_engine(load_settings().database_url)
    params = {"season": season}
    games = read_sql(engine, "SELECT * FROM games WHERE season = :season", params)
    completed = games[games["completed"].astype("boolean").fillna(False).astype(bool)]
    if completed.empty:
        typer.echo(f"error: no completed games for season {season}", err=True)
        raise typer.Exit(code=1)
    team_seasons = read_sql(
        engine,
        "SELECT team FROM team_seasons WHERE season = :season AND classification = 'fbs'",
        params,
    )
    fbs_in_games = pd.concat([
        games.loc[games["home_classification"] == "fbs", "home_team"],
        games.loc[games["away_classification"] == "fbs", "away_team"],
    ])
    teams = sorted(set(team_seasons["team"]) | set(fbs_in_games.dropna()) | {POOLED_NON_FBS})
    out = weekly_ratings(games, teams, season, n_boot=bootstrap)
    written = upsert(engine, "ratings_weekly", out, ["season", "week", "team", "method"])
    typer.echo(f"Season {season}: wrote {written} ratings_weekly rows.")


@app.command()
def coverage(
    from_season: int = typer.Option(2000, "--from", help="First season."),
    to_season: int = typer.Option(2025, "--to", help="Last season (inclusive)."),
    out: Path = typer.Option(Path("docs/data-coverage.md"), "--out", help="Markdown output."),
) -> None:
    """Report per-season data coverage; print it and write markdown."""
    engine = get_engine(load_settings().database_url)
    df = coverage_table(engine, range(from_season, to_season + 1))
    write_coverage_md(df, out)
    typer.echo(df.to_string(index=False))
    typer.echo(f"Wrote {out}")


@app.command()
def brand(
    from_season: int = typer.Option(2014, "--from", help="First season."),
    to_season: int = typer.Option(2025, "--to", help="Last season (inclusive)."),
) -> None:
    """Build team brand variables into team_brand."""
    engine = get_engine(load_settings().database_url)
    for season in range(from_season, to_season + 1):
        df = build_team_brand(engine, season)
        written = upsert(engine, "team_brand", df, ["season", "team"])
        typer.echo(f"Season {season}: wrote {written} team_brand rows.")
