import typer

from cfb_bias.cfbd import CfbdClient, CfbdError
from cfb_bias.config import load_settings
from cfb_bias.db import get_engine, init_schema
from cfb_bias.ingest.games import pull_games

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
) -> None:
    """Pull data from the CFBD API into Postgres."""
    settings = load_settings()
    client = CfbdClient(settings.cfbd_api_key, settings.cache_dir)
    try:
        n = pull_games(client, get_engine(settings.database_url), season, refresh=refresh)
    except CfbdError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Season {season}: wrote {n} rows (games, team_seasons).")


@app.command()
def ratings() -> None:
    """Compute poll-free team ratings."""
    raise typer.Exit(code=_not_implemented("ratings"))


@app.command()
def coverage() -> None:
    """Report data coverage."""
    raise typer.Exit(code=_not_implemented("coverage"))


def _not_implemented(name: str) -> int:
    typer.echo(f"{name}: not implemented yet", err=True)
    return 1
