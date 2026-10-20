import typer

from cfb_bias.cfbd import CfbdClient, CfbdError
from cfb_bias.config import load_settings
from cfb_bias.db import get_engine, init_schema
from cfb_bias.ingest.runner import ALL_SOURCES, parse_only, pull_all

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
