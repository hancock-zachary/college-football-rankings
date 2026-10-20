import typer

from cfb_bias.config import load_settings
from cfb_bias.db import get_engine, init_schema

app = typer.Typer(help="College football poll-bias analysis.", no_args_is_help=True)


@app.command("init-db")
def init_db() -> None:
    """Create the database schema."""
    settings = load_settings()
    init_schema(get_engine(settings.database_url))
    typer.echo("Database schema initialized.")


@app.command()
def pull() -> None:
    """Pull data from the CFBD API into Postgres."""
    raise typer.Exit(code=_not_implemented("pull"))


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
