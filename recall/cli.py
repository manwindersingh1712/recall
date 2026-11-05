import typer

from recall.db import connect
from recall.migrate import apply_migrations

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Recall: hybrid-search memory engine."""


@app.command()
def migrate() -> None:
    """Apply pending SQL migrations."""
    with connect() as conn:
        applied = apply_migrations(conn)
    typer.echo("\n".join(applied) if applied else "up to date")
