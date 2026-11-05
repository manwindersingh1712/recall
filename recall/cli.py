from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer

from recall.config import get_settings
from recall.db import connect
from recall.ingest.embed import get_embedder
from recall.ingest.loader import discover
from recall.ingest.pipeline import ingest_paths
from recall.migrate import apply_migrations
from recall.models import Mode
from recall.retrieval.search import search as run_search

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


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, help="A file or a folder of documents.")],
    user: Annotated[
        UUID | None, typer.Option("--user", help="User id (default: dev user).")
    ] = None,
    fake_embeddings: Annotated[
        bool, typer.Option("--fake-embeddings", help="Use hash embeddings.")
    ] = False,
) -> None:
    """Ingest a file or folder into memory."""
    settings = get_settings()
    user_id = user or settings.default_user_id
    root = path if path.is_dir() else None
    paths = discover(path) if path.is_dir() else [path]
    embedder = get_embedder(settings, fake=fake_embeddings)
    with connect() as conn:
        report = ingest_paths(conn, paths, user_id=user_id, embedder=embedder, root=root)
    typer.echo(f"{'inserted':<12}{report.inserted}")
    typer.echo(f"{'duplicates':<12}{report.duplicates}")
    typer.echo(f"{'empty':<12}{report.empty}")
    typer.echo(f"{'chunks':<12}{report.chunks}")
    typer.echo(f"{'failed':<12}{len(report.failed)}")
    for f in report.failed:
        typer.echo(f"  {Path(f.path).name}: {f.error}")
    if report.failed:
        raise typer.Exit(1)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    user: Annotated[
        UUID | None, typer.Option("--user", help="User id (default: dev user).")
    ] = None,
    mode: Annotated[Mode, typer.Option("--mode", help="vector, keyword or hybrid.")] = "hybrid",
    k: Annotated[int, typer.Option("-k", help="Number of results.")] = 10,
    fake_embeddings: Annotated[
        bool, typer.Option("--fake-embeddings", help="Use hash embeddings.")
    ] = False,
) -> None:
    """Search memory."""
    settings = get_settings()
    user_id = user or settings.default_user_id
    embedder = get_embedder(settings, fake=fake_embeddings)
    with connect() as conn:
        hits = run_search(conn, user_id=user_id, query=query, embedder=embedder, mode=mode, k=k)
    for rank, h in enumerate(hits, 1):
        snippet = " ".join(h.text.split())[:120]
        typer.echo(f"{rank:>2}  {h.score:.4f}  {h.title or '-'}  {snippet}")
