import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from recall.config import get_settings


def _configure(conn: psycopg.Connection) -> None:
    """Register pgvector types, but only once the extension exists.

    register_vector looks up the `vector` type and fails on a fresh database,
    so on an unmigrated DB we skip it (migrations create the extension).
    """
    exists = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    if exists:
        register_vector(conn)


def connect(url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(
        url or get_settings().database_url, autocommit=True, row_factory=dict_row
    )
    _configure(conn)
    return conn


def make_pool(url: str) -> ConnectionPool:
    return ConnectionPool(
        url,
        kwargs={"autocommit": True, "row_factory": dict_row},
        configure=_configure,
        open=True,
    )
