from uuid import UUID

import psycopg
import pytest

from recall.config import get_settings
from recall.db import connect
from recall.migrate import apply_migrations


@pytest.fixture(scope="session")
def db_url() -> str:
    url = get_settings().test_database_url
    try:
        with psycopg.connect(url, connect_timeout=3):
            pass
    except psycopg.OperationalError as e:
        if "does not exist" in str(e):
            msg = "recall_test database missing — run `make down -v && make up` to re-run initdb"
        else:
            msg = "Postgres not reachable on 5433 — run `make up`"
        pytest.exit(msg, returncode=2)
    with connect(url) as c:
        c.execute("DROP SCHEMA public CASCADE")
        c.execute("CREATE SCHEMA public")
        apply_migrations(c)
    return url


@pytest.fixture
def conn(db_url):
    c = connect(db_url)
    yield c
    try:
        c.execute("TRUNCATE documents CASCADE")
    except psycopg.errors.UndefinedTable:
        pass  # migrations (Task 2) haven't created the schema yet
    c.close()


@pytest.fixture
def user_a() -> UUID:
    return UUID("00000000-0000-0000-0000-0000000000a1")


@pytest.fixture
def user_b() -> UUID:
    return UUID("00000000-0000-0000-0000-0000000000b1")
