def test_pgvector_extension_available(conn):
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    version = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[
        "extversion"
    ]
    major, minor, *_ = (int(p) for p in version.split("."))
    assert (major, minor) >= (0, 8)


def test_server_is_postgres_16(conn):
    assert conn.execute("SHOW server_version_num").fetchone()["server_version_num"].startswith("16")
