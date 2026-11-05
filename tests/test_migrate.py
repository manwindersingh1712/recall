import psycopg
import pytest

from recall.migrate import apply_migrations


def test_migrations_create_tables(conn):
    names = {
        r["table_name"]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        )
    }
    assert {"documents", "chunks", "schema_migrations"} <= names


def test_apply_migrations_is_idempotent(conn):
    assert apply_migrations(conn) == []


def test_chunks_embedding_is_384_dims(conn):
    t = conn.execute(
        "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
        "WHERE attrelid='chunks'::regclass AND attname='embedding'"
    ).fetchone()["t"]
    assert t == "vector(384)"


def test_duplicate_content_hash_rejected_per_user(conn, user_a, user_b):
    ins = "INSERT INTO documents (user_id, source, content_hash) VALUES (%s, 'note', 'h1')"
    conn.execute(ins, (user_a,))
    conn.execute(ins, (user_b,))  # other user: allowed
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(ins, (user_a,))


def test_tsv_is_generated(conn, user_a):
    doc_id = conn.execute(
        "INSERT INTO documents (user_id, source, content_hash) VALUES (%s, 'note', 'h1') "
        "RETURNING id",
        (user_a,),
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO chunks (user_id, document_id, ord, text, embedding) "
        "VALUES (%s, %s, 0, %s, %s)",
        (user_a, doc_id, "We use pgvector for search", [0.0] * 384),
    )
    assert conn.execute(
        "SELECT tsv @@ to_tsquery('english','pgvector') AS m FROM chunks"
    ).fetchone()["m"]
