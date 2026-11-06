"""SQL storage for documents and chunks. Every query filters on user_id."""

from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from recall.models import Chunk, RawDocument


def document_exists(conn: psycopg.Connection, *, user_id: UUID, content_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM documents WHERE user_id = %s AND content_hash = %s",
        (user_id, content_hash),
    ).fetchone()
    return row is not None


def insert_document_with_chunks(
    conn: psycopg.Connection,
    *,
    user_id: UUID,
    doc: RawDocument,
    content_hash: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> UUID | None:
    """Insert a document and its chunks atomically; None if (user, hash) already exists."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    with conn.transaction():
        row = conn.execute(
            """
            INSERT INTO documents (user_id, source, source_ref, title, content_hash, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, content_hash) DO NOTHING
            RETURNING id
            """,
            (
                user_id,
                doc.source,
                doc.source_ref,
                doc.title,
                content_hash,
                Jsonb({**doc.metadata, "company": doc.company}),
            ),
        ).fetchone()
        if row is None:
            return None
        doc_id: UUID = row["id"]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (user_id, document_id, ord, text, embedding)"
                " VALUES (%s, %s, %s, %s, %s)",
                [
                    (user_id, doc_id, c.ord, c.text, emb)
                    for c, emb in zip(chunks, embeddings, strict=True)
                ],
            )
    return doc_id
