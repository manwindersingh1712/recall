"""Vector, keyword and RRF hybrid search over a user's live chunks."""

from uuid import UUID

import numpy as np
from psycopg import Connection

from recall.ingest.embed import Embedder
from recall.models import Mode, SearchHit

_HIT_COLUMNS = "c.id AS chunk_id, c.document_id, d.title, d.source_ref, c.text"

_VECTOR_SQL = f"""
SELECT * FROM (
  SELECT {_HIT_COLUMNS}, 1 - (c.embedding <=> %(emb)s) AS score,
         c.embedding <=> %(emb)s AS dist
  FROM chunks c JOIN documents d ON d.id = c.document_id
  WHERE c.user_id = %(user)s AND c.superseded_by IS NULL
  ORDER BY c.embedding <=> %(emb)s
  LIMIT %(k)s
) t
ORDER BY dist, chunk_id
"""

_KEYWORD_SQL = f"""
SELECT {_HIT_COLUMNS}, ts_rank_cd(c.tsv, q.query) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id,
     websearch_to_tsquery('english', %(query)s) AS q(query)
WHERE c.user_id = %(user)s AND c.superseded_by IS NULL AND c.tsv @@ q.query
ORDER BY score DESC, c.id
LIMIT %(k)s
"""

_HYBRID_SQL = f"""
WITH vec AS (
  SELECT id, row_number() OVER (ORDER BY dist, id) AS r
  FROM (SELECT id, embedding <=> %(emb)s AS dist
        FROM chunks
        WHERE user_id = %(user)s AND superseded_by IS NULL
        ORDER BY embedding <=> %(emb)s
        LIMIT %(cand)s) t
),
kw AS (
  SELECT id, row_number() OVER (ORDER BY score DESC, id) AS r
  FROM (SELECT id, ts_rank_cd(tsv, websearch_to_tsquery('english', %(query)s)) AS score
        FROM chunks
        WHERE user_id = %(user)s AND superseded_by IS NULL
          AND tsv @@ websearch_to_tsquery('english', %(query)s)
        ORDER BY score DESC, id
        LIMIT %(cand)s) t
),
fused AS (
  SELECT id, SUM(1.0 / (%(rrf_k)s + r)) AS rrf
  FROM (SELECT * FROM vec UNION ALL SELECT * FROM kw) s
  GROUP BY id
)
SELECT {_HIT_COLUMNS}, f.rrf::float8 AS score
FROM fused f
JOIN chunks c ON c.id = f.id
JOIN documents d ON d.id = c.document_id
ORDER BY f.rrf DESC, c.id
LIMIT %(k)s
"""

# Keyword-only variant used when the query embedding is all zeros.
_KEYWORD_RRF_SQL = f"""
WITH kw AS (
  SELECT id, row_number() OVER (ORDER BY score DESC, id) AS r
  FROM (SELECT id, ts_rank_cd(tsv, websearch_to_tsquery('english', %(query)s)) AS score
        FROM chunks
        WHERE user_id = %(user)s AND superseded_by IS NULL
          AND tsv @@ websearch_to_tsquery('english', %(query)s)
        ORDER BY score DESC, id
        LIMIT %(cand)s) t
)
SELECT {_HIT_COLUMNS}, (1.0 / (%(rrf_k)s + kw.r))::float8 AS score
FROM kw
JOIN chunks c ON c.id = kw.id
JOIN documents d ON d.id = c.document_id
ORDER BY kw.r
LIMIT %(k)s
"""


def _is_zero(v: list[float]) -> bool:
    return not any(v)


def _run(conn: Connection, sql: str, params: dict) -> list[SearchHit]:
    # SET LOCAL only lasts for a transaction; the connection is autocommit.
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        rows = conn.execute(sql, params).fetchall()
    return [SearchHit(**{f: r[f] for f in SearchHit.model_fields}) for r in rows]


def vector_search(
    conn: Connection, *, user_id: UUID, query_embedding: list[float], k: int = 50
) -> list[SearchHit]:
    """Top-k chunks by cosine similarity (score = 1 - cosine distance)."""
    if _is_zero(query_embedding):
        return []  # cosine distance is undefined for a zero vector
    emb = np.asarray(query_embedding, dtype=np.float32)
    return _run(conn, _VECTOR_SQL, {"emb": emb, "user": user_id, "k": k})


def keyword_search(conn: Connection, *, user_id: UUID, query: str, k: int = 50) -> list[SearchHit]:
    """Top-k chunks by ts_rank_cd over websearch_to_tsquery."""
    if not query.strip():
        return []
    return _run(conn, _KEYWORD_SQL, {"query": query, "user": user_id, "k": k})


def hybrid_search(
    conn: Connection,
    *,
    user_id: UUID,
    query: str,
    query_embedding: list[float],
    k: int = 10,
    candidates: int = 50,
    rrf_k: int = 60,
) -> list[SearchHit]:
    """Reciprocal rank fusion of the vector and keyword arms (score = rrf)."""
    params = {"query": query, "user": user_id, "k": k, "cand": candidates, "rrf_k": rrf_k}
    if _is_zero(query_embedding):
        if not query.strip():
            return []
        return _run(conn, _KEYWORD_RRF_SQL, params)
    params["emb"] = np.asarray(query_embedding, dtype=np.float32)
    return _run(conn, _HYBRID_SQL, params)


def search(
    conn: Connection,
    *,
    user_id: UUID,
    query: str,
    embedder: Embedder,
    mode: Mode = "hybrid",
    k: int = 10,
) -> list[SearchHit]:
    """Dispatch to the requested retrieval mode and return at most k hits."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if mode == "keyword":
        return keyword_search(conn, user_id=user_id, query=query, k=k)
    emb = embedder.embed_query(query)
    if mode == "vector":
        return vector_search(conn, user_id=user_id, query_embedding=emb, k=k)
    return hybrid_search(conn, user_id=user_id, query=query, query_embedding=emb, k=k)
