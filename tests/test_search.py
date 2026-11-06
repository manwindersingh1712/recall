from uuid import uuid4

import pytest

from recall.retrieval.search import hybrid_search, keyword_search, search, vector_search


def test_vector_exact_text_ranks_first(conn, seeded, user_a, embedder):
    hits = search(
        conn,
        user_id=user_a,
        query="Java Spring microservices role",
        embedder=embedder,
        mode="vector",
    )
    assert "Java Spring" in hits[0].text


def test_keyword_exact_term(conn, seeded, user_a, embedder):
    hits = search(conn, user_id=user_a, query="pgvector", embedder=embedder, mode="keyword")
    assert len(hits) == 1 and "Littlebird" in hits[0].text


@pytest.mark.parametrize("q", ["C++", '"SDE 3" -intern', "the", "'", "a & b | !c", ""])
def test_keyword_odd_queries_do_not_raise(conn, seeded, user_a, embedder, q):
    search(conn, user_id=user_a, query=q, embedder=embedder, mode="keyword")


def test_hybrid_top_in_both_lists_scores_two_over_61(conn, seeded, user_a, embedder):
    hits = search(
        conn,
        user_id=user_a,
        query="Postgres pgvector backend role at Littlebird",
        embedder=embedder,
        mode="hybrid",
    )
    assert "Littlebird" in hits[0].text and hits[0].score == pytest.approx(2 / 61)


def test_hybrid_falls_back_to_vector(conn, seeded, user_a, embedder):
    q = "zebra quartz xylophone"  # no shared term: keyword arm is empty. k=4 covers all chunks, since a LIMIT would pick arbitrarily among tied distances
    assert search(conn, user_id=user_a, query=q, embedder=embedder, mode="keyword") == []
    h = [
        x.chunk_id
        for x in search(conn, user_id=user_a, query=q, embedder=embedder, mode="hybrid", k=4)
    ]
    v = [
        x.chunk_id
        for x in search(conn, user_id=user_a, query=q, embedder=embedder, mode="vector", k=4)
    ]
    assert h == v


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_user_isolation(conn, seeded, user_a, embedder, mode):
    hits = search(
        conn,
        user_id=user_a,
        query="Postgres pgvector role SecretCo",
        embedder=embedder,
        mode=mode,
        k=50,
    )
    assert all("SecretCo" not in h.text for h in hits)


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_superseded_chunks_excluded(conn, seeded, user_a, embedder, mode):
    conn.execute(
        "UPDATE chunks c SET superseded_by = c.id WHERE text LIKE '%%Kubernetes%%'"
    )  # self-ref is enough to mark it
    hits = search(conn, user_id=user_a, query="Kubernetes SRE", embedder=embedder, mode=mode, k=50)
    assert all("Kubernetes" not in h.text for h in hits)


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_search_empty_user(conn, seeded, embedder, mode):
    assert search(conn, user_id=uuid4(), query="Postgres", embedder=embedder, mode=mode) == []


def test_k_respected(conn, seeded, user_a, embedder):
    assert (
        len(search(conn, user_id=user_a, query="role", embedder=embedder, mode="hybrid", k=2)) == 2
    )


@pytest.mark.parametrize("q", ["'", ""])
def test_zero_vector_query(conn, seeded, user_a, embedder, q):
    zero = embedder.embed_query(q)
    assert not any(zero)
    assert vector_search(conn, user_id=user_a, query_embedding=zero) == []
    assert hybrid_search(conn, user_id=user_a, query=q, query_embedding=zero) == []
    assert keyword_search(conn, user_id=user_a, query=q) == []
    assert search(conn, user_id=user_a, query=q, embedder=embedder, mode="hybrid") == []


def test_hybrid_zero_vector_uses_keyword_arm(conn, seeded, user_a):
    hits = hybrid_search(conn, user_id=user_a, query="pgvector", query_embedding=[0.0] * 384)
    assert len(hits) == 1 and "Littlebird" in hits[0].text
    assert hits[0].score == pytest.approx(1 / 61)


def test_vector_scores_non_increasing(conn, seeded, user_a, embedder):
    hits = search(conn, user_id=user_a, query="role", embedder=embedder, mode="vector", k=4)
    scores = [h.score for h in hits]
    assert len(scores) == 4 and scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(conn, seeded, user_a, embedder, k):
    with pytest.raises(ValueError):
        search(conn, user_id=user_a, query="role", embedder=embedder, k=k)


def test_hybrid_vector_arm_uses_hnsw_index(conn, seeded, user_a, embedder):
    import numpy as np
    import psycopg

    from recall.retrieval import search as search_mod

    emb = np.asarray(embedder.embed(["Java Spring microservices role"])[0], dtype=np.float32)
    params = {"query": "role", "user": user_a, "k": 10, "cand": 50, "rrf_k": 60, "emb": emb}
    plan = ""
    with conn.transaction():
        # The fixture is tiny, so the planner prefers cheap btree scans; take them away (the
        # DROP is rolled back) so only the vector index can satisfy the ORDER BY.
        conn.execute("SET LOCAL enable_seqscan = off")
        conn.execute("SET LOCAL enable_bitmapscan = off")
        conn.execute("DROP INDEX chunks_user")
        conn.execute("DROP INDEX chunks_document")
        rows = conn.execute("EXPLAIN " + search_mod._HYBRID_SQL, params).fetchall()
        plan = "\n".join(r["QUERY PLAN"] for r in rows)
        raise psycopg.Rollback
    assert "Index Scan using chunks_vec" in plan, plan


def test_keyword_matches_any_term(conn, seeded, user_a, embedder):
    # A natural-language question contains words no single document has; any-term matching still finds them.
    hits = search(
        conn,
        user_id=user_a,
        query="which pgvector role is listed with banana",
        embedder=embedder,
        mode="keyword",
        k=10,
    )
    texts = " ".join(h.text for h in hits)
    assert "pgvector" in texts and "Banana" in texts
