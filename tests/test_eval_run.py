import math
from pathlib import Path
from uuid import uuid4

import pytest

import recall.evaluation.run as run_mod
from recall.evaluation.run import (
    EVAL_USER_ID as EVAL_USER,
)
from recall.evaluation.run import (
    Gold,
    Question,
    label_hits,
    load_questions,
    run_eval,
    write_results_md,
)
from recall.ingest.loader import discover
from recall.ingest.pipeline import ingest_paths
from recall.models import SearchHit


def hit(**kw) -> SearchHit:
    base = {
        "chunk_id": uuid4(),
        "document_id": uuid4(),
        "title": None,
        "source_ref": "fx-1",
        "text": "",
        "score": 1.0,
    }
    return SearchHit(**{**base, **kw})


def test_label_hits_matches_once():
    g = [Gold(doc="fx-1", contains="pgvector")]
    hits = [
        hit(source_ref="fx-1", text="uses PGVECTOR"),
        hit(source_ref="fx-1", text="pgvector again"),
        hit(source_ref="fx-2", text="pgvector"),
    ]
    assert label_hits(hits, g) == [{"fx-1::pgvector"}, {"fx-1::pgvector"}, set()]


def test_load_questions_bad_line(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"id":"q1"}\n')
    with pytest.raises(ValueError, match="line 1"):
        load_questions(p)


def test_run_eval_end_to_end(conn, embedder, tmp_path):
    root = Path(__file__).parent / "fixtures" / "eval"
    ingest_paths(
        conn, discover(root / "corpus"), user_id=EVAL_USER, embedder=embedder, root=root / "corpus"
    )
    qs = load_questions(root / "questions.jsonl")
    res = run_eval(
        conn,
        user_id=EVAL_USER,
        embedder=embedder,
        questions=qs,
        modes=["vector", "keyword", "hybrid"],
    )
    kw = next(r for r in res if r.mode == "keyword")
    assert (
        kw.by_type["single_fact"]["recall_at_10"] == 1.0
    )  # the fixture's single-fact question hinges on an exact term
    out = tmp_path / "results.md"
    write_results_md(res, out, corpus_docs=3, n_questions=3, embed_model="hash")
    text = out.read_text()
    assert (
        "| Vector only |" in text
        and "| Hybrid (RRF) |" in text
        and "| Hybrid + re-rank |  |" in text
    )


def test_label_hits_multi_key_and_prefix_ignored():
    g = [Gold(doc="fx-1", contains="alpha"), Gold(doc="fx-1", contains="beta")]
    both = hit(title="T", text="T\n\nAlpha and beta")
    prefix_only = hit(title="alpha co", text="alpha co\n\nunrelated body")
    assert label_hits([both, prefix_only], g) == [{"fx-1::alpha", "fx-1::beta"}, set()]


def test_run_eval_averages(monkeypatch):
    qs = [
        Question(
            id="a",
            type="single_fact",
            question="qa",
            gold=[Gold(doc="d1", contains="x")],
            answer="",
        ),
        Question(
            id="b",
            type="multi_hop",
            question="qb",
            gold=[Gold(doc="d1", contains="x"), Gold(doc="d2", contains="y")],
            answer="",
        ),
    ]
    miss = hit(source_ref="d9", text="nothing")
    fixed = {
        "qa": [miss, hit(source_ref="d1", text="x here")],
        "qb": [hit(source_ref="d1", text="x here"), hit(source_ref="d2", text="y here")],
    }
    monkeypatch.setattr(run_mod, "search", lambda conn, **kw: fixed[kw["query"]])
    (r,) = run_eval(None, user_id=EVAL_USER, embedder=None, questions=qs, modes=["hybrid"])
    assert r.recall_at_10 == 1.0
    assert r.mrr == pytest.approx((0.5 + 1.0) / 2)
    n_a = 1 / math.log2(3)
    n_b = 1.0
    assert r.ndcg_at_10 == pytest.approx((n_a + n_b) / 2)
    assert r.by_type["single_fact"] == {"recall_at_10": 1.0, "mrr": 0.5, "n": 1.0}
    assert r.by_type["multi_hop"] == {"recall_at_10": 1.0, "mrr": 1.0, "n": 1.0}


def test_run_eval_two_gold_in_one_chunk(monkeypatch):
    q = Question(
        id="c",
        type="multi_hop",
        question="qc",
        gold=[Gold(doc="d1", contains="x"), Gold(doc="d1", contains="y")],
        answer="",
    )
    monkeypatch.setattr(run_mod, "search", lambda conn, **kw: [hit(source_ref="d1", text="x y")])
    (r,) = run_eval(None, user_id=EVAL_USER, embedder=None, questions=[q], modes=["vector"])
    assert (r.recall_at_10, r.mrr, r.ndcg_at_10) == (1.0, 1.0, 1.0)
