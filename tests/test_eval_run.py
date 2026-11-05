from pathlib import Path
from uuid import uuid4

import pytest

from recall.evaluation.run import (
    EVAL_USER_ID as EVAL_USER,
)
from recall.evaluation.run import (
    Gold,
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
    assert label_hits(hits, g) == ["fx-1::pgvector", "", ""]


def test_load_questions_bad_line(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"id":"q1"}\n')
    with pytest.raises(ValueError, match="line 1"):
        load_questions(p)


def test_run_eval_end_to_end(conn, embedder, tmp_path):
    root = Path("tests/fixtures/eval")
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
