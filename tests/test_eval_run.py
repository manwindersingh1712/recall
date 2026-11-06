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


def test_ingest_cli_keeps_eval_user_documents(conn, embedder, test_db_env):
    from typer.testing import CliRunner

    from recall.cli import app

    root = Path(__file__).parent / "fixtures" / "eval" / "corpus"
    ingest_paths(conn, discover(root)[:1], user_id=EVAL_USER, embedder=embedder, root=root)
    before = conn.execute("SELECT count(*) AS n FROM documents WHERE user_id = %s", (EVAL_USER,))
    n = before.fetchone()["n"]
    assert n == 1
    docs = Path(__file__).parent / "fixtures" / "docs"
    res = CliRunner().invoke(app, ["ingest", str(docs), "--fake-embeddings"])
    assert res.exit_code == 0, res.output
    after = conn.execute("SELECT count(*) AS n FROM documents WHERE user_id = %s", (EVAL_USER,))
    assert after.fetchone()["n"] == 1


def test_eval_cli_fake_embeddings_requires_out(test_db_env):
    from typer.testing import CliRunner

    from recall.cli import app

    fx = Path(__file__).parent / "fixtures" / "eval"
    res = CliRunner().invoke(
        app,
        [
            "eval",
            "--fake-embeddings",
            "--corpus",
            str(fx / "corpus"),
            "--questions",
            str(fx / "questions.jsonl"),
        ],
    )
    assert res.exit_code == 2


FX = Path(__file__).parent / "fixtures" / "eval"


def _eval_args(out, questions=FX / "questions.jsonl"):
    return [
        "eval",
        "--fake-embeddings",
        "--corpus",
        str(FX / "corpus"),
        "--questions",
        str(questions),
        "--out",
        str(out),
    ]


def test_eval_cli_success_path(conn, embedder, test_db_env, tmp_path):
    from typer.testing import CliRunner

    from recall.cli import app

    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "stray.md").write_text("A stray document that must be wiped.")
    ingest_paths(conn, discover(stray), user_id=EVAL_USER, embedder=embedder, root=stray)
    out = tmp_path / "results.md"
    res = CliRunner().invoke(app, _eval_args(out))
    assert res.exit_code == 0, res.output
    text = out.read_text()
    assert "Corpus: 3 docs" in text
    assert "Gold reachable: 4/4" in text
    assert "MRR@10" in text
    for label in ("Vector only", "Keyword only", "Hybrid (RRF)"):
        row = next(ln for ln in text.splitlines() if ln.startswith(f"| {label} |"))
        assert all(cell.strip() for cell in row.strip("|").split("|"))
    n = conn.execute(
        "SELECT count(*) AS n FROM documents WHERE user_id = %s AND source_ref = %s",
        (EVAL_USER, "stray"),
    ).fetchone()["n"]
    assert n == 0


def test_eval_cli_unreachable_gold_exits_nonzero_without_output(test_db_env, tmp_path):
    from typer.testing import CliRunner

    from recall.cli import app

    bad = tmp_path / "q.jsonl"
    bad.write_text(
        '{"id":"q1","type":"single_fact","question":"x?",'
        '"gold":[{"doc":"fx-1","contains":"phrase that is nowhere"}],"answer":"a"}\n'
    )
    out = tmp_path / "results.md"
    res = CliRunner().invoke(app, _eval_args(out, bad))
    assert res.exit_code == 1
    assert "unreachable" in res.output and "phrase that is nowhere" in res.output
    assert not out.exists()


def test_eval_cli_blank_questions_exits_nonzero_without_output(test_db_env, tmp_path):
    from typer.testing import CliRunner

    from recall.cli import app

    blank = tmp_path / "q.jsonl"
    blank.write_text("\n  \n")
    out = tmp_path / "results.md"
    res = CliRunner().invoke(app, _eval_args(out, blank))
    assert res.exit_code != 0
    assert not out.exists()


def test_run_eval_rejects_empty_questions():
    with pytest.raises(ValueError, match="at least one question"):
        run_eval(None, user_id=EVAL_USER, embedder=None, questions=[], modes=["hybrid"])


def test_load_questions_rejects_blank_file(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text("\n\n")
    with pytest.raises(ValueError, match="no questions"):
        load_questions(p)


@pytest.mark.parametrize("gold", [{"doc": "", "contains": "x"}, {"doc": "d", "contains": ""}])
def test_load_questions_rejects_empty_gold_fields(tmp_path, gold):
    p = tmp_path / "q.jsonl"
    q = {"id": "q", "type": "single_fact", "question": "?", "gold": [gold], "answer": "a"}
    p.write_text(__import__("json").dumps(q) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        load_questions(p)
