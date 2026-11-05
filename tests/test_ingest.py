from pathlib import Path

from typer.testing import CliRunner

from recall.cli import app
from recall.ingest.embed import HashEmbedder
from recall.ingest.loader import discover
from recall.ingest.pipeline import ingest_paths

FIX = Path("tests/fixtures/docs")
BAD = Path("tests/fixtures/bad_docs")


def test_ingest_fixture_folder(conn, user_a, embedder):
    r = ingest_paths(conn, discover(FIX), user_id=user_a, embedder=embedder, root=FIX)
    assert r.inserted == 5 and r.chunks >= 5
    assert (
        conn.execute("SELECT count(*) c FROM chunks WHERE user_id=%s", (user_a,)).fetchone()["c"]
        == r.chunks
    )


def test_reingest_is_noop(conn, user_a, embedder):
    paths = discover(FIX)
    ingest_paths(conn, paths, user_id=user_a, embedder=embedder, root=FIX)
    r2 = ingest_paths(conn, paths, user_id=user_a, embedder=embedder, root=FIX)
    assert (r2.inserted, r2.duplicates, r2.chunks) == (0, 5, 0)


def test_same_content_other_user_is_inserted(conn, user_a, user_b, embedder):
    ingest_paths(conn, discover(FIX), user_id=user_a, embedder=embedder, root=FIX)
    assert (
        ingest_paths(conn, discover(FIX), user_id=user_b, embedder=embedder, root=FIX).inserted == 5
    )


def test_bad_files_do_not_abort_batch(conn, user_a, embedder):
    r = ingest_paths(conn, discover(BAD) + discover(FIX)[:2], user_id=user_a, embedder=embedder)
    assert r.inserted == 3  # latin1.txt + 2 good docs
    assert r.empty == 1  # empty.html
    assert [Path(f.path).name for f in r.failed] == ["bad_yaml.md"]


def test_embedder_failure_leaves_no_orphan_document(conn, user_a):
    class Boom(HashEmbedder):
        def embed(self, texts):
            raise RuntimeError("model down")

    r = ingest_paths(conn, discover(FIX)[:1], user_id=user_a, embedder=Boom(), root=FIX)
    assert len(r.failed) == 1
    assert conn.execute("SELECT count(*) c FROM documents").fetchone()["c"] == 0


def test_cli_ingest_exit_codes(conn, test_db_env):
    ok = CliRunner().invoke(app, ["ingest", str(FIX), "--fake-embeddings"])
    assert ok.exit_code == 0 and "inserted" in ok.output
    bad = CliRunner().invoke(app, ["ingest", str(BAD), "--fake-embeddings"])
    assert bad.exit_code == 1 and "bad_yaml.md" in bad.output
