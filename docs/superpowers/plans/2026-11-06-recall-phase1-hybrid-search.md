# Recall Phase 1 — Foundation and Hybrid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working Recall core: one command ingests a folder of job posts, emails and notes into Postgres, and vector, keyword and RRF-hybrid search work from the CLI and HTTP API, with an eval harness that produces the first three ablation rows.

**Architecture:** One PostgreSQL 16 + pgvector database (Docker Compose) holds documents and chunks; chunks carry a 384-dim embedding (HNSW) and a generated `tsvector` (GIN). A Python 3.12 package `recall/` provides normalize → chunk → embed → insert ingestion, three search modes (hybrid fused in a single SQL query with RRF), a Typer CLI, a FastAPI app and an eval runner. Tests run against a real `recall_test` database; unit tests use a deterministic hashing embedder so they need no model download.

**Tech Stack:** Python 3.12, uv, PostgreSQL 16 + pgvector ≥ 0.8, psycopg 3 + psycopg-pool + pgvector-python, fastembed (`BAAI/bge-small-en-v1.5`), FastAPI + uvicorn, Typer, Pydantic v2 + pydantic-settings, BeautifulSoup4, python-frontmatter, pytest, ruff, anthropic SDK (Task 11 only).

**Spec:** `docs/spec/recall-brief.md` (Phase 1 of its Build plan; Phases 2–4 get their own plans, listed at the end).

## Global Constraints

- Python `>=3.12,<3.13`, managed by uv (the system Python 3.9 is never used).
- PostgreSQL 16 with pgvector extension version ≥ 0.8.0 (needed for `hnsw.iterative_scan`).
- Embedding model `BAAI/bge-small-en-v1.5`, dimension **384**; chunks column is `vector(384)`.
- Vector and keyword retrievers each return top **50**; RRF score = Σ 1 / (**60** + rank).
- Every row carries `user_id`, and every query filters on `user_id` explicitly (RLS is added in Phase 3 on top of this, not instead of it).
- Parameterized SQL only; no user input formatted into SQL strings.
- Real emails/posts live in `data/private/` (gitignored) and are never committed.
- Never invent eval numbers: `eval/results.md` is only ever written by `recall eval`.
- Type hints everywhere; Pydantic models at every boundary (CLI/API/eval files); stdlib `logging`, no `print` outside the CLI.
- Postgres is exposed on host port **5433** (avoids clashing with a local Postgres on 5432).
- Default dev user id `00000000-0000-0000-0000-000000000001`; eval user id `00000000-0000-0000-0000-0000000000e1`.

## Deviations from the spec (decided here)

- `documents` gains `metadata jsonb NOT NULL DEFAULT '{}'` to hold `company` (needed for contextual chunk prefixes) and other front-matter fields.
- Phase 1 migration creates only `documents` and `chunks`; graph, applications and jobs tables arrive with the phases that use them.
- Gold labels in `questions.jsonl` are `{doc: source_ref, contains: phrase}` instead of chunk ids, because chunk UUIDs change every re-ingest and every chunking tweak. A hit matches a gold item when its document's `source_ref` equals `doc` and its text contains `contains` (case-insensitive).
- "Tokens" in chunk sizing are counted as whitespace words. Limits are 300 target / 380 max / 40 overlap words, which keeps chunks under bge-small's 512-token window (~1.3 tokens per word).

## Review Focus

1. Re-ingesting the same folder, or the same content with different whitespace/HTML wrapping, must add zero documents and zero chunks (Task 3 hash test, Task 6 `test_reingest_is_noop`).
2. Keyword queries with punctuation, operators or only stopwords (`C++`, `"SDE 3" -intern`, `the`, `'`) must never raise; a keyword query with no matches returns `[]` and hybrid falls back to vector order (Task 7 `test_keyword_odd_queries_do_not_raise`, `test_hybrid_falls_back_to_vector`).
3. One bad file in a folder (invalid front-matter YAML, non-UTF-8 bytes, empty body) must not stop the batch; the report lists it and the good files are ingested (Task 6 `test_bad_files_do_not_abort_batch`).
4. A long document with no headings (1,000+ words) must still produce chunks of ≤ 380 words with 40-word overlap and lose no text at the end (Task 4 `test_long_unstructured_text_windows`).
5. Searching as a user with no data, or before anything is ingested, returns an empty list over CLI and HTTP 200, not an error (Task 7 `test_search_empty_user`, Task 8 `test_search_no_data_returns_empty`).

## How to test as you go

- `make up` starts Postgres; `make test` runs the fast suite (no model download); `make test-all` adds `@pytest.mark.slow` tests that use the real embedding model.
- Each task ends with a **Try it** line: a command you can run by hand to see the feature working.
- Push after each task (`git push`) once the GitHub remote exists (see Task 1, Step 7).

## File Structure

```text
job-agent/
├── pyproject.toml              # deps, ruff, pytest markers, `recall` console script
├── docker-compose.yml          # postgres (pgvector/pgvector:pg16) on 5433
├── docker/initdb/01-test-db.sql
├── Makefile                    # up | down | migrate | test | test-all | ingest | search | api | eval | lint
├── .env.example
├── migrations/001_schema.sql
├── recall/
│   ├── __init__.py
│   ├── config.py               # Settings (pydantic-settings)
│   ├── db.py                   # connect(), make_pool()
│   ├── migrate.py              # apply_migrations()
│   ├── models.py               # RawDocument, Chunk, SearchHit, IngestReport, Mode
│   ├── ingest/
│   │   ├── normalize.py        # normalize(), content_hash()
│   │   ├── loader.py           # discover(), load_file()
│   │   ├── chunk.py            # chunk_document()
│   │   ├── embed.py            # Embedder, FastEmbedder, HashEmbedder, get_embedder()
│   │   └── pipeline.py         # ingest_paths()
│   ├── store.py                # document_exists(), insert_document_with_chunks()
│   ├── retrieval/search.py     # vector_search(), keyword_search(), hybrid_search(), search()
│   ├── api/app.py              # create_app()
│   ├── evaluation/
│   │   ├── metrics.py          # recall_at_k(), reciprocal_rank(), ndcg_at_k(), percentile()
│   │   └── run.py              # Question, label_hits(), run_eval(), write_results_md()
│   └── cli.py                  # Typer app: migrate, ingest, search, eval
├── scripts/
│   ├── generate_corpus.py      # Claude-generated synthetic corpus from eval/world.yaml
│   └── draft_questions.py      # Claude-drafted questions -> eval/questions.draft.jsonl
├── eval/
│   ├── world.yaml              # seeded companies/roles/skills/people/statuses
│   ├── corpus/                 # generated synthetic docs (committed)
│   ├── questions.jsonl         # 50 human-verified questions
│   └── results.md              # written only by `recall eval`
└── tests/
    ├── conftest.py
    ├── fixtures/docs/          # 5 tiny hand-written docs + 3 bad files
    ├── fixtures/eval/          # 3-doc corpus + 3 questions for the runner test
    └── test_*.py
```

---

### Task 1: Project scaffold, Postgres in Docker, test database fixture

**Files:**
- Create: `pyproject.toml`, `docker-compose.yml`, `docker/initdb/01-test-db.sql`, `Makefile`, `.env.example`, `recall/__init__.py`, `recall/config.py`, `recall/db.py`, `tests/conftest.py`, `tests/test_db.py`

**Interfaces:**
- Produces:
  - `recall.config.Settings` (pydantic-settings, env prefix `RECALL_`): `database_url: str = "postgresql://recall:recall@localhost:5433/recall"`, `test_database_url: str = "postgresql://recall:recall@localhost:5433/recall_test"`, `embed_model: str = "BAAI/bge-small-en-v1.5"`, `embed_dim: int = 384`, `default_user_id: UUID = UUID("00000000-0000-0000-0000-000000000001")`; `get_settings() -> Settings` (cached).
  - `recall.db.connect(url: str | None = None) -> psycopg.Connection` — opens a connection with `pgvector.psycopg.register_vector` applied (after the extension exists) and `row_factory=dict_row`.
  - `recall.db.make_pool(url: str) -> psycopg_pool.ConnectionPool` — same configure hook.
  - pytest fixtures: `db_url` (session), `conn` (function; truncates `documents CASCADE` after each test, tolerant of the table not existing yet), `user_a`, `user_b` (fixed UUIDs).

- [ ] **Step 1: Install uv and create the project**

Run: `brew install uv && uv python install 3.12 && uv init --package --name recall --python 3.12 .`
Then edit `pyproject.toml`: dependencies `fastapi`, `uvicorn[standard]`, `psycopg[binary]`, `psycopg-pool`, `pgvector`, `pydantic>=2`, `pydantic-settings`, `typer`, `beautifulsoup4`, `python-frontmatter`, `fastembed`; dev group `pytest`, `httpx`, `ruff`; console script `recall = "recall.cli:app"`; `[tool.pytest.ini_options] markers = ["slow: uses the real embedding model"]`, `addopts = "-ra"`.
Expected: `uv sync` completes; `uv run python --version` prints 3.12.x.

- [ ] **Step 2: Docker Compose and test DB**

`docker-compose.yml`: one service `postgres`, image `pgvector/pgvector:pg16`, env `POSTGRES_USER=recall POSTGRES_PASSWORD=recall POSTGRES_DB=recall`, port `5433:5432`, named volume, `pg_isready` healthcheck, mounts `./docker/initdb` to `/docker-entrypoint-initdb.d`. `01-test-db.sql` = `CREATE DATABASE recall_test;`.
Makefile targets: `up` (`docker compose up -d --wait`), `down`, `test` (`uv run pytest -m "not slow"`), `test-all` (`uv run pytest`), `lint` (`uv run ruff check . && uv run ruff format --check .`). Later tasks add `migrate`, `ingest`, `search`, `api`, `eval`.
Run: `make up` → Expected: container healthy.

- [ ] **Step 3: Write the failing test** (`tests/test_db.py`)

```python
def test_pgvector_extension_available(conn):
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    version = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()["extversion"]
    major, minor, *_ = (int(p) for p in version.split("."))
    assert (major, minor) >= (0, 8)

def test_server_is_postgres_16(conn):
    assert conn.execute("SHOW server_version_num").fetchone()["server_version_num"].startswith("16")
```

- [ ] **Step 4: Run to verify it fails**

Run: `make test` → Expected: FAIL/ERROR (`recall.db` / fixture `conn` not found).

- [ ] **Step 5: Implement `Settings`, `get_settings`, `connect`, `make_pool`, and `tests/conftest.py`**

`connect` uses `autocommit=False`. `conftest.py`: `db_url` reads `get_settings().test_database_url`; if connecting fails, call `pytest.exit("Postgres not reachable on 5433 — run `make up`", returncode=2)` (fail loudly, never silently skip).

- [ ] **Step 6: Run to verify it passes**

Run: `make test` → Expected: 2 passed.

- [ ] **Step 7: Commit and create the remote**

```bash
git add -A && git commit -m "chore: scaffold recall with uv, docker postgres+pgvector and test fixtures"
```
Remote (needs the user's go-ahead, since it publishes code): `gh repo create recall --private --source . --push` (switch to `--public` when Phase 2 results are in).

**Try it:** `docker compose exec postgres psql -U recall -c '\l'` lists `recall` and `recall_test`.

---

### Task 2: SQL migrations and the Phase 1 schema

**Files:**
- Create: `migrations/001_schema.sql`, `recall/migrate.py`, `tests/test_migrate.py`
- Modify: `tests/conftest.py` (session fixture resets schema then migrates), `recall/cli.py` (create with `migrate` command), `Makefile` (`migrate`)

**Interfaces:**
- Consumes: `recall.db.connect`.
- Produces:
  - `recall.migrate.MIGRATIONS_DIR: Path` (repo `migrations/`).
  - `recall.migrate.apply_migrations(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]` — applies `*.sql` files in name order not yet in `schema_migrations(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())`, each in its own transaction; returns names applied this call.
  - CLI: `recall migrate` prints applied names or "up to date".
  - Schema: `documents` and `chunks` exactly as in the spec's Data model, plus `documents.metadata jsonb NOT NULL DEFAULT '{}'` and `CREATE INDEX chunks_document ON chunks (document_id)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_migrations_create_tables(conn):
    names = {r["table_name"] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'")}
    assert {"documents", "chunks", "schema_migrations"} <= names

def test_apply_migrations_is_idempotent(conn):
    assert apply_migrations(conn) == []

def test_chunks_embedding_is_384_dims(conn):
    t = conn.execute("SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
                     "WHERE attrelid='chunks'::regclass AND attname='embedding'").fetchone()["t"]
    assert t == "vector(384)"

def test_duplicate_content_hash_rejected_per_user(conn, user_a, user_b):
    ins = "INSERT INTO documents (user_id, source, content_hash) VALUES (%s, 'note', 'h1')"
    conn.execute(ins, (user_a,)); conn.execute(ins, (user_b,))   # other user: allowed
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(ins, (user_a,))

def test_tsv_is_generated(conn, user_a):
    # insert a document + one chunk with a zero vector, then:
    assert conn.execute("SELECT tsv @@ to_tsquery('english','pgvector') AS m FROM chunks").fetchone()["m"]
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_migrate.py -v` → FAIL (no module `recall.migrate`).

- [ ] **Step 3: Implement `001_schema.sql` and `apply_migrations`**; conftest's session fixture runs `DROP SCHEMA public CASCADE; CREATE SCHEMA public;` on `recall_test` then `apply_migrations`.

- [ ] **Step 4: Run to verify they pass** — `make test` → all pass.

- [ ] **Step 5: Commit** — `git commit -m "feat: plain-SQL migrations with documents and chunks schema"`

**Try it:** `make migrate` twice: first prints `001_schema.sql`, second prints `up to date`.

---

### Task 3: Normalization and content hashing

**Files:**
- Create: `recall/ingest/__init__.py`, `recall/ingest/normalize.py`, `tests/test_normalize.py`

**Interfaces:**
- Produces:
  - `normalize(raw: str) -> str`
  - `content_hash(text: str) -> str` — lowercase hex sha256 of the UTF-8 bytes.

Rules, in order: if `raw` contains an HTML tag (`<(html|body|p|div|br|li|ul|ol|h[1-6]|span|table|tr|td|b|i|a|strong|em)\b`, case-insensitive), parse with BeautifulSoup (`html.parser`), drop `script`/`style`, insert `"\n"` around block elements (`p, div, br, li, h1–h6, tr`) and take text; then cut everything from the first line matching `^On .+ wrote:\s*$` or `^-- ?$`; drop lines starting with `>`; collapse runs of spaces/tabs to one space; strip each line; collapse 3+ newlines to 2; strip.

- [ ] **Step 1: Write the failing tests**

```python
def test_html_block_and_inline():
    assert normalize("<p>Hello <b>world</b></p><p>Bye</p>") == "Hello world\nBye"

def test_script_and_style_removed():
    assert normalize("<div>Job<script>x=1</script><style>.a{}</style></div>") == "Job"

def test_signature_cut():
    assert normalize("Hi,\nThanks\n-- \nJane Doe\nRecruiter") == "Hi,\nThanks"

def test_quoted_reply_cut():
    raw = "Sounds good.\n\nOn Mon, 3 Nov 2026 at 10:00, Jane <j@x.com> wrote:\n> earlier text"
    assert normalize(raw) == "Sounds good."

def test_inline_quotes_dropped():
    assert normalize("Yes\n> old line\nThursday works") == "Yes\nThursday works"

def test_whitespace_collapsed():
    assert normalize("  a   b \n\n\n\n c ") == "a b\n\nc"

def test_html_only_whitespace_is_empty():
    assert normalize("<div>   </div>") == ""

def test_hash_ignores_wrapping():
    assert content_hash(normalize("<p>Same  text</p>")) == content_hash(normalize("Same text\n"))

def test_hash_is_sha256_hex():
    assert content_hash("a") == "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"

def test_less_than_in_plain_text_is_not_html():
    assert normalize("salary < 50k, team > 5") == "salary < 50k, team > 5"
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_normalize.py -v` → FAIL.
- [ ] **Step 3: Implement `normalize` and `content_hash`.**
- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/test_normalize.py -v` → 10 passed.
- [ ] **Step 5: Commit** — `git commit -m "feat: normalize html/email text and hash content"`

**Try it:** `uv run python -c "from recall.ingest.normalize import normalize; print(normalize(open('some.html').read()))"`

---

### Task 4: Document loading and structural chunking

**Files:**
- Create: `recall/models.py`, `recall/ingest/loader.py`, `recall/ingest/chunk.py`, `tests/test_loader.py`, `tests/test_chunk.py`, `tests/fixtures/docs/` (5 good docs: 2 job posts with `##` headings, 1 email thread with two `From:` messages, 1 note, 1 HTML job post)

**Interfaces:**
- Produces (in `recall/models.py`):
  - `Source = Literal["job_post", "email", "note", "interview"]`
  - `RawDocument(BaseModel)`: `path: Path`, `source: Source = "note"`, `title: str`, `company: str | None = None`, `source_ref: str`, `body: str`, `metadata: dict[str, Any] = {}`
  - `Chunk(BaseModel)`: `ord: int`, `text: str`
- Produces (functions):
  - `loader.SUFFIXES = {".md", ".txt", ".html", ".eml"}`
  - `loader.discover(root: Path) -> list[Path]` — recursive, sorted, only `SUFFIXES`, skips dotfiles.
  - `loader.load_file(path: Path, root: Path | None = None) -> RawDocument` — bytes decoded UTF-8 with `errors="replace"`; YAML front matter via `python-frontmatter`; `title` defaults to `path.stem`; `source_ref` defaults to `str(path.relative_to(root))` when `root` is given, else `path.name`; unknown front-matter keys go to `metadata`. Invalid YAML or an invalid `source` raises `ValueError` naming the file.
  - `chunk.word_count(text: str) -> int`
  - `chunk.chunk_document(text: str, *, source: Source, title: str | None, company: str | None, target_words: int = 300, max_words: int = 380, overlap_words: int = 40) -> list[Chunk]`

Chunking algorithm:
1. Split into sections. `job_post`: a new section starts at a markdown heading (`^#{1,6} `) or a line that is only a known section name with optional trailing colon (about, about the role, about us, responsibilities, what you'll do, requirements, qualifications, nice to have, benefits, compensation — case-insensitive). `email`: a new section starts at each line beginning `From:`. Other sources: split on blank lines (paragraphs).
2. Pack: merge consecutive sections while the merged body stays ≤ `target_words`. Any single section > `max_words` is split into word windows of `target_words` with `overlap_words` overlap (the last window ends at the last word).
3. Prefix every chunk body with `f"{title} — {company}\n\n"`, or just `f"{title}\n\n"` when company is missing, or nothing when both are missing. The prefix doesn't count toward word limits. `ord` runs 0..n-1. Empty/whitespace text → `[]`.

- [ ] **Step 1: Write the failing tests**

```python
def words(n, w="lorem"): return " ".join(f"{w}{i}" for i in range(n))

def test_small_sections_merge():
    text = f"## About\n{words(50,'a')}\n## Requirements\n{words(50,'r')}\n## Benefits\n{words(50,'b')}"
    out = chunk_document(text, source="job_post", title="Senior Backend Engineer", company="Littlebird")
    assert len(out) == 1 and "a0" in out[0].text and "b49" in out[0].text

def test_large_sections_split_with_prefix():
    text = f"## About\n{words(250,'a')}\n## Requirements\n{words(250,'r')}\n## Benefits\n{words(250,'b')}"
    out = chunk_document(text, source="job_post", title="Senior Backend Engineer", company="Littlebird")
    assert [c.ord for c in out] == [0, 1, 2]
    assert all(c.text.startswith("Senior Backend Engineer — Littlebird\n\n") for c in out)

def test_long_unstructured_text_windows():
    out = chunk_document(words(1000), source="note", title=None, company=None)
    bodies = [c.text.split() for c in out]
    assert all(len(b) <= 380 for b in bodies)
    assert bodies[0][-40:] == bodies[1][:40]
    assert bodies[-1][-1] == "lorem999"

def test_email_splits_by_message():
    text = f"From: Jane\n{words(200,'x')}\nFrom: Me\n{words(200,'y')}"
    assert len(chunk_document(text, source="email", title="Re: next round", company="MongoDB")) == 2

def test_empty_text(): assert chunk_document("  \n", source="note", title="t", company=None) == []

def test_no_title_no_prefix():
    assert chunk_document("hello there", source="note", title=None, company=None)[0].text == "hello there"

def test_load_front_matter(tmp_path):
    p = tmp_path / "jd.md"
    p.write_text("---\nsource: job_post\ntitle: SBE\ncompany: Littlebird\nsource_ref: https://x/1\nremote: true\n---\n## About\nWe build memory.")
    d = load_file(p)
    assert (d.source, d.title, d.company, d.source_ref, d.metadata) == ("job_post", "SBE", "Littlebird", "https://x/1", {"remote": True})

def test_load_defaults_and_bad_bytes(tmp_path):
    p = tmp_path / "note.txt"; p.write_bytes(b"caf\xe9 notes")
    d = load_file(p, root=tmp_path)
    assert d.source == "note" and d.title == "note" and "�" in d.body

def test_invalid_source_raises(tmp_path):
    p = tmp_path / "x.md"; p.write_text("---\nsource: tweet\n---\nhi")
    with pytest.raises(ValueError, match="x.md"): load_file(p)

def test_discover_filters_and_sorts(tmp_path):
    for n in ["b.md", "a.txt", ".hidden.md", "c.pdf"]: (tmp_path / n).write_text("x")
    assert [p.name for p in discover(tmp_path)] == ["a.txt", "b.md"]
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_chunk.py tests/test_loader.py -v` → FAIL.
- [ ] **Step 3: Implement models, loader and chunker** per the rules above.
- [ ] **Step 4: Run to verify they pass** — same command → all pass.
- [ ] **Step 5: Commit** — `git commit -m "feat: front-matter loader and structure-aware contextual chunker"`

**Try it:** `uv run python -c "from pathlib import Path; from recall.ingest.loader import load_file; from recall.ingest.chunk import chunk_document as c; d=load_file(Path('tests/fixtures/docs/<a job post>.md')); [print(x.ord, x.text[:80]) for x in c(d.body, source=d.source, title=d.title, company=d.company)]"`

---

### Task 5: Embedders (real and test double)

**Files:**
- Create: `recall/ingest/embed.py`, `tests/test_embed.py`
- Modify: `tests/conftest.py` (`embedder` fixture → `HashEmbedder()`)

**Interfaces:**
- Produces:
  - `class Embedder(Protocol)`: `dim: int`; `embed(texts: list[str]) -> list[list[float]]` (passages, L2-normalized); `embed_query(text: str) -> list[float]` (L2-normalized).
  - `FastEmbedder(model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64)` — wraps `fastembed.TextEmbedding`; `embed` uses `passage_embed` (or `embed`), `embed_query` uses `query_embed` (bge's query instruction). Model loads lazily on first call.
  - `HashEmbedder(dim: int = 384)` — deterministic bag-of-words: lowercase, split on `\W+`, each token adds 1.0 at `int(sha1(token).hexdigest(), 16) % dim`, then L2-normalize; empty text → zero vector... except normalization: return the zero vector unchanged. `embed_query` = same as passage.
  - `get_embedder(settings: Settings, fake: bool = False) -> Embedder`.

- [ ] **Step 1: Write the failing tests**

```python
def cos(a, b): return sum(x*y for x, y in zip(a, b))

def test_hash_embedder_shape_and_norm():
    v = HashEmbedder().embed(["postgres backend role"])[0]
    assert len(v) == 384 and math.isclose(sum(x*x for x in v), 1.0, rel_tol=1e-6)

def test_hash_embedder_deterministic_and_similarity():
    e = HashEmbedder()
    assert e.embed(["a b"]) == e.embed(["a b"])
    q = e.embed_query("postgres backend")
    assert cos(q, e.embed(["postgres backend job"])[0]) > cos(q, e.embed(["banana bread recipe"])[0])

@pytest.mark.slow
def test_fastembedder_semantics():
    e = FastEmbedder()
    q = e.embed_query("database engineer job")
    near, far = e.embed(["Senior PostgreSQL backend engineer role", "Recipe for banana bread"])
    assert len(q) == 384 and cos(q, near) > cos(q, far)
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_embed.py -v` → FAIL.
- [ ] **Step 3: Implement the protocol, both embedders and `get_embedder`.**
- [ ] **Step 4: Run to verify they pass** — `make test-all` (downloads the ~130 MB model once) → all pass.
- [ ] **Step 5: Commit** — `git commit -m "feat: fastembed bge-small embedder and deterministic hash embedder for tests"`

---

### Task 6: Ingest pipeline, storage and `recall ingest`

**Files:**
- Create: `recall/store.py`, `recall/ingest/pipeline.py`, `tests/test_ingest.py`, `tests/fixtures/bad_docs/` (a sibling of `docs/`, so `discover(docs)` never picks them up: `bad_yaml.md` with `---\ntitle: [unclosed\n---\nx`, `latin1.txt` with byte `\xe9`, `empty.html` with `<div> </div>`)
- Modify: `recall/models.py` (`IngestReport`), `recall/cli.py` (`ingest`), `Makefile` (`ingest`)

**Interfaces:**
- Consumes: `load_file`, `discover`, `normalize`, `content_hash`, `chunk_document`, `Embedder`.
- Produces:
  - `IngestReport(BaseModel)`: `inserted: int = 0`, `duplicates: int = 0`, `empty: int = 0`, `chunks: int = 0`, `failed: list[FailedFile] = []` where `FailedFile(path: str, error: str)`.
  - `store.document_exists(conn, *, user_id: UUID, content_hash: str) -> bool`
  - `store.insert_document_with_chunks(conn, *, user_id: UUID, doc: RawDocument, content_hash: str, chunks: list[Chunk], embeddings: list[list[float]]) -> UUID | None` — one transaction; `INSERT … ON CONFLICT (user_id, content_hash) DO NOTHING RETURNING id`; returns `None` on conflict without inserting chunks; stores `company` + `metadata` in `documents.metadata`; chunks inserted with `executemany`.
  - `pipeline.ingest_paths(conn, paths: list[Path], *, user_id: UUID, embedder: Embedder, root: Path | None = None) -> IngestReport` — per file: load → normalize → empty? (`empty += 1`) → hash → `document_exists`? (`duplicates += 1`, skip embedding) → chunk → `embedder.embed` → insert. Any exception for a file is logged with its path, appended to `failed`, its transaction rolled back, and the loop continues.
  - CLI: `recall ingest PATH [--user UUID] [--fake-embeddings]` — PATH may be a file or a folder; prints the report as a table; exit code 1 if `failed` is non-empty.

- [ ] **Step 1: Write the failing tests**

```python
FIX = Path("tests/fixtures/docs")
BAD = Path("tests/fixtures/bad_docs")

def test_ingest_fixture_folder(conn, user_a, embedder):
    r = ingest_paths(conn, discover(FIX), user_id=user_a, embedder=embedder, root=FIX)
    assert r.inserted == 5 and r.chunks >= 5
    assert conn.execute("SELECT count(*) c FROM chunks WHERE user_id=%s", (user_a,)).fetchone()["c"] == r.chunks

def test_reingest_is_noop(conn, user_a, embedder):
    paths = discover(FIX)
    ingest_paths(conn, paths, user_id=user_a, embedder=embedder, root=FIX)
    r2 = ingest_paths(conn, paths, user_id=user_a, embedder=embedder, root=FIX)
    assert (r2.inserted, r2.duplicates, r2.chunks) == (0, 5, 0)

def test_same_content_other_user_is_inserted(conn, user_a, user_b, embedder):
    ingest_paths(conn, discover(FIX), user_id=user_a, embedder=embedder, root=FIX)
    assert ingest_paths(conn, discover(FIX), user_id=user_b, embedder=embedder, root=FIX).inserted == 5

def test_bad_files_do_not_abort_batch(conn, user_a, embedder):
    r = ingest_paths(conn, discover(BAD) + discover(FIX)[:2], user_id=user_a, embedder=embedder)
    assert r.inserted == 3          # latin1.txt + 2 good docs
    assert r.empty == 1             # empty.html
    assert [Path(f.path).name for f in r.failed] == ["bad_yaml.md"]

def test_embedder_failure_leaves_no_orphan_document(conn, user_a):
    class Boom(HashEmbedder):
        def embed(self, texts): raise RuntimeError("model down")
    r = ingest_paths(conn, discover(FIX)[:1], user_id=user_a, embedder=Boom(), root=FIX)
    assert len(r.failed) == 1
    assert conn.execute("SELECT count(*) c FROM documents").fetchone()["c"] == 0

def test_cli_ingest_exit_codes(conn, test_db_env):   # test_db_env: monkeypatch RECALL_DATABASE_URL to the test DB, clear get_settings cache
    ok = CliRunner().invoke(app, ["ingest", str(FIX), "--fake-embeddings"])
    assert ok.exit_code == 0 and "inserted" in ok.output
    bad = CliRunner().invoke(app, ["ingest", str(BAD), "--fake-embeddings"])
    assert bad.exit_code == 1 and "bad_yaml.md" in bad.output
```
Add the `test_db_env` fixture to `tests/conftest.py`.

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_ingest.py -v` → FAIL.
- [ ] **Step 3: Implement `store`, `pipeline` and the CLI command.**
- [ ] **Step 4: Run to verify they pass** — `make test` → all pass.
- [ ] **Step 5: Commit** — `git commit -m "feat: idempotent ingest pipeline and recall ingest CLI"`

**Try it:** `make migrate && uv run recall ingest tests/fixtures/docs` (real model), run it again and see `duplicates 5`, then `docker compose exec postgres psql -U recall -c "select d.title, count(*) from chunks c join documents d on d.id=c.document_id group by 1"`.

---

### Task 7: Vector, keyword and RRF hybrid search + `recall search`

**Files:**
- Create: `recall/retrieval/__init__.py`, `recall/retrieval/search.py`, `tests/test_search.py`
- Modify: `recall/models.py` (`SearchHit`, `Mode`), `recall/cli.py` (`search`), `Makefile` (`search`)

**Interfaces:**
- Consumes: `Embedder.embed_query`, schema from Task 2, `ingest_paths` (tests seed via it).
- Produces:
  - `Mode = Literal["vector", "keyword", "hybrid"]`
  - `SearchHit(BaseModel)`: `chunk_id: UUID`, `document_id: UUID`, `title: str | None`, `source_ref: str | None`, `text: str`, `score: float`
  - `vector_search(conn, *, user_id: UUID, query_embedding: list[float], k: int = 50) -> list[SearchHit]` — `score = 1 - cosine distance`; runs `SET LOCAL hnsw.iterative_scan = relaxed_order` and `SET LOCAL hnsw.ef_search = 100` in its transaction.
  - `keyword_search(conn, *, user_id: UUID, query: str, k: int = 50) -> list[SearchHit]` — `websearch_to_tsquery('english', %s)`, `score = ts_rank_cd`.
  - `hybrid_search(conn, *, user_id: UUID, query: str, query_embedding: list[float], k: int = 10, candidates: int = 50, rrf_k: int = 60) -> list[SearchHit]` — the spec's single-statement CTE query with `user_id = %s AND superseded_by IS NULL` in both arms, joined to `chunks`/`documents` for the hit fields; `score = rrf`.
  - `search(conn, *, user_id: UUID, query: str, embedder: Embedder, mode: Mode = "hybrid", k: int = 10) -> list[SearchHit]` — dispatcher; vector/keyword modes are truncated to `k`.
  - CLI: `recall search QUERY [--user UUID] [--mode hybrid] [-k 10] [--fake-embeddings]` — prints rank, score (4 dp), title, first 120 chars.

- [ ] **Step 1: Write the failing tests** (seed with a `seeded` fixture that inserts 4 docs for `user_a` through `ingest_paths` with `HashEmbedder`: "Postgres pgvector backend role at Littlebird", "Java Spring microservices role", "Kubernetes SRE on-call role", "Banana bread recipe note"; and 1 doc for `user_b`: "Postgres pgvector role at SecretCo")

```python
def test_vector_exact_text_ranks_first(conn, seeded, user_a, embedder):
    hits = search(conn, user_id=user_a, query="Java Spring microservices role", embedder=embedder, mode="vector")
    assert "Java Spring" in hits[0].text

def test_keyword_exact_term(conn, seeded, user_a, embedder):
    hits = search(conn, user_id=user_a, query="pgvector", embedder=embedder, mode="keyword")
    assert len(hits) == 1 and "Littlebird" in hits[0].text

@pytest.mark.parametrize("q", ["C++", '"SDE 3" -intern', "the", "'", "a & b | !c", ""])
def test_keyword_odd_queries_do_not_raise(conn, seeded, user_a, embedder, q):
    search(conn, user_id=user_a, query=q, embedder=embedder, mode="keyword")

def test_hybrid_top_in_both_lists_scores_two_over_61(conn, seeded, user_a, embedder):
    hits = search(conn, user_id=user_a, query="Postgres pgvector backend role at Littlebird", embedder=embedder, mode="hybrid")
    assert "Littlebird" in hits[0].text and hits[0].score == pytest.approx(2 / 61)

def test_hybrid_falls_back_to_vector(conn, seeded, user_a, embedder):
    q = "banana bread java"   # websearch_to_tsquery ANDs terms: no doc has all three, so the keyword arm is empty
    assert search(conn, user_id=user_a, query=q, embedder=embedder, mode="keyword") == []
    h = [x.chunk_id for x in search(conn, user_id=user_a, query=q, embedder=embedder, mode="hybrid", k=2)]
    v = [x.chunk_id for x in search(conn, user_id=user_a, query=q, embedder=embedder, mode="vector", k=2)]
    assert h == v

@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_user_isolation(conn, seeded, user_a, embedder, mode):
    hits = search(conn, user_id=user_a, query="Postgres pgvector role SecretCo", embedder=embedder, mode=mode, k=50)
    assert all("SecretCo" not in h.text for h in hits)

@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_superseded_chunks_excluded(conn, seeded, user_a, embedder, mode):
    conn.execute("UPDATE chunks c SET superseded_by = c.id WHERE text LIKE '%%Kubernetes%%'")  # self-ref is enough to mark it
    hits = search(conn, user_id=user_a, query="Kubernetes SRE", embedder=embedder, mode=mode, k=50)
    assert all("Kubernetes" not in h.text for h in hits)

@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_search_empty_user(conn, seeded, embedder, mode):
    assert search(conn, user_id=uuid4(), query="Postgres", embedder=embedder, mode=mode) == []

def test_k_respected(conn, seeded, user_a, embedder):
    assert len(search(conn, user_id=user_a, query="role", embedder=embedder, mode="hybrid", k=2)) == 2
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_search.py -v` → FAIL.
- [ ] **Step 3: Implement the four functions and the CLI command.** Empty/whitespace `query` in keyword mode returns `[]` without hitting the DB.
- [ ] **Step 4: Run to verify they pass** — `make test` → all pass.
- [ ] **Step 5: Commit** — `git commit -m "feat: vector, keyword and RRF hybrid search with user isolation"`

**Try it:** `uv run recall search "postgres remote role" --mode keyword` then `--mode vector` then `--mode hybrid`, and compare which fixture docs come back.

---

### Task 8: HTTP API

**Files:**
- Create: `recall/api/__init__.py`, `recall/api/app.py`, `tests/test_api.py`
- Modify: `Makefile` (`api`: `uv run uvicorn recall.api.app:create_app --factory --reload --port 8000`)

**Interfaces:**
- Consumes: `search`, `make_pool`, `get_embedder`, `SearchHit`, `Mode`.
- Produces:
  - `create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI` — opens a `ConnectionPool` in the lifespan, closes it on shutdown; endpoints are plain `def` (psycopg is sync; FastAPI runs them in its threadpool).
  - `GET /health` → `{"status": "ok", "db": true}` (runs `SELECT 1`); `db: false` with HTTP 503 if the query fails.
  - `POST /search` body `SearchRequest`: `query: str` (1–2000 chars), `user_id: UUID`, `mode: Mode = "hybrid"`, `k: int = 10` (1–50) → `SearchResponse`: `hits: list[SearchHit]`, `took_ms: float`.

- [ ] **Step 1: Write the failing tests** (client fixture: `TestClient(create_app(Settings(database_url=test_url), embedder=HashEmbedder()))`)

```python
def test_health(client): assert client.get("/health").json() == {"status": "ok", "db": True}

def test_search_returns_hits(client, seeded, user_a):
    r = client.post("/search", json={"query": "pgvector", "user_id": str(user_a), "mode": "keyword"})
    assert r.status_code == 200 and "Littlebird" in r.json()["hits"][0]["text"] and r.json()["took_ms"] >= 0

def test_search_no_data_returns_empty(client):
    r = client.post("/search", json={"query": "anything", "user_id": str(uuid4())})
    assert r.status_code == 200 and r.json()["hits"] == []

@pytest.mark.parametrize("body", [{"query": ""}, {"query": "x", "k": 0}, {"query": "x", "k": 51}, {"query": "x", "mode": "graph"}])
def test_search_validation(client, user_a, body):
    assert client.post("/search", json={"user_id": str(user_a), **body}).status_code == 422
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_api.py -v` → FAIL.
- [ ] **Step 3: Implement `create_app`.**
- [ ] **Step 4: Run to verify they pass** — `make test` → all pass.
- [ ] **Step 5: Commit** — `git commit -m "feat: FastAPI /health and /search"`

**Try it:** `make api`, then `curl -s localhost:8000/search -H 'content-type: application/json' -d '{"query":"postgres","user_id":"00000000-0000-0000-0000-000000000001"}' | jq`, and open `localhost:8000/docs`.

---

### Task 9: Retrieval metrics

**Files:**
- Create: `recall/evaluation/__init__.py`, `recall/evaluation/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces (pure functions over string keys):
  - `recall_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float` — `|set(retrieved[:k]) ∩ relevant| / |relevant|`; empty `relevant` raises `ValueError`.
  - `reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float` — 1/rank of first relevant, else 0.0.
  - `ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float` — binary gain, discount `1/log2(i+2)` for 0-based i; a key counts once.
  - `percentile(values: list[float], p: float) -> float` — linear interpolation (same as numpy's default); empty raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

```python
def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "d"}) == 0.5
    assert recall_at_k(["x", "a"], {"a"}, k=1) == 0.0
    with pytest.raises(ValueError): recall_at_k(["a"], set())

def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], {"a"}) == 0.0

def test_ndcg():
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}) == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(["a", "a"], {"a"}) == pytest.approx(1.0)
    assert ndcg_at_k(["x"], {"a"}) == 0.0

def test_percentile():
    assert percentile([float(i) for i in range(1, 101)], 95) == pytest.approx(95.05)
    assert percentile([7.0], 95) == 7.0
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_metrics.py -v` → FAIL.
- [ ] **Step 3: Implement the four functions.**
- [ ] **Step 4: Run to verify they pass** — same → 4 passed.
- [ ] **Step 5: Commit** — `git commit -m "feat: recall@k, MRR, nDCG and percentile metrics"`

---

### Task 10: Eval runner and `make eval`

**Files:**
- Create: `recall/evaluation/run.py`, `tests/test_eval_run.py`, `tests/fixtures/eval/corpus/` (3 docs with front matter `source_ref: fx-1..3`), `tests/fixtures/eval/questions.jsonl` (3 questions, one per type)
- Modify: `recall/cli.py` (`eval`), `Makefile` (`eval`)

**Interfaces:**
- Consumes: `ingest_paths`, `discover`, `search`, metrics from Task 9.
- Produces:
  - `QuestionType = Literal["single_fact", "filter_semantic", "multi_hop"]`
  - `Gold(BaseModel)`: `doc: str`, `contains: str`; `Question(BaseModel)`: `id: str`, `type: QuestionType`, `question: str`, `gold: list[Gold]` (min 1), `answer: str`
  - `load_questions(path: Path) -> list[Question]` (JSONL; a bad line raises `ValueError` with its line number).
  - `label_hits(hits: list[SearchHit], gold: list[Gold]) -> list[str]` — for each hit, the key `f"{g.doc}::{g.contains}"` of the first not-yet-matched gold item it satisfies (`hit.source_ref == g.doc` and `g.contains.lower() in hit.text.lower()`), else `""`.
  - `ModeResult(BaseModel)`: `mode: Mode`, `recall_at_10: float`, `mrr: float`, `ndcg_at_10: float`, `p95_ms: float`, `cost_per_query_usd: float` (0.0 in Phase 1: local embeddings, no LLM), `by_type: dict[QuestionType, dict[str, float]]` (`recall_at_10`, `mrr`, `n`).
  - `run_eval(conn, *, user_id: UUID, embedder: Embedder, questions: list[Question], modes: list[Mode], k: int = 10) -> list[ModeResult]` — latency measured per query with `time.perf_counter` around `search` (includes query embedding).
  - `write_results_md(results: list[ModeResult], path: Path, *, corpus_docs: int, n_questions: int, embed_model: str) -> None` — writes a header line (date, corpus size, question count, model), the spec's ablation table with rows `Vector only`, `Keyword only`, `Hybrid (RRF)` filled (3 dp, ms 1 dp, `$0.0000`) and the remaining three spec rows left blank, then a per-type table.
  - CLI: `recall eval [--corpus eval/corpus] [--questions eval/questions.jsonl] [--out eval/results.md] [--fake-embeddings]` — ingests the corpus as the eval user (idempotent), runs all three modes, writes the file, prints the table.

- [ ] **Step 1: Write the failing tests**

```python
def test_label_hits_matches_once():
    g = [Gold(doc="fx-1", contains="pgvector")]
    hits = [hit(source_ref="fx-1", text="uses PGVECTOR"), hit(source_ref="fx-1", text="pgvector again"), hit(source_ref="fx-2", text="pgvector")]
    assert label_hits(hits, g) == ["fx-1::pgvector", "", ""]

def test_load_questions_bad_line(tmp_path):
    p = tmp_path / "q.jsonl"; p.write_text('{"id":"q1"}\n')
    with pytest.raises(ValueError, match="line 1"): load_questions(p)

def test_run_eval_end_to_end(conn, embedder, tmp_path):
    root = Path("tests/fixtures/eval")
    ingest_paths(conn, discover(root / "corpus"), user_id=EVAL_USER, embedder=embedder, root=root / "corpus")
    qs = load_questions(root / "questions.jsonl")
    res = run_eval(conn, user_id=EVAL_USER, embedder=embedder, questions=qs, modes=["vector", "keyword", "hybrid"])
    kw = next(r for r in res if r.mode == "keyword")
    assert kw.by_type["single_fact"]["recall_at_10"] == 1.0     # the fixture's single-fact question hinges on an exact term
    out = tmp_path / "results.md"
    write_results_md(res, out, corpus_docs=3, n_questions=3, embed_model="hash")
    text = out.read_text()
    assert "| Vector only |" in text and "| Hybrid (RRF) |" in text and "| Hybrid + re-rank |  |" in text
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_eval_run.py -v` → FAIL.
- [ ] **Step 3: Implement `run.py` and the CLI command.**
- [ ] **Step 4: Run to verify they pass** — `make test` → all pass.
- [ ] **Step 5: Commit** — `git commit -m "feat: eval runner writing ablation results"`

**Try it:** `uv run recall eval --corpus tests/fixtures/eval/corpus --questions tests/fixtures/eval/questions.jsonl --out /tmp/r.md && cat /tmp/r.md`

---

### Task 11: Synthetic corpus, 50 verified questions, first real ablation rows, README

**Files:**
- Create: `eval/world.yaml`, `scripts/generate_corpus.py`, `scripts/draft_questions.py`, `tests/test_scripts.py`, `eval/corpus/**` (generated), `eval/questions.jsonl` (verified), `eval/results.md` (generated), `README.md`
- Modify: `pyproject.toml` (add `anthropic`, `pyyaml`), `.env.example` (`ANTHROPIC_API_KEY=`)

Load the `claude-api` skill before writing either script.

**Interfaces:**
- Consumes: `RawDocument` front-matter format (Task 4), `Question`/`Gold` (Task 10).
- Produces:
  - `eval/world.yaml`: ~40 companies (name, location, remote policy incl. "remote from India" yes/no), ~60 roles (company, title, required skills, status ∈ saved | applied | interviewing | rejected | offer), ~25 people (name, company, role: recruiter/hiring manager/engineer), skills list. This file is the ground truth that makes multi-hop questions checkable.
  - `generate_corpus.py`: `build_jobs(world: dict, target_docs: int) -> list[DocSpec]` (pure: plans ~45% job posts, ~30% recruiter emails, ~15% interview notes, ~10% personal notes, each tied to world entities, with ~5% deliberate near-duplicate reposts); `render(doc: DocSpec, client) -> str` calls `claude-sonnet-5-5` to write the body only (facts come from the spec, the model only writes prose); `main(--target 400 --limit N --out eval/corpus --dry-run)` writes `eval/corpus/<source>/<source_ref>.md` with front matter; skips files that already exist (resumable).
  - `draft_questions.py`: drafts 20 single_fact / 15 filter_semantic / 15 multi_hop questions from `world.yaml` + corpus into `eval/questions.draft.jsonl` in the Task 10 schema.

- [ ] **Step 1: Write the failing tests** (no network; a fake client returns a fixed body)

```python
def test_build_jobs_mix_and_determinism():
    world = yaml.safe_load(Path("eval/world.yaml").read_text())
    jobs = build_jobs(world, target_docs=200)
    assert len(jobs) == 200 and build_jobs(world, 200) == jobs
    share = Counter(j.source for j in jobs)
    assert 0.40 <= share["job_post"] / 200 <= 0.50

def test_render_writes_loadable_front_matter(tmp_path):
    path = write_doc(build_jobs(WORLD, 1)[0], FakeClient("## About\nWe build things."), tmp_path)
    d = load_file(path)
    assert d.source in {"job_post", "email", "note", "interview"} and d.source_ref and d.body

def test_draft_questions_schema(tmp_path):
    out = draft(WORLD, corpus_dir=FIXTURE_CORPUS, client=FakeQuestionClient(), out=tmp_path / "q.jsonl")
    qs = load_questions(out)
    assert Counter(q.type for q in qs) == {"single_fact": 20, "filter_semantic": 15, "multi_hop": 15}
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_scripts.py -v` → FAIL.
- [ ] **Step 3: Write `world.yaml` and implement both scripts.**
- [ ] **Step 4: Run to verify they pass** — `uv run pytest tests/test_scripts.py -v` → PASS.
- [ ] **Step 5: Generate the corpus.** Run `uv run python scripts/generate_corpus.py --limit 5` first and read the 5 docs; then `--target 400`. Expected: ~400 files under `eval/corpus/`, `uv run recall ingest eval/corpus --user 00000000-0000-0000-0000-0000000000e1` reports 0 failed.
- [ ] **Step 6: Draft questions, then human verification (checkpoint — stop and hand to the user).** `uv run python scripts/draft_questions.py` → `eval/questions.draft.jsonl`. The user checks every question, gold `doc`/`contains` and answer against the corpus, fixes or drops bad ones, and saves the result as `eval/questions.jsonl` (50 lines). Do not proceed until the user confirms.
- [ ] **Step 7: Run the real eval.** `make eval` → `eval/results.md` has the three Phase 1 rows filled. Record the numbers as they come out; if hybrid does not beat vector-only, say so in the README rather than tuning until it does.
- [ ] **Step 8: README.** Quickstart (`make up && make migrate && uv run recall ingest eval/corpus && uv run recall search "…"`), the architecture paragraph from the spec, the results table copied from `eval/results.md`, and "What Phase 2 adds".
- [ ] **Step 9: Commit and push**

```bash
git add -A && git commit -m "feat: synthetic eval corpus, verified questions and phase 1 results"
git push
```

---

## After Phase 1: follow-up plans

Each gets its own plan file in `docs/superpowers/plans/` when Phase 1 is merged:

- **Phase 2 — graph and re-ranking:** `002_graph.sql` (entities, edges, chunk_entities, applications); Haiku extraction with Pydantic validation; alias map + embedding entity resolution (0.9); edge upserts with evidence; recursive 2-hop expansion with degree cap; `bge-reranker-base` cross-encoder (top 40 → 15); MMR (λ = 0.7, final 8); structured-query tool; `/answer` with Sonnet citing chunk ids; ablation rows 4–6; README v1 + architecture diagram.
- **Phase 3 — production polish:** `jobs` table + `SKIP LOCKED` worker with retries; near-dup (0.95) supersede, consolidation, decay and forgetting jobs; RLS with a non-owner role + `SET LOCAL app.user_id` + isolation test; bearer auth; OpenTelemetry spans, Prometheus metrics, Grafana JSON; JSON logs with trace ids; Recall as a tool in the job agent; "Where plain RAG broke" write-up.
- **Phase 4 — stretch:** Go ingestion worker, `pg_search` BM25 comparison, public demo deploy.
