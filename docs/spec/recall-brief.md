# Recall — Memory Engine for Job Agent

Source: https://claude.ai/artifact/RoP18iNphg3sBRrGd2XXvg (exported 2026-11-06; the live doc wins if they differ)

## Why and what

Recall is a memory engine for my job-finding agent. It turns messy inputs (job posts, recruiter emails, notes, interview feedback) into a store the agent can query for structured, connected context. It exists to prove hands-on RAG, hybrid search, re-ranking and graph-based retrieval in **Python on PostgreSQL**, which are the gaps on my resume for roles like Littlebird's Senior Backend Engineer.

Questions it should answer well:

- "Which roles I saved want PostgreSQL and allow remote from India?" (filters + semantic match)
- "What did the MongoDB recruiter say about the next round, and who else is on that team?" (entity lookup + graph hop)
- "Which skills keep showing up in roles where I was rejected?" (multi-hop aggregation; plain vector RAG fails here)

**Done means:**

- One command ingests 500+ documents end to end.
- Hybrid + graph retrieval measurably beats vector-only on a 50-question test set (recall@10, MRR).
- p95 retrieval latency and cost per query are measured and published.
- Public GitHub repo with README, architecture diagram, eval results and a short write-up: "Where plain RAG broke and what the graph fixed."
- The job agent calls Recall as a tool.

**Not in scope:** a web UI (CLI + HTTP API only), model fine-tuning, multi-tenant SaaS billing.

## Architecture

Ingestion writes chunks and the graph into one Postgres. Each query searches three ways at once (vector, keyword, graph), fuses the results, then narrows to about 8 chunks for Sonnet to answer from.

## Tech stack

Everything lives in one PostgreSQL database: vectors, full-text and the graph. That single-store choice is the main design decision to defend in interviews.

| Layer | Choice | Why |
| --- | --- | --- |
| Language & API | Python 3.12, FastAPI, uv | JD asks for Python; async API for the agent to call |
| Database | PostgreSQL 16 + pgvector (HNSW index) | One store for vectors, text and graph; JD names Postgres |
| Keyword search | Postgres full-text (`tsvector`, `ts_rank_cd`); stretch: ParadeDB `pg_search` for true BM25 | Catches exact terms ("pgvector", "SDE 3") that embeddings miss |
| Embeddings | Voyage AI or local `bge-small-en-v1.5` | Claude has no embedding model; local keeps cost at zero |
| LLM | Claude API: Haiku for entity extraction, Sonnet for answers | Cheap model for bulk work, strong model where quality shows |
| Re-ranker | Local cross-encoder (`bge-reranker-base`) or a hosted rerank API | Re-ranking is named in the JD |
| Background jobs | Postgres job table with `SELECT … FOR UPDATE SKIP LOCKED` | No extra queue to run; good talking point |
| Observability | OpenTelemetry → Prometheus + Grafana; optional Langfuse for LLM traces | Already know Grafana; JD stresses observability |
| Local infra | Docker Compose (Postgres, API, worker, Grafana) | One-command setup for reviewers |
| Stretch | Rewrite the ingestion worker in Go | Doubles as Go practice for interviews |

## Data model

Five core tables: documents hold raw sources, chunks hold searchable text, and entities + edges form the graph. Every row carries `user_id`, and row-level security keeps each user's memory isolated.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      uuid NOT NULL,
  source       text NOT NULL,        -- job_post | email | note | interview
  source_ref   text,                 -- URL or message id
  title        text,
  content_hash text NOT NULL,        -- sha256 of normalized text: exact-dup guard
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, content_hash)
);

CREATE TABLE chunks (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          uuid NOT NULL,
  document_id      uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ord              int  NOT NULL,
  text             text NOT NULL,
  embedding        vector(384) NOT NULL,   -- match your embedding model's size
  tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  importance       real NOT NULL DEFAULT 0.5,
  last_accessed_at timestamptz,
  superseded_by    uuid REFERENCES chunks(id),
  created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX chunks_vec ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX chunks_user ON chunks (user_id);

CREATE TABLE entities (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id        uuid NOT NULL,
  type           text NOT NULL,      -- company | role | skill | person | location
  name           text NOT NULL,
  canonical_name text NOT NULL,      -- normalized: 'postgres' -> 'postgresql'
  embedding      vector(384),        -- used for entity resolution
  UNIQUE (user_id, type, canonical_name)
);

CREATE TABLE edges (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           uuid NOT NULL,
  src               uuid NOT NULL REFERENCES entities(id),
  dst               uuid NOT NULL REFERENCES entities(id),
  rel               text NOT NULL,   -- REQUIRES | AT_COMPANY | LOCATED_IN | WORKS_AT
  weight            real NOT NULL DEFAULT 1,
  evidence_chunk_id uuid REFERENCES chunks(id),  -- why this edge exists
  UNIQUE (user_id, src, dst, rel)
);
CREATE INDEX edges_src ON edges (src);
CREATE INDEX edges_dst ON edges (dst);

CREATE TABLE chunk_entities (
  chunk_id  uuid REFERENCES chunks(id)   ON DELETE CASCADE,
  entity_id uuid REFERENCES entities(id) ON DELETE CASCADE,
  PRIMARY KEY (chunk_id, entity_id)
);

CREATE TABLE applications (              -- your own pipeline state
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id        uuid NOT NULL,
  role_entity_id uuid NOT NULL REFERENCES entities(id),
  status         text NOT NULL,          -- saved | applied | interviewing | rejected | offer
  updated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE jobs (                      -- background work queue
  id        bigserial PRIMARY KEY,
  kind      text NOT NULL,               -- ingest | extract | consolidate
  payload   jsonb NOT NULL,
  status    text NOT NULL DEFAULT 'queued',
  attempts  int  NOT NULL DEFAULT 0,
  run_after timestamptz NOT NULL DEFAULT now()
);

-- Per-user isolation. Repeat for documents, entities, edges, applications.
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
CREATE POLICY own_rows ON chunks
  USING (user_id = current_setting('app.user_id')::uuid);
```

The API connects as a non-owner role and runs `SET LOCAL app.user_id = '<id>'` at the start of each transaction.

## Ingestion pipeline

Each document runs through eight steps as background jobs. Every step is idempotent (keyed on `content_hash`), so a crashed worker can safely retry.

1. **Normalize.** Strip HTML, email signatures and quoted replies; compute `content_hash`. If the hash exists for this user, stop (exact duplicate).
2. **Chunk by structure.** Split job posts on their sections (about, responsibilities, requirements) and emails by message, targeting ~300–500 tokens with a small overlap. Prefix each chunk with its document title and company so it makes sense alone (contextual chunking).
3. **Embed** chunks in batches and insert them.
4. **Near-duplicate check.** If a new chunk has cosine similarity above ~0.95 with an existing one (the same job posted on two sites), mark the older one `superseded_by` instead of keeping both.
5. **Extract entities and relations** with Claude Haiku using a strict JSON schema: entities `{type, name}` and relations `{src, rel, dst}`. Validate with Pydantic and drop anything that fails.
6. **Resolve entities.** Canonicalize names with an alias map (`postgres` → `postgresql`, `k8s` → `kubernetes`). If there's no exact match, compare embeddings within the same type and merge above ~0.9 similarity; otherwise create a new entity.
7. **Upsert edges** with `ON CONFLICT (user_id, src, dst, rel) DO UPDATE SET weight = edges.weight + 1`, saving the evidence chunk so every edge can be traced to its source text.
8. **Link** chunks to their entities in `chunk_entities`.

The 0.95 and 0.9 thresholds are starting points. Tune them against the eval set and report what you chose and why.

## Retrieval pipeline

A query goes wide first (vector + keyword + graph), then narrows (re-rank, then MMR) to about 8 chunks plus a short block of graph facts.

1. **Understand the query.** Haiku pulls out filters (location, remote, status) and named entities (companies, skills, people).
2. **Vector search:** top 50 chunks by cosine distance. With RLS filtering, turn on pgvector's iterative index scans so filtered HNSW queries don't come back short.
3. **Keyword search:** top 50 by `ts_rank_cd` using `websearch_to_tsquery`.
4. **Fuse with reciprocal rank fusion (RRF):** score = sum of 1 / (60 + rank) across both lists. No score normalization needed.
5. **Graph expansion.** Seed with the query's entities plus entities in the top fused chunks, walk 1–2 hops, and pull those neighbors' evidence chunks in at a lower prior. Skip hub nodes (e.g. "Python" linked to everything) by capping degree.
6. **Re-rank** the top ~40 candidates with the cross-encoder; keep 15.
7. **MMR** picks the final ~8 for relevance plus diversity (start at λ = 0.7).
8. **Assemble context:** chunks plus compact graph facts such as "Littlebird —REQUIRES→ PostgreSQL (3 sources)". Sonnet answers and cites chunk ids.

Hybrid search with RRF in one query (pass the query embedding and text as parameters so the HNSW index is used):

```sql
WITH vec AS (
  SELECT id, row_number() OVER (ORDER BY dist) AS r
  FROM (SELECT id, embedding <=> $1 AS dist
        FROM chunks
        WHERE superseded_by IS NULL
        ORDER BY embedding <=> $1
        LIMIT 50) t
),
kw AS (
  SELECT id, row_number() OVER (ORDER BY score DESC) AS r
  FROM (SELECT id, ts_rank_cd(tsv, websearch_to_tsquery('english', $2)) AS score
        FROM chunks
        WHERE tsv @@ websearch_to_tsquery('english', $2)
          AND superseded_by IS NULL
        ORDER BY score DESC
        LIMIT 50) t
)
SELECT id, SUM(1.0 / (60 + r)) AS rrf
FROM (SELECT * FROM vec UNION ALL SELECT * FROM kw) s
GROUP BY id
ORDER BY rrf DESC
LIMIT 40;
```

Graph expansion with a recursive query (2 hops, no cycles):

```sql
WITH RECURSIVE walk(entity_id, depth, path) AS (
  SELECT id, 0, ARRAY[id] FROM entities WHERE id = ANY($1::uuid[])
  UNION ALL
  SELECT nxt, w.depth + 1, w.path || nxt
  FROM walk w
  JOIN edges e ON w.entity_id IN (e.src, e.dst)
  CROSS JOIN LATERAL (SELECT CASE WHEN e.src = w.entity_id THEN e.dst ELSE e.src END AS nxt) n
  WHERE w.depth < 2 AND NOT nxt = ANY(w.path)
)
SELECT DISTINCT ON (ce.chunk_id) ce.chunk_id, w.depth
FROM walk w
JOIN chunk_entities ce ON ce.entity_id = w.entity_id
ORDER BY ce.chunk_id, w.depth
LIMIT 30;
```

**Some questions aren't retrieval at all.** "Which skills show up in roles where I was rejected?" is an aggregation, so give the agent a separate structured-query tool:

```sql
SELECT s.name, count(*) AS roles
FROM applications a
JOIN edges e    ON e.src = a.role_entity_id AND e.rel = 'REQUIRES'
JOIN entities s ON s.id = e.dst
WHERE a.status = 'rejected'
GROUP BY s.name
ORDER BY roles DESC;
```

## Memory lifecycle

- **Dedup:** exact duplicates are stopped by `content_hash`; near duplicates are marked `superseded_by` (ingestion step 4).
- **Consolidation:** a nightly job groups chunks about the same entity and has Haiku write one summary chunk (`source = 'summary'`) linked to its sources.
- **Updates and conflicts:** when a newer fact contradicts an older one, keep both with timestamps and prefer the newest. Add `valid_from` / `valid_to` columns to `edges`.
- **Decay:** never delete for age; rank lower instead. Final score = relevance × (0.5 + 0.5 × recency), with recency = exp(−age_days / 90), nudged up by `importance` and recent access.
- **Forgetting:** deleting a document cascades to its chunks and links; a cleanup job removes entities and edges left with no evidence.

## Security and observability

**Security**

- Row-level security on every table, enforced by a non-owner database role. A pytest proves user A can never read user B's rows.
- Bearer-token auth on the HTTP API; secrets from environment variables, never committed.
- Treat ingested text as untrusted (prompt injection): extraction prompts frame content as data, and retrieved text can never trigger tool calls on its own.
- Stretch: redact phone numbers and emails before text is sent to the LLM.

**Observability**

- One OpenTelemetry trace per query, with a span per stage: embed, vector, keyword, graph, re-rank, LLM.
- Metrics: p50/p95 latency per stage, candidates surviving each stage, tokens and cost per query, queue depth, job failures, ingestion throughput.
- A Grafana dashboard committed to the repo as JSON.
- Structured JSON logs carrying the trace id.

## Evaluation

Every resume number comes from this table, and `make eval` must reproduce it.

- **Corpus:** 300–500 documents. Keep real emails and posts local; publish a synthetic or anonymized corpus in the repo.
- **Test set:** 50 hand-checked questions, each with its gold chunks and a reference answer. Mix: 20 single-fact lookups, 15 filter + semantic, 15 multi-hop or aggregation. Claude can draft them; a human verifies every one.
- **Metrics:** recall@10, MRR and nDCG@10 for retrieval; answer correctness via LLM-as-judge, spot-checked by hand; p95 latency; cost per query.
- **Break results down by question type.**

| Configuration | Recall@10 | MRR | p95 latency (ms) | Cost per query |
| --- | --- | --- | --- | --- |
| Vector only | | | | |
| Keyword only | | | | |
| Hybrid (RRF) | | | | |
| Hybrid + re-rank | | | | |
| Hybrid + re-rank + graph | | | | |
| Full (+ MMR, decay) | | | | |

## Build plan

**Phase 1 — foundation and hybrid search:** repo with uv; Docker Compose with Postgres 16 + pgvector; schema as plain SQL migrations; ingest CLI (normalize, chunk, embed, insert, exact dedup); search endpoint (vector, keyword, RRF hybrid); 50-question test set and eval script; first ablation rows.

**Phase 2 — graph and re-ranking:** entity/relation extraction with Haiku + Pydantic; entity resolution and edge upserts; graph expansion, cross-encoder re-rank, MMR; structured-query tool; finish ablation table; README v1.

**Phase 3 — production polish:** Postgres job queue with `SKIP LOCKED`; RLS, isolation test, API auth; OpenTelemetry, Prometheus, Grafana; near-dup, consolidation and decay jobs; plug Recall into the job agent; publish the write-up.

**Phase 4 — stretch:** Go ingestion worker; `pg_search` BM25; public demo.

## Repo layout

```text
recall/
├── docker-compose.yml
├── Makefile                  # make up | ingest | eval | test
├── pyproject.toml
├── migrations/               # 001_schema.sql, 002_rls.sql, ...
├── recall/
│   ├── api/                  # FastAPI: /ingest, /search, /answer, /query
│   ├── ingest/               # normalize, chunk, embed, dedup
│   ├── graph/                # extraction, entity resolution, edges, expansion
│   ├── retrieval/            # vector, keyword, rrf, rerank, mmr, pipeline
│   ├── memory/               # consolidation, decay, forgetting
│   ├── worker/               # SKIP LOCKED job runner
│   ├── llm.py
│   ├── db.py
│   └── telemetry.py
├── eval/
│   ├── corpus/
│   ├── questions.jsonl
│   └── run.py
├── dashboards/grafana.json
├── tests/
└── docs/
```
