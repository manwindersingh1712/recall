# Recall

A memory engine for my job-finding agent: it turns messy inputs (job posts, recruiter emails, notes, interview feedback) into a store the agent can query for structured, connected context, built in Python on PostgreSQL.

## Quickstart

```
make up && make migrate && uv run recall ingest tests/fixtures/docs && uv run recall search "postgres remote"
make test
```

Note: (re)start the API after the first `make migrate`; the pgvector types are registered when a connection opens, so a server started before the extension exists will not see them.

## Eval results

Eval results: pending (Phase 1 corpus generation)
