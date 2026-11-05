"""Ingest pipeline: load -> normalize -> dedupe -> chunk -> embed -> store."""

import logging
from pathlib import Path
from uuid import UUID

import psycopg

from recall.ingest.chunk import chunk_document
from recall.ingest.embed import Embedder
from recall.ingest.loader import load_file
from recall.ingest.normalize import content_hash, normalize
from recall.models import FailedFile, IngestReport
from recall.store import document_exists, insert_document_with_chunks

log = logging.getLogger(__name__)


def ingest_paths(
    conn: psycopg.Connection,
    paths: list[Path],
    *,
    user_id: UUID,
    embedder: Embedder,
    root: Path | None = None,
) -> IngestReport:
    """Ingest files idempotently; a failing file is reported and never aborts the batch."""
    report = IngestReport()
    for path in paths:
        try:
            doc = load_file(path, root)
            text = normalize(doc.body)
            if not text:
                report.empty += 1
                continue
            digest = content_hash(text)
            if document_exists(conn, user_id=user_id, content_hash=digest):
                report.duplicates += 1
                continue
            chunks = chunk_document(text, source=doc.source, title=doc.title, company=doc.company)
            if not chunks:
                report.empty += 1
                continue
            embeddings = embedder.embed([c.text for c in chunks])
            doc_id = insert_document_with_chunks(
                conn,
                user_id=user_id,
                doc=doc,
                content_hash=digest,
                chunks=chunks,
                embeddings=embeddings,
            )
            if doc_id is None:  # lost a race with a concurrent ingest
                report.duplicates += 1
                continue
            report.inserted += 1
            report.chunks += len(chunks)
        except Exception as exc:
            log.exception("failed to ingest %s", path)
            report.failed.append(FailedFile(path=str(path), error=str(exc)))
    return report
