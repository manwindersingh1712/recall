"""Pydantic models shared across Recall."""

from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel

Source = Literal["job_post", "email", "note", "interview"]


class RawDocument(BaseModel):
    """A document as loaded from disk, before normalization."""

    path: Path
    source: Source = "note"
    title: str
    company: str | None = None
    source_ref: str
    body: str
    metadata: dict[str, Any] = {}


class Chunk(BaseModel):
    """A contextualized slice of a document."""

    ord: int
    text: str


class FailedFile(BaseModel):
    """A file that could not be ingested."""

    path: str
    error: str


class IngestReport(BaseModel):
    """Outcome counts for one ingest run."""

    inserted: int = 0
    duplicates: int = 0
    empty: int = 0
    chunks: int = 0
    failed: list[FailedFile] = []


Mode = Literal["vector", "keyword", "hybrid"]


class SearchHit(BaseModel):
    """One retrieved chunk with its document context."""

    chunk_id: UUID
    document_id: UUID
    title: str | None
    source_ref: str | None
    text: str
    score: float
