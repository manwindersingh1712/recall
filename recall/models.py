"""Pydantic models shared across Recall."""

from pathlib import Path
from typing import Any, Literal

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
