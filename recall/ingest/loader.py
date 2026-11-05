"""Discover and load documents (with optional YAML front matter) from disk."""

import json
from pathlib import Path
from typing import Any

import frontmatter
import yaml
from pydantic import ValidationError

from recall.models import RawDocument

SUFFIXES = {".md", ".txt", ".html", ".eml"}
_KNOWN_KEYS = ("source", "title", "company", "source_ref")


def discover(root: Path) -> list[Path]:
    """Return supported files under root, recursively, sorted, skipping dotfiles."""
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUFFIXES and not p.name.startswith(".")
    )


def load_file(path: Path, root: Path | None = None) -> RawDocument:
    """Load one file into a RawDocument; raise ValueError naming the file if it is invalid."""
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    try:
        post = frontmatter.loads(text)
        fields: dict[str, Any] = {k: post.metadata[k] for k in _KNOWN_KEYS if k in post.metadata}
        extra = {k: v for k, v in post.metadata.items() if k not in _KNOWN_KEYS}
        metadata = json.loads(json.dumps(extra, default=str))
        fields.setdefault("title", path.stem)
        fields.setdefault(
            "source_ref", str(path.relative_to(root)) if root is not None else path.name
        )
        return RawDocument(path=path, body=post.content, metadata=metadata, **fields)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
