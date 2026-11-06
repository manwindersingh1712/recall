"""Structure-aware, contextual chunking (sizes are whitespace-word counts)."""

import re
from collections.abc import Callable

from recall.models import Chunk, Source

_HEADING = re.compile(r"^#{1,6} ")
_SECTION_NAMES = re.compile(
    r"^(about|about the role|about us|responsibilities|what you'll do|requirements|"
    r"qualifications|nice to have|benefits|compensation):?$",
    re.IGNORECASE,
)


def word_count(text: str) -> int:
    return len(text.split())


def _starts_job_section(line: str) -> bool:
    stripped = line.strip()
    return bool(_HEADING.match(stripped) or _SECTION_NAMES.match(stripped))


def _split_lines(text: str, is_start: Callable[[str], bool]) -> list[str]:
    sections: list[list[str]] = [[]]
    for line in text.splitlines():
        if is_start(line) and sections[-1]:
            sections.append([])
        sections[-1].append(line)
    return ["\n".join(s).strip() for s in sections if "".join(s).strip()]


def _sections(text: str, source: Source) -> tuple[list[str], str]:
    """Return (sections, separator used when merging them)."""
    if source == "job_post":
        return _split_lines(text, _starts_job_section), "\n"
    if source == "email":
        return _split_lines(text, lambda ln: ln.startswith("From:")), "\n"
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()], "\n\n"


def _windows(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = size - overlap
    out: list[str] = []
    start = 0
    while True:
        end = start + size
        out.append(" ".join(words[start:end]))
        if end >= len(words):
            return out
        start += step


def chunk_document(
    text: str,
    *,
    source: Source,
    title: str | None,
    company: str | None,
    target_words: int = 300,
    max_words: int = 380,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Split text into contextual chunks; every chunk is prefixed with title and company."""
    if not text.strip():
        return []
    sections, sep = _sections(text, source)
    bodies: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            bodies.append(sep.join(current))
        current, current_words = [], 0

    for section in sections:
        n = word_count(section)
        if n > max_words:
            flush()
            bodies.extend(_windows(section, target_words, overlap_words))
        elif current and current_words + n > target_words:
            flush()
            current, current_words = [section], n
        else:
            current.append(section)
            current_words += n
    flush()

    if title and company:
        prefix = f"{title} — {company}\n\n"
    elif title:
        prefix = f"{title}\n\n"
    else:
        prefix = ""
    return [Chunk(ord=i, text=prefix + body) for i, body in enumerate(bodies)]
