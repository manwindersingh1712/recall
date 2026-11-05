"""Text normalization and content hashing for ingest."""

import hashlib
import re

from bs4 import BeautifulSoup, NavigableString

_HTML_TAG = re.compile(
    r"<(html|body|p|div|br|li|ul|ol|h[1-6]|span|table|tr|td|b|i|a|strong|em)\b",
    re.IGNORECASE,
)
_REPLY_HEADER = re.compile(r"^On .+ wrote:\s*$")
_SIGNATURE = re.compile(r"^-- ?$")
_BLOCKS = ["p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]


def _html_to_text(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Source whitespace (including newlines and NBSP) is never significant in HTML.
    for node in soup.find_all(string=True):
        if type(node) is NavigableString:
            node.replace_with(re.sub(r"\s+", " ", node))
    for tag in soup.find_all(_BLOCKS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    # Every block boundary is exactly one line break: drop blank lines.
    lines = (line.strip() for line in soup.get_text().splitlines())
    return "\n".join(line for line in lines if line)


def normalize(raw: str) -> str:
    """Return plain text with markup, quoted replies, signatures and extra whitespace removed."""
    text = _html_to_text(raw) if _HTML_TAG.search(raw) else raw
    lines: list[str] = []
    for line in text.replace("\xa0", " ").splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if _REPLY_HEADER.match(line) or _SIGNATURE.match(line):
            break
        if line.startswith(">"):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def content_hash(text: str) -> str:
    """Lowercase hex sha256 of the UTF-8 bytes of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
