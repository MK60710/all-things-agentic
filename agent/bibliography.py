"""Session bibliography export: turns this session's papers into a real
BibTeX file, not just a list living inside the chat scroll.

Known, deliberate limitations (not bugs to fix later): the cite key is
the sanitized paper_id itself, not an author+year-derived key like
"hu2021lora" - deterministic and collision-free without parsing a first
author's last name out of a free-text authors string, which arXiv/upload
metadata doesn't reliably support. The `author` field is a best-effort
", " -> " and " join of that same free-text string, not a true
Last/First reparse - good enough for BibTeX to import cleanly, not a
guarantee every name splits perfectly.
"""

from __future__ import annotations

import re
from typing import Any

_ARXIV_PAPER_ID = re.compile(r"^arxiv-(\d{2})(\d{2})\.\d{4,5}(?:v\d+)?$")
_CITE_KEY_UNSAFE = re.compile(r"[^A-Za-z0-9]")


def _cite_key(paper_id: str) -> str:
    return _CITE_KEY_UNSAFE.sub("_", paper_id)


def _escape(value: str) -> str:
    # BibTeX field values are wrapped in {}; a stray brace in a title/
    # author string would unbalance the entry, so the only truly unsafe
    # characters here are the delimiters themselves.
    return value.replace("{", "").replace("}", "")


def _bibtex_entry(paper: dict[str, Any]) -> str:
    paper_id = paper.get("id") or ""
    key = _cite_key(paper_id)
    fields: list[tuple[str, str]] = []

    title = paper.get("title")
    if title:
        fields.append(("title", _escape(title)))

    authors = paper.get("authors")
    if authors:
        fields.append(("author", _escape(authors.replace(", ", " and "))))

    arxiv_match = _ARXIV_PAPER_ID.match(paper_id)
    arxiv_id = paper_id.removeprefix("arxiv-") if arxiv_match else None
    if arxiv_match:
        year = 2000 + int(arxiv_match.group(1))
        fields.append(("year", str(year)))

    abstract = paper.get("abstract")
    if abstract:
        fields.append(("abstract", _escape(abstract)))

    if arxiv_id:
        fields.append(("eprint", arxiv_id))
        fields.append(("archivePrefix", "arXiv"))

    pdf_url = paper.get("pdf_url")
    url = pdf_url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None)
    if url:
        fields.append(("url", url))

    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
    return f"@misc{{{key},\n{body}\n}}"


def build_bibtex(papers: list[dict[str, Any]]) -> str:
    if not papers:
        return "% No papers in this session yet.\n"
    return "\n\n".join(_bibtex_entry(paper) for paper in papers) + "\n"
