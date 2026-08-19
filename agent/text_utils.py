"""Shared text-processing helpers for lexical search, untrusted-content
prompt construction, and entity-embedding input, used by retrieval.py,
gemini_extractor.py, graph_manager.py, query_agent.py,
scripts/run_extraction_batch.py, and service/state.py.

Previously each of those files kept its own copy, and the copies had
already drifted (gemini_extractor.py and retrieval.py each had their own
escape helper; query_agent.py's stopword list included "can"/"did"/"who"
that retrieval.py's list was missing). One drifted copy is a bug waiting to
happen the next time the escaping or stopword rule changes and only some
copies get updated.
"""

from __future__ import annotations

import re

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did",
    "do", "does", "for", "from", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "what", "when",
    "where", "which", "with", "who", "why",
}


def search_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in STOP_WORDS}


def escape_tag_delimiters(value: str) -> str:
    """Escape the angle brackets that delimit prompt-context wrapper tags
    (<source_metadata>, <SOURCE>, <gap_candidate>, <RETRIEVED_RESEARCH>).
    json.dumps() only escapes JSON-syntax characters (quotes, backslashes)
    - it does nothing for '<'/'>', so untrusted text containing a literal
    closing tag could otherwise close the wrapper early and forge content
    that appears to be outside the untrusted-evidence boundary. Every
    untrusted value placed near a tag boundary needs this."""
    return value.replace("<", "&lt;").replace(">", "&gt;")


def entity_embedding_text(name: str, description: str, name_weight: int = 4) -> str:
    """Text to feed a bag-of-hashed-tokens embedder for entity
    canonicalization. Repeating the name biases the averaged vector toward
    the one signal that actually distinguishes entities - without this, a
    long generic description ("a large language model used in
    experiments") can dominate the vector and make differently-named
    entities in the same topical category (different metrics, different
    model names) collide. Observed live: "Claude" vs "GPT-4o" and "F1
    score" vs "Exact Match (EM) score" both false-positived into
    needs_clarification before this weighting was added."""
    return f"{' '.join([name] * name_weight)}: {description}"
