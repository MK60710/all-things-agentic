"""Cheap, deterministic chunk retrieval for the primary read path.

The default embedder is local feature hashing: it has no model download, API
call, or per-token charge. A hosted embedding function can be injected later
without changing the index interface.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.schema import ExtractionChunk
from agent.text_utils import escape_tag_delimiters as _escape_tag_delimiters
from agent.text_utils import search_tokens as _search_tokens


EmbeddingFn = Callable[[str], list[float]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LocalHashingEmbedder:
    """Stateless lexical embeddings suitable for a free baseline."""

    def __init__(self, dimensions: int = 512):
        if dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        self.dimensions = dimensions

    def __call__(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            vector[index] += 1.0 if value & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class FastEmbedEmbedder:
    """Local semantic embeddings backed by an opt-in ONNX model download."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
    ):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ModuleNotFoundError(
                "install the 'local-semantic' project extra to use FastEmbed"
            ) from exc
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def __call__(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    paper_id: str
    ordinal: int
    text: str
    embedding: list[float] = field(repr=False)
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    source: str = "pdf_text"
    content_type: str = "prose"
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    paper_id: str
    text: str
    score: float
    ordinal: int
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class AssembledContext:
    hits: list[SearchHit]
    text: str


class ChunkIndex:
    """Retry-safe local index with optional Firestore-like persistence."""

    def __init__(
        self,
        embedding_fn: EmbeddingFn | None = None,
        db_client: Any | None = None,
    ):
        self._embedding_fn = embedding_fn or LocalHashingEmbedder()
        self._db = db_client
        self._records: dict[str, ChunkRecord] = {}

    def upsert_paper(
        self, paper_id: str, chunks: list[str] | list[ExtractionChunk]
    ) -> list[str]:
        prepared: list[tuple[ExtractionChunk, str, str]] = []
        for ordinal, value in enumerate(chunks):
            chunk = (
                value
                if isinstance(value, ExtractionChunk)
                else ExtractionChunk(text=value, ordinal=ordinal)
            )
            text = chunk.text
            normalized = re.sub(r"\s+", " ", text).strip()
            if not normalized:
                continue
            chunk_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"chunk:{paper_id}:{ordinal}:{normalized}",
                )
            )
            prepared.append((chunk, normalized, chunk_id))

        chunk_ids = [chunk_id for _, _, chunk_id in prepared]
        stale_ids = [
            chunk_id
            for chunk_id, record in self._records.items()
            if record.paper_id == paper_id and chunk_id not in chunk_ids
        ]
        for chunk_id in stale_ids:
            del self._records[chunk_id]
        if self._db is not None:
            persisted = self._db.collection("chunks")
            matching = persisted.where("paper_id", "==", paper_id)
            for snapshot in list(matching.stream()):
                if snapshot.id not in chunk_ids:
                    persisted.document(snapshot.id).delete()

        for index, (chunk, normalized, chunk_id) in enumerate(prepared):
            record = ChunkRecord(
                id=chunk_id,
                paper_id=paper_id,
                ordinal=chunk.ordinal,
                text=normalized,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                source=chunk.source,
                content_type=chunk.content_type,
                previous_chunk_id=chunk_ids[index - 1] if index > 0 else None,
                next_chunk_id=(
                    chunk_ids[index + 1] if index + 1 < len(chunk_ids) else None
                ),
                embedding=self._embedding_fn(normalized),
            )
            self._records[chunk_id] = record
            if self._db is not None:
                self._db.collection("chunks").document(chunk_id).set(
                    {
                        "paper_id": paper_id,
                        "ordinal": record.ordinal,
                        "text": normalized,
                        "page_start": record.page_start,
                        "page_end": record.page_end,
                        "section": record.section,
                        "source": record.source,
                        "content_type": record.content_type,
                        "previous_chunk_id": record.previous_chunk_id,
                        "next_chunk_id": record.next_chunk_id,
                        "embedding": record.embedding,
                    },
                    merge=True,
                )
        return chunk_ids

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        paper_ids: set[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        if limit < 1:
            return []
        query_embedding = self._embedding_fn(query)
        query_tokens = _search_tokens(query)
        hits = []
        for record in self._records.values():
            if paper_ids is not None and record.paper_id not in paper_ids:
                continue
            vector_score = max(
                0.0, _cosine_similarity(query_embedding, record.embedding)
            )
            record_tokens = _search_tokens(record.text)
            lexical_score = (
                len(query_tokens & record_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            score = 0.65 * lexical_score + 0.35 * vector_score
            if score >= min_score:
                hits.append(
                    SearchHit(
                        record.id,
                        record.paper_id,
                        record.text,
                        score,
                        record.ordinal,
                        record.page_start,
                        record.page_end,
                        record.section,
                    )
                )
        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return hits[:limit]

    def assemble_context(
        self,
        query: str,
        *,
        limit: int = 5,
        neighbor_window: int = 1,
        paper_ids: set[str] | None = None,
        max_characters: int = 14000,
    ) -> AssembledContext:
        """Retrieve, expand around hits, and restore paper reading order."""

        seeds = self.search(query, limit=limit, paper_ids=paper_ids)
        selected: dict[str, tuple[ChunkRecord, float]] = {}
        for seed in seeds:
            record = self._records[seed.chunk_id]
            selected[record.id] = (record, seed.score)
            for direction in ("previous_chunk_id", "next_chunk_id"):
                cursor = record
                for _ in range(max(0, neighbor_window)):
                    neighbor_id = getattr(cursor, direction)
                    if neighbor_id is None or neighbor_id not in self._records:
                        break
                    cursor = self._records[neighbor_id]
                    selected.setdefault(cursor.id, (cursor, seed.score))

        ordered = sorted(
            selected.values(), key=lambda item: (item[0].paper_id, item[0].ordinal)
        )
        hits: list[SearchHit] = []
        sections: list[str] = []
        used_characters = 0
        for record, score in ordered:
            page = (
                str(record.page_start)
                if record.page_start == record.page_end
                else f"{record.page_start}-{record.page_end}"
            )
            metadata: dict[str, str] = {
                "paper_id": _escape_tag_delimiters(record.paper_id)
            }
            if record.section:
                metadata["section"] = _escape_tag_delimiters(record.section)
            if record.page_start is not None:
                metadata["page"] = page
            rendered = (
                "<source_metadata>"
                + json.dumps(metadata, ensure_ascii=True, separators=(",", ":"))
                + "</source_metadata>\n"
                + _escape_tag_delimiters(record.text)
            )
            if sections and used_characters + len(rendered) > max_characters:
                break
            sections.append(rendered)
            used_characters += len(rendered)
            hits.append(
                SearchHit(
                    record.id,
                    record.paper_id,
                    record.text,
                    score,
                    record.ordinal,
                    record.page_start,
                    record.page_end,
                    record.section,
                )
            )
        return AssembledContext(hits=hits, text="\n\n".join(sections))

    def count(self) -> int:
        return len(self._records)
