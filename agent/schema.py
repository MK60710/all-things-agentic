"""Typed ontology for the knowledge graph. Additive-only: new members can be
appended to NodeType/EdgeType, but existing members are never renamed or
removed once real data has been ingested against them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class NodeType(str, Enum):
    PAPER = "PAPER"
    CONCEPT = "CONCEPT"
    METHOD = "METHOD"
    MODEL = "MODEL"
    BENCHMARK_DATASET = "BENCHMARK_DATASET"
    METRIC = "METRIC"
    CLAIM = "CLAIM"


class EdgeType(str, Enum):
    PROPOSES = "PROPOSES"
    USES = "USES"
    EVALUATES_ON = "EVALUATES_ON"
    # UNDERPERFORMS is represented by reversing source/target on an
    # OUTPERFORMS edge rather than a separate enum member, so "B underperforms
    # A" is stored as "A OUTPERFORMS B" with source=A, target=B.
    OUTPERFORMS = "OUTPERFORMS"
    EXTENDS = "EXTENDS"
    # The merge/alias relation the Clarification Orchestrator writes to.
    SAME_AS = "SAME_AS"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"


class ProvenanceTag(str, Enum):
    # Directly stated in a paper, has a verified source quote.
    EXTRACTED = "EXTRACTED"
    # Derived, e.g. a user-confirmed SAME_AS merge — no single source quote.
    INFERRED = "INFERRED"


class Node(BaseModel):
    id: str
    type: NodeType
    name: str
    description: str = ""
    entity_embedding: list[float] | None = None


class Edge(BaseModel):
    id: str
    source_id: str
    target_id: str
    type: EdgeType
    provenance: ProvenanceTag
    source_paper_id: str | None = None
    source_section: str | None = None
    source_quote: str | None = None
    valid_from: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    valid_until: datetime | None = None


class ExtractedEntity(BaseModel):
    """One entity as returned by the Extraction Agent's structured output."""

    name: str
    type: NodeType
    description: str


class ExtractedRelation(BaseModel):
    """One relation as returned by the Extraction Agent's structured output.

    source_quote is required for EXTRACTED edges (it backs both the
    reasoning trace and the hallucination-verification check); INFERRED
    edges are never produced by the Extraction Agent directly, only by
    canonicalization merges downstream.
    """

    source_entity: str
    source_type: NodeType | None = None
    relation: EdgeType
    target_entity: str
    target_type: NodeType | None = None
    source_quote: str
    source_section: str | None = None

    @field_validator("source_quote")
    @classmethod
    def source_quote_must_contain_evidence(cls, value: str) -> str:
        quote = value.strip()
        if not quote:
            raise ValueError("extracted relations require a non-empty source quote")
        return quote


class ExtractionChunk(BaseModel):
    """A citation-ready chunk while retaining the legacy text-only field."""

    text: str
    ordinal: int
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    source: str = "pdf_text"
    content_type: str = "prose"


class ExtractionResult(BaseModel):
    """Full structured output for one paper, from the Extraction Agent."""

    paper_id: str
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    chunks: list[str] = Field(default_factory=list)
    chunk_metadata: list[ExtractionChunk] = Field(default_factory=list)
    # Count of source windows a structured extractor failed to extract
    # (truncated/invalid model output, API error, etc.) and skipped rather
    # than letting take down the whole paper's result. A default-0 field so
    # existing extractors (e.g. ChunkOnlyStructuredExtractor) that never
    # experience this failure mode don't need to know about it, but
    # ExtractionAgent checks it to avoid reporting a partial result as a
    # clean, issue-free success.
    skipped_windows: int = 0
