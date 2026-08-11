"""Small write surface joining retrieval and optional graph ingestion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent.graph_manager import GraphIngestionReport, GraphManager
from agent.retrieval import ChunkIndex
from agent.schema import ExtractionResult, ExtractedEntity


@dataclass(frozen=True)
class ResearchIngestionReport:
    paper_id: str
    chunk_ids: list[str]
    graph: GraphIngestionReport | None


class ResearchStore:
    """Explicit CQRS write facade for downstream extraction code."""

    def __init__(
        self,
        chunk_index: ChunkIndex,
        graph_manager: GraphManager | None = None,
    ):
        self._chunks = chunk_index
        self._graph = graph_manager

    def ingest(
        self,
        extraction: ExtractionResult,
        *,
        paper_name: str | None = None,
        entity_embedding_fn: Callable[
            [ExtractedEntity], list[float] | None
        ] | None = None,
    ) -> ResearchIngestionReport:
        chunk_ids = self._chunks.upsert_paper(
            extraction.paper_id,
            extraction.chunk_metadata or extraction.chunks,
        )
        graph_report = None
        if self._graph is not None and (extraction.entities or extraction.relations):
            graph_report = self._graph.apply_extraction_result(
                extraction,
                paper_name=paper_name,
                embedding_fn=entity_embedding_fn,
            )
        return ResearchIngestionReport(
            paper_id=extraction.paper_id,
            chunk_ids=chunk_ids,
            graph=graph_report,
        )
