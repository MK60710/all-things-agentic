"""App-wide singleton wiring, built once at container startup.

ChunkIndex and ClarificationOrchestrator are plain in-process state with no
Firestore rehydration (unlike GraphManager, which does rehydrate on
construction) - if this were ever built more than once per running process,
or if Cloud Run ran more than one instance, requests could land on an
instance that never saw earlier ingests or pending questions. This is the
reason the deploy command in the handoff doc pins
--min-instances=1 --max-instances=1 and the Dockerfile pins --workers 1:
build_state() must run exactly once, and everything downstream depends on
that being true.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from google.cloud import firestore

from agent.clarification_orchestrator import ClarificationOrchestrator
from agent.document_ingestion import PdfTextExtractor
from agent.extraction_agent import ExtractionAgent
from agent.gap_finder import GapFinder, GeminiExplainer
from agent.gemini_extractor import GeminiStructuredExtractor
from agent.graph_manager import GraphManager
from agent.query_agent import QueryAgent
from agent.research_store import ResearchStore
from agent.retrieval import ChunkIndex, LocalHashingEmbedder
from agent.text_utils import entity_embedding_text


@dataclass
class AppState:
    graph: GraphManager
    chunks: ChunkIndex
    clarification: ClarificationOrchestrator
    query_agent: QueryAgent
    gap_finder: GapFinder
    extraction_agent: ExtractionAgent
    research_store: ResearchStore
    upload_root: str
    _embedder: LocalHashingEmbedder | None = None

    def __post_init__(self) -> None:
        if self._embedder is None:
            self._embedder = LocalHashingEmbedder()

    def entity_embedding_fn(self, entity) -> list[float]:
        return self._embedder(entity_embedding_text(entity.name, entity.description))


def build_state() -> AppState:
    # Fail loudly, not with a silent local fallback - a deployed service
    # with the wrong project would otherwise write real data to the wrong
    # place, or claim a real project id and quietly do nothing.
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

    db = firestore.Client(project=project)
    graph = GraphManager(project_id=project, db_client=db)
    chunks = ChunkIndex(db_client=db)
    clarification = ClarificationOrchestrator(graph_manager=graph)

    query_agent = QueryAgent(
        chunks,
        graph,
        project=project,
        location=location,
        clarification=clarification,
        db_client=db,
    )
    gap_finder = GapFinder(
        graph,
        explain_fn=GeminiExplainer(project=project, location=location),
        db_client=db,
    )

    upload_root = os.environ.get("UPLOAD_ROOT", "/tmp/uploads")
    os.makedirs(upload_root, exist_ok=True)
    extraction_agent = ExtractionAgent(
        document_extractor=PdfTextExtractor(allowed_root=upload_root),
        structured_extractor=GeminiStructuredExtractor(
            project=project, location=location
        ),
    )
    research_store = ResearchStore(chunks, graph)

    return AppState(
        graph=graph,
        chunks=chunks,
        clarification=clarification,
        query_agent=query_agent,
        gap_finder=gap_finder,
        extraction_agent=extraction_agent,
        research_store=research_store,
        upload_root=upload_root,
    )
