"""App-wide singleton wiring, built once at container startup.

GraphManager, ChunkIndex, and ClarificationOrchestrator rehydrate their
Firestore-backed state at process startup. AppState still owns one shared
in-process view per worker, so it is constructed once by FastAPI's lifespan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from google import genai
from google.cloud import firestore

from agent.clarification_orchestrator import ClarificationOrchestrator
from agent.contradiction_finder import ContradictionFinder, GeminiContradictionJudge
from agent.document_ingestion import PdfTextExtractor
from agent.extraction_agent import ExtractionAgent
from agent.feynman_checker import FeynmanChecker, GeminiFeynmanJudge
from agent.gap_finder import GapFinder, GeminiExplainer
from agent.gemini_extractor import GeminiStructuredExtractor
from agent.general_chat import GeneralChatAgent
from agent.paper_guide import PaperGuideAgent
from agent.graph_manager import GraphManager
from agent.query_agent import QueryAgent
from agent.research_store import ResearchStore
from agent.retrieval import ChunkIndex, LocalHashingEmbedder
from agent.session_summarizer import GeminiSessionSummarizer, SummarizeFn
from agent.text_utils import entity_embedding_text
from service.storage import PaperStore, SessionMessagesStore, SessionStore, UploadTokenStore
from service.rate_limits import RateLimiter
from service.document_storage import CloudStorageDocumentStore, DocumentStore, LocalDocumentStore


@dataclass
class AppState:
    gemini_client: genai.Client
    graph: GraphManager
    chunks: ChunkIndex
    clarification: ClarificationOrchestrator
    query_agent: QueryAgent
    general_chat: GeneralChatAgent
    paper_guide: PaperGuideAgent
    gap_finder: GapFinder
    contradiction_finder: ContradictionFinder
    feynman_checker: FeynmanChecker
    extraction_agent: ExtractionAgent
    research_store: ResearchStore
    upload_root: str
    document_store: DocumentStore
    paper_store: PaperStore
    upload_tokens: UploadTokenStore
    session_store: SessionStore
    session_messages_store: SessionMessagesStore
    session_summarizer: SummarizeFn
    rate_limiter: RateLimiter
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
    gemini_client = genai.Client(vertexai=True, project=project, location=location)
    graph = GraphManager(project_id=project, db_client=db)
    chunks = ChunkIndex(db_client=db)
    clarification = ClarificationOrchestrator(graph_manager=graph, db_client=db)

    query_agent = QueryAgent(
        chunks,
        graph,
        project=project,
        location=location,
        clarification=clarification,
        db_client=db,
        client=gemini_client,
    )
    general_chat = GeneralChatAgent(
        project=project,
        location=location,
        model=os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"),
        client=gemini_client,
    )
    paper_guide = PaperGuideAgent(
        project=project,
        location=location,
        model=os.environ.get("GEMINI_GUIDE_MODEL", "gemini-3.5-flash-lite"),
        client=gemini_client,
    )
    gap_finder = GapFinder(
        graph,
        explain_fn=GeminiExplainer(client=gemini_client, project=project, location=location),
        db_client=db,
    )
    contradiction_finder = ContradictionFinder(
        graph,
        judge=GeminiContradictionJudge(client=gemini_client, project=project, location=location),
        db_client=db,
    )
    feynman_checker = FeynmanChecker(
        graph,
        judge=GeminiFeynmanJudge(client=gemini_client, project=project, location=location),
    )

    upload_root = os.environ.get("UPLOAD_ROOT", "/tmp/uploads")
    os.makedirs(upload_root, exist_ok=True)
    storage_bucket = os.environ.get("PAPER_STORAGE_BUCKET")
    document_store: DocumentStore = (
        CloudStorageDocumentStore(storage_bucket, project=project)
        if storage_bucket
        else LocalDocumentStore(upload_root)
    )
    extraction_agent = ExtractionAgent(
        document_extractor=PdfTextExtractor(allowed_root=upload_root),
        structured_extractor=GeminiStructuredExtractor(
            project=project, location=location, client=gemini_client
        ),
    )
    research_store = ResearchStore(chunks, graph)
    session_summarizer = GeminiSessionSummarizer(
        client=gemini_client, project=project, location=location
    )

    return AppState(
        gemini_client=gemini_client,
        graph=graph,
        chunks=chunks,
        clarification=clarification,
        query_agent=query_agent,
        general_chat=general_chat,
        paper_guide=paper_guide,
        gap_finder=gap_finder,
        contradiction_finder=contradiction_finder,
        feynman_checker=feynman_checker,
        extraction_agent=extraction_agent,
        research_store=research_store,
        upload_root=upload_root,
        document_store=document_store,
        paper_store=PaperStore(db),
        upload_tokens=UploadTokenStore(db),
        session_store=SessionStore(db),
        session_messages_store=SessionMessagesStore(db),
        session_summarizer=session_summarizer,
        rate_limiter=RateLimiter(db),
    )
