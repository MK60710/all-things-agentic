"""HTTP-only request/response models.

Everything the agents already return as Pydantic (QueryResult, GapCandidate,
ExtractionResult) is used directly as a FastAPI response_model - these are
only for request bodies FastAPI must parse, and for flattening the
non-Pydantic PendingQuestion dataclass union into JSON that doesn't require
a client to know about Python dataclasses.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.clarification_orchestrator import (
    EntityMergeQuestion,
    PendingQuestion,
    QueryDisambiguationQuestion,
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    paper_id: str | None = None


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[ChatHistoryItem] = Field(default_factory=list, max_length=20)
    # A session's working set, not a single paper - QueryAgent.answer()
    # already accepts paper_ids: set[str] | None as a multi-paper filter,
    # this was only ever exposed here as if it took just one.
    paper_ids: list[str] | None = None
    # The current session's stated goal (SessionMetadata.goal), passed
    # through on every turn rather than looked up server-side from
    # session_id - the frontend already holds it in currentSession state,
    # and /chat has no session_id field to look it up by.
    goal: str | None = Field(default=None, max_length=300)
    # Set when the frontend already knows exactly which node the question
    # is about - e.g. clicking a specific candidate on an ambiguous
    # result - so QueryAgent skips text search/ambiguity detection and
    # evaluates that node directly.
    node_id: str | None = None


class UploadTokenResponse(BaseModel):
    token: str
    expires_at: str
    max_bytes: int


class ArxivIngestRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    arxiv_id: str = Field(min_length=5, max_length=40)
    title: str = Field(min_length=1, max_length=500)
    authors: str | None = Field(default=None, max_length=2000)
    abstract: str | None = Field(default=None, max_length=10000)
    pdf_url: str | None = Field(default=None, alias="pdfUrl")
    session_id: str | None = Field(default=None, max_length=200)


class PaperMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str
    authors: str | None = None
    abstract: str | None = None
    pdf_url: str | None = Field(default=None, alias="pdfUrl")
    status: str
    session_id: str | None = None


class SessionCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    # Only meaningful on creation - the rename endpoint reuses this same
    # request model but only ever changes name, carrying the existing goal
    # through unchanged (see rename_session in routers/sessions.py).
    goal: str | None = Field(default=None, max_length=300)


class SessionMetadata(BaseModel):
    id: str
    name: str
    created_at: str
    goal: str | None = None


class FeedbackRequest(BaseModel):
    node_id: str
    helpful: bool


class GapFeedbackRequest(BaseModel):
    node_a_id: str
    node_b_id: str
    interesting: bool


class AnswerClarificationRequest(BaseModel):
    option_id: str


class ClarificationOptionOut(BaseModel):
    id: str
    label: str
    description: str = ""


class PendingQuestionOut(BaseModel):
    """Flat, kind-discriminated serialization of EntityMergeQuestion |
    QueryDisambiguationQuestion. Kind-specific fields are optional here and
    only populated when they apply to that question's kind - the same
    shape a client would need to branch on `kind` to interpret regardless
    of whether these were required or optional at the HTTP layer."""

    id: str
    kind: str
    question: str
    options: list[ClarificationOptionOut]
    status: str
    answer_option_id: str | None = None
    provisional_node_id: str | None = None
    candidate_node_id: str | None = None
    score: float | None = None
    query_text: str | None = None

    @classmethod
    def from_domain(cls, q: PendingQuestion) -> "PendingQuestionOut":
        base = dict(
            id=q.id,
            kind=q.kind,
            question=q.question,
            status=q.status,
            answer_option_id=q.answer_option_id,
            options=[
                ClarificationOptionOut(
                    id=opt.id, label=opt.label, description=opt.description
                )
                for opt in q.options
            ],
        )
        if isinstance(q, EntityMergeQuestion):
            base.update(
                provisional_node_id=q.provisional_node_id,
                candidate_node_id=q.candidate_node_id,
                score=q.score,
            )
        else:
            assert isinstance(q, QueryDisambiguationQuestion)
            base.update(query_text=q.query_text)
        return cls(**base)


class GraphVizNode(BaseModel):
    node_id: str
    name: str
    type: str | None = None
    reused_existing_node: bool = False


class GraphVizEdge(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    relation: str


class SessionGraphNode(BaseModel):
    node_id: str
    name: str
    type: str
    description: str = ""


class SessionGraphEdge(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    relation: str


class SessionGraphResponse(BaseModel):
    nodes: list[SessionGraphNode]
    edges: list[SessionGraphEdge]


class PaperIngestResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    paper_id: str
    id: str
    title: str
    authors: str | None = None
    abstract: str | None = None
    pdf_url: str | None = Field(default=None, alias="pdfUrl")
    status: str = "ready"
    extraction_ok: bool
    issue_message: str | None = None
    chunk_ids: list[str]
    entities_added: int
    relations_added: int
    # The exact post-canonicalization node/edge writes from this ingest -
    # GraphManager already computes these (GraphIngestionReport.node_writes/
    # edge_writes), just never returned over HTTP. Powers the frontend's
    # live graph-building animation with real data, not synthesized counts.
    new_nodes: list[GraphVizNode] = Field(default_factory=list)
    new_edges: list[GraphVizEdge] = Field(default_factory=list)
    pending_clarification_count: int
