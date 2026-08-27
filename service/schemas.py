"""HTTP-only request/response models.

Everything the agents already return as Pydantic (QueryResult, GapCandidate,
ExtractionResult) is used directly as a FastAPI response_model - these are
only for request bodies FastAPI must parse, and for flattening the
non-Pydantic PendingQuestion dataclass union into JSON that doesn't require
a client to know about Python dataclasses.
"""

from __future__ import annotations

from typing import Any, Literal

import json

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    # Optional: when given, the router validates paper_ids against this
    # session's real paper membership server-side rather than trusting
    # the client array outright - every other session-scoped route
    # already resolves things from session_id server-side (gaps.py,
    # feynman.py, sessions.py); chat was the one place scoping was only
    # ever reconstructed and resent by the client with nothing to check
    # it against. Optional (not required) so unscoped/general chat and
    # any caller that genuinely has no session concept keep working
    # unchanged.
    session_id: str | None = Field(default=None, max_length=200)
    # Set when the frontend already knows exactly which node the question
    # is about - e.g. clicking a specific candidate on an ambiguous
    # result - so QueryAgent skips text search/ambiguity detection and
    # evaluates that node directly.
    node_id: str | None = None
    # Optional deep-dive scope. When present, chat uses only chunks whose
    # extracted section matches this value and skips graph-wide evidence.
    section: str | None = Field(default=None, max_length=500)


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
    session_ids: list[str] = Field(default_factory=list)


class DeepDiveSource(BaseModel):
    text: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None


class DeepDiveSection(BaseModel):
    section_id: str
    title: str
    plain_language: str
    key_points: list[str] = Field(default_factory=list)
    why_it_matters: str
    page_start: int | None = None
    page_end: int | None = None
    diagram: Any | None = None
    sources: list[DeepDiveSource] = Field(default_factory=list)


class DeepDiveResponse(BaseModel):
    paper_id: str
    title: str
    big_picture: str
    reading_time_minutes: int
    sections: list[DeepDiveSection] = Field(default_factory=list)


class PaperConnectionEvidence(BaseModel):
    topic: str
    paper_id: str
    section: str | None = None
    quote: str = ""


class PaperConnection(BaseModel):
    paper_a_id: str
    paper_a_title: str
    paper_b_id: str
    paper_b_title: str
    summary: str
    shared_topics: list[str] = Field(default_factory=list)
    evidence: list[PaperConnectionEvidence] = Field(default_factory=list)


class SessionPaperMapResponse(BaseModel):
    session_id: str
    connections: list[PaperConnection] = Field(default_factory=list)


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


# Every other request-body field in this file caps its size (query/text/
# message at max_length=8000, goal at 300, etc.) - a plain dict has no
# such per-field ceiling to lean on, so SessionMessagesRequest enforces
# its own total-size cap explicitly instead of silently being the one
# unbounded field in the whole schema.
#
# Kept just under Firestore's real ~1MiB per-document limit (not a much
# larger "reject clearly-abusive payloads" number) on purpose: over the
# 800KB compaction threshold (see save_session_messages), the ORIGINAL
# array gets archived verbatim before being replaced - if this ceiling
# were allowed to sit far above Firestore's real cap, an oversized-but
# -under-this-limit payload could pass validation, trigger compaction,
# and then have the archive write itself fail against Firestore's actual
# limit (a different, larger failure than the one this ceiling exists to
# prevent). Staying under Firestore's cap here means that can't happen -
# anything this validator accepts is provably small enough to archive.
_MAX_MESSAGES_REQUEST_BYTES = 950_000


class SessionMessagesRequest(BaseModel):
    # Deliberately list[dict], not a typed Message model - the frontend's
    # own Message shape (frontend/app/page.tsx) has ~19 fields covering
    # guide content, citations, clarification cards, comprehension-check
    # state, etc., and this store's whole job is to persist that rich
    # client shape opaquely, not re-validate/narrow it server-side.
    # max_length is a sanity ceiling on item count, not the real per-byte
    # guard - see the field_validator below for that.
    messages: list[dict] = Field(default_factory=list, max_length=2000)

    @field_validator("messages")
    @classmethod
    def messages_must_stay_under_size_ceiling(cls, value: list[dict]) -> list[dict]:
        size = len(json.dumps(value).encode("utf-8"))
        if size > _MAX_MESSAGES_REQUEST_BYTES:
            raise ValueError(
                f"messages payload is {size} bytes, over the "
                f"{_MAX_MESSAGES_REQUEST_BYTES} byte limit"
            )
        return value


class SessionMessagesResponse(BaseModel):
    messages: list[dict] = Field(default_factory=list)
    compacted: bool = False


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


class NodeCitationOut(BaseModel):
    paper_id: str
    section: str | None = None
    source_quote: str = ""


class SessionGraphNode(BaseModel):
    node_id: str
    name: str
    type: str
    description: str = ""
    citations: list[NodeCitationOut] = Field(default_factory=list)


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
