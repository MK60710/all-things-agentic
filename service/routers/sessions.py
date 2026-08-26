from __future__ import annotations

import uuid
from collections import defaultdict
from itertools import combinations
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response

from agent.bibliography import build_bibtex
from service.deps import get_state, require_api_key
from service.schemas import (
    SessionCreateRequest,
    SessionGraphEdge,
    SessionGraphNode,
    SessionGraphResponse,
    SessionMetadata,
    PaperConnection,
    PaperConnectionEvidence,
    SessionPaperMapResponse,
)
from service.state import AppState
from service.storage import _paper_session_ids

router = APIRouter(prefix="/sessions", tags=["sessions"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=SessionMetadata)
def create_session(body: SessionCreateRequest, state: AppState = Depends(get_state)) -> SessionMetadata:
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    saved = state.session_store.save(
        session_id, name=body.name, created_at=created_at, goal=body.goal
    )
    return SessionMetadata.model_validate(saved)


@router.get("", response_model=list[SessionMetadata])
def list_sessions(state: AppState = Depends(get_state)) -> list[SessionMetadata]:
    return [SessionMetadata.model_validate(item) for item in state.session_store.list()]


@router.patch("/{session_id}", response_model=SessionMetadata)
def rename_session(
    session_id: str, body: SessionCreateRequest, state: AppState = Depends(get_state)
) -> SessionMetadata:
    existing = next(
        (s for s in state.session_store.list() if s.get("id") == session_id), None
    )
    if existing is None:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    # SessionStore.save's returned dict is only {id, updated_at, **values} -
    # not a Firestore read-back - so created_at (and goal, which this
    # endpoint never changes) must be carried through explicitly or the
    # response would fail SessionMetadata validation / silently clear the
    # goal, even though the merge=True write itself only touches
    # name/updated_at server-side.
    saved = state.session_store.save(
        session_id,
        name=body.name,
        created_at=existing.get("created_at"),
        goal=existing.get("goal"),
    )
    return SessionMetadata.model_validate(saved)


@router.get("/{session_id}/graph", response_model=SessionGraphResponse)
def get_session_graph(
    session_id: str, state: AppState = Depends(get_state)
) -> SessionGraphResponse:
    existing = next(
        (s for s in state.session_store.list() if s.get("id") == session_id), None
    )
    if existing is None:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    export = state.graph.export_session_graph(session_id)
    return SessionGraphResponse(
        nodes=[
            SessionGraphNode(
                **{**vars(n), "citations": [vars(c) for c in n.citations]}
            )
            for n in export.nodes
        ],
        edges=[SessionGraphEdge(**vars(e)) for e in export.edges],
    )


@router.get("/{session_id}/paper-map", response_model=SessionPaperMapResponse)
def get_session_paper_map(
    session_id: str, state: AppState = Depends(get_state)
) -> SessionPaperMapResponse:
    """Return an optional paper-level projection of the session graph.

    It is deliberately deterministic: papers connect only when their
    extracted graph evidence shares a canonical topic, and every summary
    points back to that topic's paper/section evidence.
    """
    existing = next((s for s in state.session_store.list() if s.get("id") == session_id), None)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    paper_titles = {
        p["id"]: p.get("title", p["id"])
        for p in state.paper_store.list()
        if session_id in _paper_session_ids(p)
    }
    export = state.graph.export_session_graph(session_id)
    pair_topics: defaultdict[tuple[str, str], list[tuple[str, dict[str, list[dict]]]]] = defaultdict(list)
    for node in export.nodes:
        if node.type == "PAPER":
            continue
        by_paper: defaultdict[str, list[dict]] = defaultdict(list)
        for citation in node.citations:
            if citation.paper_id in paper_titles:
                by_paper[citation.paper_id].append({"section": citation.section, "quote": citation.source_quote})
        for paper_a, paper_b in combinations(sorted(by_paper), 2):
            pair_topics[(paper_a, paper_b)].append((node.name, dict(by_paper)))

    connections = []
    for (paper_a, paper_b), topics in sorted(pair_topics.items()):
        topic_names = sorted({topic for topic, _ in topics})
        evidence = []
        for topic, citations_by_paper in topics[:6]:
            for paper_id in (paper_a, paper_b):
                citation = citations_by_paper[paper_id][0]
                evidence.append(PaperConnectionEvidence(topic=topic, paper_id=paper_id, section=citation["section"], quote=citation["quote"]))
        joined = ", ".join(topic_names[:3])
        if len(topic_names) > 3:
            joined += f" and {len(topic_names) - 3} more topics"
        connections.append(PaperConnection(
            paper_a_id=paper_a,
            paper_a_title=paper_titles[paper_a],
            paper_b_id=paper_b,
            paper_b_title=paper_titles[paper_b],
            summary=f"These papers are connected through shared extracted topics: {joined}.",
            shared_topics=topic_names,
            evidence=evidence,
        ))
    return SessionPaperMapResponse(session_id=session_id, connections=connections)


@router.get("/{session_id}/bibliography")
def get_session_bibliography(
    session_id: str, state: AppState = Depends(get_state)
) -> Response:
    existing = next(
        (s for s in state.session_store.list() if s.get("id") == session_id), None
    )
    if existing is None:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    papers = [
        p for p in state.paper_store.list() if session_id in _paper_session_ids(p)
    ]
    bibtex = build_bibtex(papers)
    return Response(content=bibtex, media_type="application/x-bibtex")


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str, state: AppState = Depends(get_state)) -> Response:
    """Cascade delete, scoped to what actually becomes ownerless: a
    paper (or graph node) still genuinely shared with another session
    survives this session's deletion, only this session's membership is
    removed from it - see PaperStore.detach_session and
    agent/graph_manager.py's remove_by_session, both of which already
    make this distinction rather than deleting unconditionally."""
    existing = next(
        (s for s in state.session_store.list() if s.get("id") == session_id), None
    )
    if existing is None:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")

    papers = [
        p for p in state.paper_store.list() if session_id in _paper_session_ids(p)
    ]
    for paper in papers:
        paper_id = paper["id"]
        updated = state.paper_store.detach_session(paper_id, session_id)
        if not updated.get("session_ids"):
            state.chunks.remove_paper(paper_id)
            state.paper_store.delete(paper_id)

    removed_node_ids = state.graph.remove_by_session(session_id)
    state.clarification.remove_for_node_ids(removed_node_ids)

    state.session_store.delete(session_id)
    return Response(status_code=204)
