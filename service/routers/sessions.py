from __future__ import annotations

import uuid
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
