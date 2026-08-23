from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response

from service.deps import get_state, require_api_key
from service.schemas import (
    SessionCreateRequest,
    SessionGraphEdge,
    SessionGraphNode,
    SessionGraphResponse,
    SessionMetadata,
)
from service.state import AppState

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
        nodes=[SessionGraphNode(**vars(n)) for n in export.nodes],
        edges=[SessionGraphEdge(**vars(e)) for e in export.edges],
    )


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str, state: AppState = Depends(get_state)) -> Response:
    """Full cascade delete: the session's papers, graph nodes/edges,
    chunks and clarifications go with it, not just the session record -
    see agent/graph_manager.py's remove_by_session for why this can't be
    a Firestore-only delete while the server is running."""
    existing = next(
        (s for s in state.session_store.list() if s.get("id") == session_id), None
    )
    if existing is None:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")

    papers = [
        p for p in state.paper_store.list() if p.get("session_id") == session_id
    ]
    for paper in papers:
        paper_id = paper["id"]
        state.chunks.remove_paper(paper_id)
        state.paper_store.delete(paper_id)

    removed_node_ids = state.graph.remove_by_session(session_id)
    state.clarification.remove_for_node_ids(removed_node_ids)

    state.session_store.delete(session_id)
    return Response(status_code=204)
