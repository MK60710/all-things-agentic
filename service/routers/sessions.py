from __future__ import annotations

import json
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
    SessionMessagesRequest,
    SessionMessagesResponse,
    SessionMetadata,
)
from service.state import AppState
from service.storage import _paper_session_ids

router = APIRouter(prefix="/sessions", tags=["sessions"], dependencies=[Depends(require_api_key)])

# Headroom below Firestore's ~1MiB per-document cap - the JSON-text byte
# count computed below isn't a 1:1 match to Firestore's actual internal
# storage encoding, so this stays well clear of the real limit rather than
# cutting close to it.
_MAX_STORED_MESSAGES_BYTES = 800_000
# Not yet tuned against real usage - adjust after live testing.
_COMPACT_TAIL_MESSAGES = 3


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


@router.get("/{session_id}/messages", response_model=SessionMessagesResponse)
def get_session_messages(
    session_id: str, state: AppState = Depends(get_state)
) -> SessionMessagesResponse:
    # Deliberately no session-existence check, unlike every other route in
    # this file - a brand-new session's first switchToSession load has
    # nothing stored yet, which is a normal empty-list state, not a 404.
    # This store is intentionally decoupled from session metadata's
    # lifecycle (see SessionMessagesStore's own docstring).
    return SessionMessagesResponse(messages=state.session_messages_store.get(session_id))


@router.put("/{session_id}/messages", response_model=SessionMessagesResponse)
def save_session_messages(
    session_id: str, body: SessionMessagesRequest, state: AppState = Depends(get_state)
) -> SessionMessagesResponse:
    serialized_size = len(json.dumps(body.messages).encode("utf-8"))
    if serialized_size <= _MAX_STORED_MESSAGES_BYTES:
        state.session_messages_store.save(session_id, body.messages)
        return SessionMessagesResponse(messages=body.messages, compacted=False)

    summary = state.session_summarizer(body.messages)
    if summary is None:
        # Gemini unavailable - fail safe by storing as-is rather than
        # silently dropping history; the next PUT simply retries
        # compaction.
        state.session_messages_store.save(session_id, body.messages)
        return SessionMessagesResponse(messages=body.messages, compacted=False)

    notice_message = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "notice": True,
        "text": f"Session compacted to keep it fast. Here's what we covered: {summary}",
    }
    tail = body.messages[-_COMPACT_TAIL_MESSAGES:] if _COMPACT_TAIL_MESSAGES else []
    compacted_messages = [notice_message, *tail]
    try:
        # Archive before replacing - SessionMessagesRequest's own byte
        # ceiling keeps body.messages provably under Firestore's real
        # document cap (see its docstring), so this write shouldn't fail
        # on size, but a genuine Firestore outage is still a real,
        # distinct failure mode worth converting to a clean error instead
        # of an unhandled 500 - the Gemini summarization cost is already
        # spent by this point, so failing loudly here (rather than
        # silently losing the archive) is the safer default.
        state.session_messages_store.archive(session_id, body.messages)
        state.session_messages_store.save(session_id, compacted_messages)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SessionMessagesResponse(messages=compacted_messages, compacted=True)


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
    state.session_messages_store.delete(session_id)
    return Response(status_code=204)
