from __future__ import annotations

import json
import uuid
from collections import defaultdict
from itertools import combinations
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response

from agent.bibliography import build_bibtex
from service.auth import get_current_user
from service.deps import get_state, require_api_key
from service.schemas import (
    SessionCreateRequest,
    SessionGraphEdge,
    SessionGraphNode,
    SessionGraphResponse,
    SessionMessagesRequest,
    SessionMessagesResponse,
    SessionMetadata,
    PaperConnection,
    PaperConnectionEvidence,
    SessionPaperMapResponse,
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


def _get_owned_session(state: AppState, session_id: str, uid: str) -> dict[str, Any]:
    """Every route below that takes a session_id resolves it through here,
    never a bare `next(... if s.get("id") == session_id)` lookup - a
    session that exists but belongs to a different uid returns the exact
    same 404 as one that doesn't exist at all, so a caller can't use this
    endpoint to probe which session ids are real for another account."""
    session = state.session_store.get(session_id)
    if session is None or session.get("owner_uid") != uid:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    return session


@router.post("", response_model=SessionMetadata)
def create_session(
    body: SessionCreateRequest,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> SessionMetadata:
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    saved = state.session_store.save(
        session_id, name=body.name, created_at=created_at, goal=body.goal, owner_uid=uid
    )
    return SessionMetadata.model_validate(saved)


@router.get("", response_model=list[SessionMetadata])
def list_sessions(
    state: AppState = Depends(get_state), uid: str = Depends(get_current_user)
) -> list[SessionMetadata]:
    return [
        SessionMetadata.model_validate(item)
        for item in state.session_store.list()
        if item.get("owner_uid") == uid
    ]


@router.patch("/{session_id}", response_model=SessionMetadata)
def rename_session(
    session_id: str,
    body: SessionCreateRequest,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> SessionMetadata:
    existing = _get_owned_session(state, session_id, uid)
    # SessionStore.save's returned dict is only {id, updated_at, **values} -
    # not a Firestore read-back - so created_at/goal/owner_uid (none of
    # which this endpoint changes) must be carried through explicitly or
    # the response would fail SessionMetadata validation / silently clear
    # them, even though the merge=True write itself only touches
    # name/updated_at server-side.
    saved = state.session_store.save(
        session_id,
        name=body.name,
        created_at=existing.get("created_at"),
        goal=existing.get("goal"),
        owner_uid=existing.get("owner_uid"),
    )
    return SessionMetadata.model_validate(saved)


@router.get("/{session_id}/graph", response_model=SessionGraphResponse)
def get_session_graph(
    session_id: str,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> SessionGraphResponse:
    _get_owned_session(state, session_id, uid)
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
    session_id: str,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> SessionPaperMapResponse:
    """Return an optional paper-level projection of the session graph.

    It is deliberately deterministic: papers connect only when their
    extracted graph evidence shares a canonical topic, and every summary
    points back to that topic's paper/section evidence.
    """
    _get_owned_session(state, session_id, uid)
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
    session_id: str,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> Response:
    _get_owned_session(state, session_id, uid)
    papers = [
        p for p in state.paper_store.list() if session_id in _paper_session_ids(p)
    ]
    bibtex = build_bibtex(papers)
    return Response(content=bibtex, media_type="application/x-bibtex")


@router.get("/{session_id}/messages", response_model=SessionMessagesResponse)
def get_session_messages(
    session_id: str,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> SessionMessagesResponse:
    # A session always exists (created via POST /sessions) before the
    # frontend ever fetches its messages, so requiring ownership here
    # doesn't conflict with the empty-list-not-404 behavior for a
    # brand-new session that just has nothing saved yet - see
    # SessionMessagesStore's own docstring for that part.
    _get_owned_session(state, session_id, uid)
    return SessionMessagesResponse(messages=state.session_messages_store.get(session_id))


@router.put("/{session_id}/messages", response_model=SessionMessagesResponse)
def save_session_messages(
    session_id: str,
    body: SessionMessagesRequest,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> SessionMessagesResponse:
    _get_owned_session(state, session_id, uid)
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
def delete_session(
    session_id: str,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> Response:
    """Cascade delete, scoped to what actually becomes ownerless: a
    paper (or graph node) still genuinely shared with another session
    survives this session's deletion, only this session's membership is
    removed from it - see PaperStore.detach_session and
    agent/graph_manager.py's remove_by_session, both of which already
    make this distinction rather than deleting unconditionally."""
    _get_owned_session(state, session_id, uid)

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
