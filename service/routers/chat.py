from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent.general_chat import ChatTurn
from agent.query_agent import QueryResult
from service.auth import get_current_user
from service.deps import consume_rate_limit, get_state, require_api_key
from service.schemas import ChatRequest
from service.state import AppState
from service.storage import _paper_session_ids

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=QueryResult)
def chat(
    body: ChatRequest,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> QueryResult:
    if body.session_id:
        owned = state.session_store.get(body.session_id)
        if owned is None or owned.get("owner_uid") != uid:
            raise HTTPException(status_code=404, detail=f"no session {body.session_id!r}")
    consume_rate_limit(state, uid, "chat")
    history = [ChatTurn(role=item.role, text=item.text) for item in body.history]
    paper_ids = body.paper_ids
    if body.session_id and paper_ids:
        # Validate the client-supplied working set against this
        # session's real paper membership rather than trusting it
        # outright - if the client's papers state has ever drifted from
        # the server's real session membership (a stale fetch, a race
        # right after upload, a missed refetch after switching
        # sessions), a paper_id that isn't actually in this session
        # gets silently dropped here instead of chat quietly answering
        # against the wrong scope with nothing to catch it.
        allowed = {
            p["id"]
            for p in state.paper_store.list()
            if body.session_id in _paper_session_ids(p)
        }
        paper_ids = [pid for pid in paper_ids if pid in allowed]
    if paper_ids:
        return state.query_agent.answer(
            body.message,
            paper_ids=set(paper_ids),
            history=history,
            goal=body.goal,
            node_id=body.node_id,
            section=body.section,
            owner_uid=uid,
        )
    # An empty working set doesn't mean "skip the graph" - it means search
    # the whole graph instead of a specific session's papers. Previously
    # this branch went straight to general_chat, so every unscoped question
    # (including gap-suggestion clicks, which are always about real graph
    # content) got an ungrounded answer instead of a real one. Only fall
    # back to plain chat when the graph genuinely has nothing relevant, not
    # just when no papers were attached.
    graph_result = state.query_agent.answer(
        body.message,
        paper_ids=None,
        history=history,
        goal=body.goal,
        node_id=body.node_id,
        section=body.section,
        owner_uid=uid,
    )
    if graph_result.retrieval_mode != "no_results":
        return graph_result
    try:
        answer = state.general_chat.answer(body.message, history)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QueryResult(answer=answer, retrieval_mode="general")
