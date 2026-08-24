from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent.general_chat import ChatTurn
from agent.query_agent import QueryResult
from service.deps import get_state, require_api_key
from service.schemas import ChatRequest
from service.state import AppState

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=QueryResult)
def chat(body: ChatRequest, state: AppState = Depends(get_state)) -> QueryResult:
    history = [ChatTurn(role=item.role, text=item.text) for item in body.history]
    if body.paper_ids:
        return state.query_agent.answer(
            body.message,
            paper_ids=set(body.paper_ids),
            history=history,
            goal=body.goal,
            node_id=body.node_id,
        )
    # An empty working set doesn't mean "skip the graph" - it means search
    # the whole graph instead of a specific session's papers. Previously
    # this branch went straight to general_chat, so every unscoped question
    # (including gap-suggestion clicks, which are always about real graph
    # content) got an ungrounded answer instead of a real one. Only fall
    # back to plain chat when the graph genuinely has nothing relevant, not
    # just when no papers were attached.
    graph_result = state.query_agent.answer(
        body.message, paper_ids=None, history=history, goal=body.goal, node_id=body.node_id
    )
    if graph_result.retrieval_mode != "no_results":
        return graph_result
    try:
        answer = state.general_chat.answer(body.message, history)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QueryResult(answer=answer, retrieval_mode="general")
