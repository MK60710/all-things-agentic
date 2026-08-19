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
    if body.paper_id:
        return state.query_agent.answer(body.message, paper_ids={body.paper_id})
    try:
        answer = state.general_chat.answer(
            body.message,
            [ChatTurn(role=item.role, text=item.text) for item in body.history],
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QueryResult(answer=answer, retrieval_mode="general")
