from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from agent.query_agent import QueryResult
from service.auth import get_current_user
from service.deps import consume_rate_limit, get_state, require_api_key
from service.schemas import FeedbackRequest, QueryRequest
from service.state import AppState

router = APIRouter(tags=["query"], dependencies=[Depends(require_api_key)])


@router.post("/query", response_model=QueryResult)
def answer_query(
    body: QueryRequest,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> QueryResult:
    consume_rate_limit(state, uid, "chat")
    paper_ids = {body.paper_id} if body.paper_id else None
    return state.query_agent.answer(body.query, paper_ids=paper_ids, owner_uid=uid)


@router.post("/query/feedback", status_code=204)
def query_feedback(
    body: FeedbackRequest,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> Response:
    state.query_agent.record_feedback(body.node_id, helpful=body.helpful)
    return Response(status_code=204)
