from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from agent.query_agent import QueryResult
from service.deps import get_state, require_api_key
from service.schemas import FeedbackRequest, QueryRequest
from service.state import AppState

router = APIRouter(tags=["query"], dependencies=[Depends(require_api_key)])


@router.post("/query", response_model=QueryResult)
def answer_query(body: QueryRequest, state: AppState = Depends(get_state)) -> QueryResult:
    paper_ids = {body.paper_id} if body.paper_id else None
    return state.query_agent.answer(body.query, paper_ids=paper_ids)


@router.post("/query/feedback", status_code=204)
def query_feedback(body: FeedbackRequest, state: AppState = Depends(get_state)) -> Response:
    state.query_agent.record_feedback(body.node_id, helpful=body.helpful)
    return Response(status_code=204)
