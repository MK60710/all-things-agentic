from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from agent.gap_finder import GapCandidate
from agent.schema import NodeType
from service.deps import get_state, require_api_key
from service.schemas import GapFeedbackRequest
from service.state import AppState

router = APIRouter(prefix="/gaps", tags=["gaps"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=list[GapCandidate])
def list_gaps(
    node_type: NodeType | None = None,
    limit: int = 5,
    session_id: str | None = None,
    state: AppState = Depends(get_state),
) -> list[GapCandidate]:
    goal = None
    if session_id is not None:
        session = next(
            (s for s in state.session_store.list() if s.get("id") == session_id), None
        )
        goal = session.get("goal") if session else None
    return state.gap_finder.find_candidates(
        node_type=node_type, limit=limit, session_id=session_id, goal=goal
    )


@router.post("/feedback", status_code=204)
def gap_feedback(body: GapFeedbackRequest, state: AppState = Depends(get_state)) -> Response:
    state.gap_finder.record_feedback(
        body.node_a_id, body.node_b_id, interesting=body.interesting
    )
    return Response(status_code=204)
