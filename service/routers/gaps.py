from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from agent.gap_finder import GapCandidate
from agent.schema import NodeType
from service.auth import get_current_user
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
    uid: str = Depends(get_current_user),
) -> list[GapCandidate]:
    goal = None
    if session_id is not None:
        session = state.session_store.get(session_id)
        if session is None or session.get("owner_uid") != uid:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        goal = session.get("goal")
    # owner_uid is always passed, session_id-or-not - find_sparse_pairs
    # would otherwise scan the entire graph across every account when
    # session_id is omitted (see agent/graph_manager.py).
    return state.gap_finder.find_candidates(
        node_type=node_type, limit=limit, session_id=session_id, goal=goal, owner_uid=uid
    )


@router.post("/feedback", status_code=204)
def gap_feedback(
    body: GapFeedbackRequest,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> Response:
    state.gap_finder.record_feedback(
        body.node_a_id, body.node_b_id, interesting=body.interesting
    )
    return Response(status_code=204)
