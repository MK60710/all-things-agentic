from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent.contradiction_finder import ContradictionCandidate
from service.deps import get_state, require_api_key
from service.state import AppState

router = APIRouter(tags=["contradictions"], dependencies=[Depends(require_api_key)])


@router.post(
    "/sessions/{session_id}/contradictions/check",
    response_model=list[ContradictionCandidate],
)
def check_contradictions(
    session_id: str, state: AppState = Depends(get_state)
) -> list[ContradictionCandidate]:
    existing = next(
        (s for s in state.session_store.list() if s.get("id") == session_id), None
    )
    if existing is None:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    return state.contradiction_finder.check_session(session_id)
