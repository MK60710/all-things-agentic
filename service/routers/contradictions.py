from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent.contradiction_finder import ContradictionCandidate
from service.auth import get_current_user
from service.deps import get_state, require_api_key
from service.state import AppState

router = APIRouter(tags=["contradictions"], dependencies=[Depends(require_api_key)])


@router.post(
    "/sessions/{session_id}/contradictions/check",
    response_model=list[ContradictionCandidate],
)
def check_contradictions(
    session_id: str,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> list[ContradictionCandidate]:
    existing = state.session_store.get(session_id)
    if existing is None or existing.get("owner_uid") != uid:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    return state.contradiction_finder.check_session(session_id, owner_uid=uid)
