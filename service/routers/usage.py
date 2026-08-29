from __future__ import annotations

from fastapi import APIRouter, Depends

from service.auth import get_current_user
from service.deps import get_state, require_api_key
from service.state import AppState

router = APIRouter(tags=["usage"], dependencies=[Depends(require_api_key)])


@router.get("/usage")
def usage_status(
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> dict:
    chat = state.rate_limiter.status(uid, "chat")
    return {
        "chat": {
            "allowed": chat.allowed,
            "remaining": chat.remaining,
            "retry_after": chat.retry_after,
            "reset_at": chat.reset_at,
            "scope": chat.scope,
        }
    }
