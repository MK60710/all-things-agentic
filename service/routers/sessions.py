from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from service.deps import get_state, require_api_key
from service.schemas import SessionCreateRequest, SessionMetadata
from service.state import AppState

router = APIRouter(prefix="/sessions", tags=["sessions"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=SessionMetadata)
def create_session(body: SessionCreateRequest, state: AppState = Depends(get_state)) -> SessionMetadata:
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    saved = state.session_store.save(session_id, name=body.name, created_at=created_at)
    return SessionMetadata.model_validate(saved)


@router.get("", response_model=list[SessionMetadata])
def list_sessions(state: AppState = Depends(get_state)) -> list[SessionMetadata]:
    return [SessionMetadata.model_validate(item) for item in state.session_store.list()]
