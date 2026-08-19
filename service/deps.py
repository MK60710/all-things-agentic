"""FastAPI dependencies: reading shared app state, and a cost-protection
gate (not user auth - see service/state.py's docstring and the handoff
note for why no end-user login exists in this project)."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, Request

from service.state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """A no-op if API_SHARED_SECRET isn't set, so local dev doesn't need
    it. When set (as it will be on the deployed Cloud Run service), every
    request needs the matching header - a cost gate against an
    unauthenticated public URL calling paid Gemini/Firestore on every hit,
    not identity or session auth."""
    expected = os.environ.get("API_SHARED_SECRET")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
