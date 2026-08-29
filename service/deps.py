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


def consume_rate_limit(state: AppState, uid: str, action: str) -> None:
    decision = state.rate_limiter.consume(uid, action)
    if decision.allowed:
        return
    messages = {
        "chat": "Your free chat limit has been reached. Try again later.",
        "paper_ingest": "Your free paper-processing limit has been reached. Try again later.",
        "guide": "Your free walkthrough limit has been reached. Try again later.",
        "contradictions": "Your free contradiction-checking limit has been reached. Try again later.",
        "feynman": "Your free Feynman-check limit has been reached. Try again later.",
        "gaps": "Your free research-gap limit has been reached. Try again later.",
    }
    raise HTTPException(
        status_code=429,
        detail=("Atlas is temporarily unavailable because the app's safety budget has been reached."
                 if decision.scope == "global"
                 else messages.get(action, f"Your free {action.replace('_', ' ')} limit has been reached. Try again later.")),
        headers={
            "Retry-After": str(decision.retry_after),
            "X-RateLimit-Action": action,
            "X-RateLimit-Reset": decision.reset_at or "",
            "X-RateLimit-Scope": decision.scope,
        },
    )
