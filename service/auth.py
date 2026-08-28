"""Firebase Auth verification - a separate, orthogonal concern from
service/deps.py's require_api_key (a cost gate against the public URL,
not identity). Routes that need a real signed-in user depend on both.

Initialization is lazy, inside get_current_user itself, not at module
import time - every router module imports this file just to reference
get_current_user as a dependency, and initializing the Admin SDK eagerly
would mean every test run (which imports service.app -> every router)
needs live GCP credentials just to import, even though tests always
override this dependency via app.dependency_overrides and never call the
real function body. Same "construct the expensive client lazily, on
first real use" reasoning as agent/gemini_judge.py's LazyVertexClient.
"""

from __future__ import annotations

import threading

import firebase_admin
from fastapi import Header, HTTPException
from firebase_admin import auth as firebase_auth

_init_lock = threading.Lock()
_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        try:
            firebase_admin.initialize_app()
        except ValueError:
            # Already initialized elsewhere in this process (e.g. a
            # second AppState build in the same worker) - initialize_app
            # raises rather than being a no-op on a repeat call.
            pass
        _initialized = True


def get_current_user(authorization: str = Header(default="")) -> str:
    """Verifies an `Authorization: Bearer <Firebase ID token>` header and
    returns the caller's uid. 401 on anything missing, malformed, or
    invalid - there is no local-dev-without-auth fallback the way
    require_api_key has for its shared-secret cost gate, since there is
    no meaningful notion of "auth not configured" for identity."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    _ensure_initialized()
    try:
        decoded = firebase_auth.verify_id_token(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="invalid or expired ID token") from exc
    return decoded["uid"]
