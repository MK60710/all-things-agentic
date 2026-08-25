from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


# Not "/healthz" - confirmed live on the real Cloud Run deployment that
# Google's own infrastructure intercepts that exact path before it ever
# reaches this app (a generic Google 404 page, not FastAPI's own 404 -
# "/" and every other route reach the app fine). Renamed to "/health" on
# the theory that "/healthz" specifically collides with a
# Kubernetes/GCP-convention reserved path; not yet re-verified against a
# real deployment that "/health" itself avoids the same interception.
@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
