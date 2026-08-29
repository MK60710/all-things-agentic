"""FastAPI entrypoint. HTTP-serving layer over agent/ - imports from agent/,
never the reverse. Run locally with:

    GOOGLE_CLOUD_PROJECT=<project> uv run uvicorn service.app:app --reload

See the Dockerfile for the container entrypoint used on Cloud Run.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from service.routers import chat, clarifications, contradictions, feynman, gaps, health, papers, query, sessions, usage
from service.config import validate_production_environment
from service.state import build_state

logger = logging.getLogger("atlas.request")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Blocking - constructs the real GraphManager/ChunkIndex/QueryAgent/etc
    # exactly once at container startup. See service/state.py's docstring
    # for why this must never run more than once per process.
    validate_production_environment()
    app.state.app_state = build_state()
    yield


app = FastAPI(title="all-things-agentic API", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Attach a correlation ID and log every request's outcome and latency.

    The ID is also returned to the browser so a user can report one value
    and operators can find the matching backend logs without logging request
    bodies, prompts, PDFs, or explanations.
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%d",
            request_id,
            request.method,
            request.url.path,
            round((time.monotonic() - started) * 1000),
        )
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%d duration_ms=%d",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        round((time.monotonic() - started) * 1000),
    )
    return response

origins = [
    value.strip()
    for value in os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Upload-Token", "Authorization"],
)

for router in (health.router, usage.router, chat.router, query.router, clarifications.router, gaps.router, contradictions.router, feynman.router, papers.router, sessions.router):
    app.include_router(router)
