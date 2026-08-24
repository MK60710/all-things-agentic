"""FastAPI entrypoint. HTTP-serving layer over agent/ - imports from agent/,
never the reverse. Run locally with:

    GOOGLE_CLOUD_PROJECT=<project> uv run uvicorn service.app:app --reload

See the Dockerfile for the container entrypoint used on Cloud Run.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service.routers import chat, clarifications, contradictions, feynman, gaps, health, papers, query, sessions
from service.state import build_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Blocking - constructs the real GraphManager/ChunkIndex/QueryAgent/etc
    # exactly once at container startup. See service/state.py's docstring
    # for why this must never run more than once per process.
    app.state.app_state = build_state()
    yield


app = FastAPI(title="all-things-agentic API", lifespan=lifespan)

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
    allow_headers=["Content-Type", "X-API-Key", "X-Upload-Token"],
)

for router in (health.router, chat.router, query.router, clarifications.router, gaps.router, contradictions.router, feynman.router, papers.router, sessions.router):
    app.include_router(router)
