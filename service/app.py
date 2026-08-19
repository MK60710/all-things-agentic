"""FastAPI entrypoint. HTTP-serving layer over agent/ - imports from agent/,
never the reverse. Run locally with:

    GOOGLE_CLOUD_PROJECT=<project> uv run uvicorn service.app:app --reload

See the Dockerfile for the container entrypoint used on Cloud Run.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from service.routers import clarifications, gaps, health, papers, query
from service.state import build_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Blocking - constructs the real GraphManager/ChunkIndex/QueryAgent/etc
    # exactly once at container startup. See service/state.py's docstring
    # for why this must never run more than once per process.
    app.state.app_state = build_state()
    yield


app = FastAPI(title="all-things-agentic API", lifespan=lifespan)

for router in (health.router, query.router, clarifications.router, gaps.router, papers.router):
    app.include_router(router)
