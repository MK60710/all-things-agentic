from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent.feynman_checker import FeynmanCheckResult, FeynmanPrompt
from service.deps import get_state, require_api_key
from service.state import AppState

router = APIRouter(prefix="/papers", tags=["feynman"], dependencies=[Depends(require_api_key)])

# A real "explain it in your own words" answer is a sentence to a couple of
# paragraphs - generous headroom over that, but nowhere near what's needed
# to tie up the request handler or pad a Gemini call. Confirmed live:
# with no cap at all, a 5MB explanation was accepted and took 17+ seconds
# before failing - free cost/DoS surface on a paid, per-call Gemini
# endpoint. Pydantic rejects an oversized body with a 422 before any of
# this ever reaches the graph lookup or the judge.
MAX_EXPLANATION_CHARS = 4000


class FeynmanCheckRequest(BaseModel):
    node_id: str
    session_id: str
    explanation: str = Field(max_length=MAX_EXPLANATION_CHARS)


@router.get("/{paper_id}/feynman/prompts", response_model=list[FeynmanPrompt])
def get_feynman_prompts(
    paper_id: str, session_id: str, state: AppState = Depends(get_state)
) -> list[FeynmanPrompt]:
    if state.paper_store.get(paper_id) is None:
        raise HTTPException(status_code=404, detail=f"no paper {paper_id!r}")
    return state.feynman_checker.pick_prompts(paper_id, session_id)


@router.post("/{paper_id}/feynman/check", response_model=FeynmanCheckResult)
def check_feynman_explanation(
    paper_id: str, body: FeynmanCheckRequest, state: AppState = Depends(get_state)
) -> FeynmanCheckResult:
    if state.paper_store.get(paper_id) is None:
        raise HTTPException(status_code=404, detail=f"no paper {paper_id!r}")
    result = state.feynman_checker.check(
        body.node_id, body.explanation, paper_id=paper_id, session_id=body.session_id
    )
    if result is None:
        raise HTTPException(status_code=502, detail="could not grade this explanation, try again")
    return result
