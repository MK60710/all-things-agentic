from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from agent.session_membership import session_ids as _session_ids
from service.deps import get_state, require_api_key
from service.schemas import AnswerClarificationRequest, PendingQuestionOut
from service.state import AppState

router = APIRouter(
    prefix="/clarifications", tags=["clarifications"], dependencies=[Depends(require_api_key)]
)


@router.get("", response_model=list[PendingQuestionOut])
def list_pending(
    session_id: str | None = Query(default=None),
    state: AppState = Depends(get_state),
) -> list[PendingQuestionOut]:
    questions = state.clarification.pending()
    if session_id is not None:
        # Only entity_merge questions reference an ingest-created node
        # (provisional_node_id) - a query_disambiguation question has no
        # such field and is meant to be resolved within the same chat
        # exchange it came from, so it's never filtered out here.
        questions = [
            q
            for q in questions
            if getattr(q, "provisional_node_id", None) is None
            or session_id
            in _session_ids(state.graph.graph.nodes.get(q.provisional_node_id, {}))
        ]
    return [PendingQuestionOut.from_domain(q) for q in questions]


@router.get("/{question_id}", response_model=PendingQuestionOut)
def get_question(question_id: str, state: AppState = Depends(get_state)) -> PendingQuestionOut:
    question = state.clarification.get(question_id)
    if question is None:
        raise HTTPException(status_code=404, detail=f"no question {question_id!r}")
    return PendingQuestionOut.from_domain(question)


@router.post("/{question_id}/answer", response_model=PendingQuestionOut)
def answer_question(
    question_id: str,
    body: AnswerClarificationRequest,
    state: AppState = Depends(get_state),
) -> PendingQuestionOut:
    try:
        question = state.clarification.answer(question_id, body.option_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PendingQuestionOut.from_domain(question)
