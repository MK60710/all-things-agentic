from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from service.deps import get_state, require_api_key
from service.schemas import AnswerClarificationRequest, PendingQuestionOut
from service.state import AppState

router = APIRouter(
    prefix="/clarifications", tags=["clarifications"], dependencies=[Depends(require_api_key)]
)


@router.get("", response_model=list[PendingQuestionOut])
def list_pending(state: AppState = Depends(get_state)) -> list[PendingQuestionOut]:
    return [PendingQuestionOut.from_domain(q) for q in state.clarification.pending()]


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
