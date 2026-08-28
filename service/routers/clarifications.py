from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from agent.session_membership import session_ids as _session_ids
from service.auth import get_current_user
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
    uid: str = Depends(get_current_user),
) -> list[PendingQuestionOut]:
    questions = state.clarification.pending()
    # Always filter by owner, session_id-or-not - an entity_merge question
    # references an ingest-created node (provisional_node_id), and that
    # node now carries owner_uid (see agent/graph_manager.py), so without
    # this an omitted session_id would surface every account's pending
    # questions. A query_disambiguation question has no provisional_node_id
    # and is meant to be resolved within the same chat exchange it came
    # from, so it's never filtered out by owner or session here. A
    # provisional_node_id that doesn't resolve to any real node at all is
    # also never filtered out by owner - genuinely nothing to leak, and
    # hiding it would just be confusing (the answer endpoint below still
    # handles a missing/invalid node correctly on its own terms).
    questions = [
        q
        for q in questions
        if getattr(q, "provisional_node_id", None) is None
        or q.provisional_node_id not in state.graph.graph.nodes
        or state.graph.graph.nodes[q.provisional_node_id].get("owner_uid") == uid
    ]
    if session_id is not None:
        questions = [
            q
            for q in questions
            if getattr(q, "provisional_node_id", None) is None
            or session_id
            in _session_ids(state.graph.graph.nodes.get(q.provisional_node_id, {}))
        ]
    return [PendingQuestionOut.from_domain(q) for q in questions]


@router.get("/{question_id}", response_model=PendingQuestionOut)
def get_question(
    question_id: str,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> PendingQuestionOut:
    question = state.clarification.get(question_id)
    if question is None:
        raise HTTPException(status_code=404, detail=f"no question {question_id!r}")
    node_id = getattr(question, "provisional_node_id", None)
    if (
        node_id is not None
        and node_id in state.graph.graph.nodes
        and state.graph.graph.nodes[node_id].get("owner_uid") != uid
    ):
        raise HTTPException(status_code=404, detail=f"no question {question_id!r}")
    return PendingQuestionOut.from_domain(question)


@router.post("/{question_id}/answer", response_model=PendingQuestionOut)
def answer_question(
    question_id: str,
    body: AnswerClarificationRequest,
    state: AppState = Depends(get_state),
    uid: str = Depends(get_current_user),
) -> PendingQuestionOut:
    existing = state.clarification.get(question_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"no question {question_id!r}")
    node_id = getattr(existing, "provisional_node_id", None)
    if (
        node_id is not None
        and node_id in state.graph.graph.nodes
        and state.graph.graph.nodes[node_id].get("owner_uid") != uid
    ):
        raise HTTPException(status_code=404, detail=f"no question {question_id!r}")
    try:
        question = state.clarification.answer(question_id, body.option_id, owner_uid=uid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PendingQuestionOut.from_domain(question)
