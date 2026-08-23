from __future__ import annotations

import uuid

import pytest

from agent.clarification_orchestrator import (
    DISTINCT_OPTION_ID,
    ClarificationOrchestrator,
    EntityMergeQuestion,
    QueryDisambiguationQuestion,
)
from agent.graph_manager import GraphManager, NodeSearchHit
from agent.schema import Node, NodeType


def _make_graph(fake_db) -> GraphManager:
    return GraphManager(project_id="test", db_client=fake_db)


def _node(name: str) -> Node:
    return Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT, name=name)


def test_register_entity_merge_question_returns_the_discriminated_type(fake_db):
    """provisional_node_id/candidate_node_id are required, non-optional
    fields on EntityMergeQuestion specifically - not str | None on a
    dataclass shared with query_disambiguation, which would let a
    question ever exist with those unset even though answer() feeds them
    straight into GraphManager.resolve_alias's non-Optional str params."""
    orchestrator = ClarificationOrchestrator(graph_manager=_make_graph(fake_db))
    question = orchestrator.register_entity_merge_question(
        provisional_node_id="p", entity_name="X", candidate_node_id="c", candidate_name="Y"
    )

    assert isinstance(question, EntityMergeQuestion)
    assert not isinstance(question, QueryDisambiguationQuestion)


def test_entity_merge_question_requires_provisional_and_candidate_ids():
    with pytest.raises(TypeError):
        EntityMergeQuestion(
            id="q1",
            question="?",
            options=[],
            candidate_node_id="c",
            # provisional_node_id omitted - must not silently default to None.
        )


def test_register_query_disambiguation_returns_the_discriminated_type(fake_db):
    orchestrator = ClarificationOrchestrator()
    question = orchestrator.register_query_disambiguation(
        "what is attention?",
        [NodeSearchHit(node_id="a", score=0.5, name="A", type="CONCEPT", description="")],
    )

    assert isinstance(question, QueryDisambiguationQuestion)
    assert not isinstance(question, EntityMergeQuestion)


def test_query_disambiguation_question_requires_query_text():
    with pytest.raises(TypeError):
        QueryDisambiguationQuestion(id="q1", question="?", options=[])


def test_entity_merge_question_merge_answer_writes_same_as_edge(fake_db):
    gm = _make_graph(fake_db)
    existing = _node("Fine-Tuning")
    provisional = _node("Parameter Tuning")
    gm.add_node(existing)
    gm.add_node(provisional)

    orchestrator = ClarificationOrchestrator(graph_manager=gm)
    question = orchestrator.register_entity_merge_question(
        provisional_node_id=provisional.id,
        entity_name="Parameter Tuning",
        candidate_node_id=existing.id,
        candidate_name="Fine-Tuning",
        score=0.8,
    )

    assert orchestrator.pending() == [question]

    orchestrator.answer(question.id, existing.id)

    edges = list(gm.graph.get_edge_data(provisional.id, existing.id).values())
    assert len(edges) == 1
    assert edges[0]["type"] == "SAME_AS"
    assert orchestrator.pending() == []
    assert orchestrator.get(question.id).status == "answered"


def test_entity_merge_question_distinct_answer_marks_known_distinct(fake_db):
    gm = _make_graph(fake_db)
    existing = _node("Fine-Tuning")
    provisional = _node("Parameter Tuning")
    gm.add_node(existing)
    gm.add_node(provisional)

    orchestrator = ClarificationOrchestrator(graph_manager=gm)
    question = orchestrator.register_entity_merge_question(
        provisional_node_id=provisional.id,
        entity_name="Parameter Tuning",
        candidate_node_id=existing.id,
        candidate_name="Fine-Tuning",
    )

    orchestrator.answer(question.id, DISTINCT_OPTION_ID)

    assert tuple(sorted((existing.id, provisional.id))) in gm._known_distinct
    # No merge edge should have been written.
    assert gm.graph.get_edge_data(provisional.id, existing.id) is None


def test_entity_merge_answer_without_graph_manager_raises():
    orchestrator = ClarificationOrchestrator()
    question = orchestrator.register_entity_merge_question(
        provisional_node_id="p",
        entity_name="X",
        candidate_node_id="c",
        candidate_name="Y",
    )

    with pytest.raises(RuntimeError):
        orchestrator.answer(question.id, "c")


def test_answer_rejects_unknown_option(fake_db):
    orchestrator = ClarificationOrchestrator(graph_manager=_make_graph(fake_db))
    question = orchestrator.register_entity_merge_question(
        provisional_node_id="p",
        entity_name="X",
        candidate_node_id="c",
        candidate_name="Y",
    )

    with pytest.raises(ValueError):
        orchestrator.answer(question.id, "not-a-real-option")


def test_answer_rejects_unknown_question_id():
    orchestrator = ClarificationOrchestrator()
    with pytest.raises(KeyError):
        orchestrator.answer("does-not-exist", "anything")


def test_query_disambiguation_answer_records_choice_without_graph_mutation(fake_db):
    """Answering a query_disambiguation question shouldn't need a
    graph_manager or write anything - unlike entity_merge, nothing here
    was ever wrong, just ambiguous, so applying the answer is just
    recording which one the person meant."""
    orchestrator = ClarificationOrchestrator()
    candidates = [
        NodeSearchHit(
            node_id="a", score=0.5, name="Attention (method)", type="METHOD", description=""
        ),
        NodeSearchHit(
            node_id="b", score=0.45, name="Attention (concept)", type="CONCEPT", description=""
        ),
    ]

    question = orchestrator.register_query_disambiguation(
        "What is attention?", candidates
    )
    orchestrator.answer(question.id, "b")

    answered = orchestrator.get(question.id)
    assert answered.status == "answered"
    assert answered.answer_option_id == "b"
    assert answered.query_text == "What is attention?"


def test_pending_only_returns_open_questions(fake_db):
    orchestrator = ClarificationOrchestrator(graph_manager=_make_graph(fake_db))
    q1 = orchestrator.register_entity_merge_question(
        provisional_node_id="p1", entity_name="X", candidate_node_id="c1", candidate_name="Y"
    )
    orchestrator.register_entity_merge_question(
        provisional_node_id="p2", entity_name="Z", candidate_node_id="c2", candidate_name="W"
    )

    orchestrator.answer(q1.id, DISTINCT_OPTION_ID)

    assert len(orchestrator.pending()) == 1
    assert q1 not in orchestrator.pending()


def test_terminal_review_loop_answers_and_skips(fake_db):
    gm = _make_graph(fake_db)
    existing = _node("Fine-Tuning")
    provisional = _node("Parameter Tuning")
    gm.add_node(existing)
    gm.add_node(provisional)

    orchestrator = ClarificationOrchestrator(graph_manager=gm)
    answer_now = orchestrator.register_entity_merge_question(
        provisional_node_id=provisional.id,
        entity_name="Parameter Tuning",
        candidate_node_id=existing.id,
        candidate_name="Fine-Tuning",
    )
    skip_this = orchestrator.register_query_disambiguation(
        "ambiguous query",
        [
            NodeSearchHit(node_id="a", score=0.5, name="A", type="CONCEPT", description=""),
            NodeSearchHit(node_id="b", score=0.48, name="B", type="CONCEPT", description=""),
        ],
    )

    responses = iter(["1", "s"])  # answer the first question, skip the second
    printed: list[str] = []

    answered = orchestrator.run_terminal_review_loop(
        input_fn=lambda _: next(responses), print_fn=lambda *a: printed.append(" ".join(map(str, a)))
    )

    assert answered == 1
    assert orchestrator.get(answer_now.id).status == "answered"
    assert orchestrator.get(skip_this.id).status == "open"


def test_terminal_review_loop_rejects_zero_instead_of_wrapping_to_last_option(fake_db):
    """Options are shown 1-indexed. int("0") - 1 == -1, which Python list
    indexing silently resolves to the last element instead of raising -
    typing "0" (a plausible off-by-one typo) must not silently apply
    whichever option happens to be last, e.g. the "no, genuinely
    different" option on an entity_merge question, which triggers a real
    resolve_alias(distinct=True) graph mutation the user never chose."""
    gm = _make_graph(fake_db)
    existing = _node("Fine-Tuning")
    provisional = _node("Parameter Tuning")
    gm.add_node(existing)
    gm.add_node(provisional)

    orchestrator = ClarificationOrchestrator(graph_manager=gm)
    question = orchestrator.register_entity_merge_question(
        provisional_node_id=provisional.id,
        entity_name="Parameter Tuning",
        candidate_node_id=existing.id,
        candidate_name="Fine-Tuning",
    )

    answered = orchestrator.run_terminal_review_loop(
        input_fn=lambda _: "0", print_fn=lambda *a: None
    )

    assert answered == 0
    assert orchestrator.get(question.id).status == "open"
    # No merge/distinct mutation should have happened.
    assert gm.graph.get_edge_data(provisional.id, existing.id) is None
    assert tuple(sorted((existing.id, provisional.id))) not in gm._known_distinct


def test_terminal_review_loop_rejects_negative_choice(fake_db):
    orchestrator = ClarificationOrchestrator(graph_manager=_make_graph(fake_db))
    question = orchestrator.register_entity_merge_question(
        provisional_node_id="p", entity_name="X", candidate_node_id="c", candidate_name="Y"
    )

    answered = orchestrator.run_terminal_review_loop(
        input_fn=lambda _: "-1", print_fn=lambda *a: None
    )

    assert answered == 0
    assert orchestrator.get(question.id).status == "open"


def test_terminal_review_loop_handles_invalid_choice_gracefully(fake_db):
    orchestrator = ClarificationOrchestrator(graph_manager=_make_graph(fake_db))
    question = orchestrator.register_entity_merge_question(
        provisional_node_id="p", entity_name="X", candidate_node_id="c", candidate_name="Y"
    )

    answered = orchestrator.run_terminal_review_loop(
        input_fn=lambda _: "99", print_fn=lambda *a: None
    )

    assert answered == 0
    assert orchestrator.get(question.id).status == "open"


def test_terminal_review_loop_reports_no_pending_questions():
    orchestrator = ClarificationOrchestrator()
    printed: list[str] = []

    answered = orchestrator.run_terminal_review_loop(
        input_fn=lambda _: "", print_fn=lambda *a: printed.append(" ".join(map(str, a)))
    )

    assert answered == 0
    assert any("No pending" in line for line in printed)


def test_open_questions_rehydrate_from_firestore(fake_db):
    first = ClarificationOrchestrator(db_client=fake_db)
    question = first.register_entity_merge_question(
        provisional_node_id="new-node",
        entity_name="New",
        candidate_node_id="existing-node",
        candidate_name="Existing",
    )

    restored = ClarificationOrchestrator(db_client=fake_db)

    assert restored.get(question.id) is not None
    assert [item.id for item in restored.pending()] == [question.id]


def test_remove_for_node_ids_clears_matching_questions_in_memory_and_firestore(
    fake_db,
):
    orchestrator = ClarificationOrchestrator(db_client=fake_db)
    matching = orchestrator.register_entity_merge_question(
        provisional_node_id="dying-node",
        entity_name="New",
        candidate_node_id="existing-node",
        candidate_name="Existing",
    )
    other = orchestrator.register_entity_merge_question(
        provisional_node_id="unrelated-node",
        entity_name="Unrelated",
        candidate_node_id="another-node",
        candidate_name="Another",
    )
    query_question = orchestrator.register_query_disambiguation(
        "what is attention?",
        [NodeSearchHit(node_id="a", score=0.5, name="A", type="CONCEPT", description="")],
    )

    removed = orchestrator.remove_for_node_ids({"dying-node"})

    assert removed == 1
    assert orchestrator.get(matching.id) is None
    assert orchestrator.get(other.id) is not None
    assert orchestrator.get(query_question.id) is not None

    restored = ClarificationOrchestrator(db_client=fake_db)
    assert restored.get(matching.id) is None
    assert restored.get(other.id) is not None
