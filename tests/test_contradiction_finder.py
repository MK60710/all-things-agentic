from __future__ import annotations

import uuid

import pytest

from agent.contradiction_finder import ContradictionFinder, _VerdictPayload
from agent.graph_manager import GraphManager
from agent.schema import Edge, EdgeType, Node, NodeType, ProvenanceTag


def _make_manager(fake_db) -> GraphManager:
    return GraphManager(project_id="test-project", db_client=fake_db)


def _claim(name: str, embedding: list[float], session_id: str = "session-a") -> Node:
    return Node(
        id=str(uuid.uuid4()),
        type=NodeType.CLAIM,
        name=name,
        description=name,
        entity_embedding=embedding,
        session_id=session_id,
    )


class _CountingJudge:
    """Records every call it receives and returns a fixed verdict - lets
    tests assert both what verdict was applied and, separately, whether a
    call happened at all (the whole point of the already-checked skip)."""

    def __init__(self, verdict: str | None = "consistent", explanation: str = "explanation"):
        self.calls: list[tuple[str, str]] = []
        self._verdict = verdict
        self._explanation = explanation

    def __call__(self, claim_a: str, claim_b: str) -> _VerdictPayload | None:
        self.calls.append((claim_a, claim_b))
        if self._verdict is None:
            return None
        return _VerdictPayload(verdict=self._verdict, explanation=self._explanation)


def test_only_pairs_above_similarity_threshold_are_judged(fake_db):
    gm = _make_manager(fake_db)
    close_a = _claim("Close A", [1.0, 0.0])
    close_b = _claim("Close B", [1.0, 0.0])
    far = _claim("Unrelated", [0.0, 1.0])
    gm.add_node(close_a)
    gm.add_node(close_b)
    gm.add_node(far)
    judge = _CountingJudge()
    finder = ContradictionFinder(gm, judge=judge)

    finder.check_session("session-a")

    judged_pairs = {frozenset(pair) for pair in judge.calls}
    assert frozenset({"Close A", "Close B"}) in judged_pairs
    assert not any("Unrelated" in pair for pair in judged_pairs)


def test_contradicts_verdict_writes_a_real_edge(fake_db):
    gm = _make_manager(fake_db)
    a, b = _claim("Claim A", [1.0, 0.0]), _claim("Claim B", [1.0, 0.0])
    gm.add_node(a)
    gm.add_node(b)
    judge = _CountingJudge(verdict="contradicts", explanation="They disagree.")
    finder = ContradictionFinder(gm, judge=judge)

    results = finder.check_session("session-a")

    assert len(results) == 1
    assert results[0].explanation == "They disagree."
    edges = gm.get_incident_edges(a.id)
    assert len(edges) == 1
    assert edges[0].relation == EdgeType.CONTRADICTS.value


def test_consistent_verdict_does_not_write_an_edge(fake_db):
    gm = _make_manager(fake_db)
    a, b = _claim("Claim A", [1.0, 0.0]), _claim("Claim B", [1.0, 0.0])
    gm.add_node(a)
    gm.add_node(b)
    judge = _CountingJudge(verdict="consistent")
    finder = ContradictionFinder(gm, judge=judge)

    results = finder.check_session("session-a")

    assert results == []
    assert gm.get_incident_edges(a.id) == []


def test_second_run_does_not_recall_the_judge_for_an_already_checked_pair(fake_db):
    gm = _make_manager(fake_db)
    a, b = _claim("Claim A", [1.0, 0.0]), _claim("Claim B", [1.0, 0.0])
    gm.add_node(a)
    gm.add_node(b)
    judge = _CountingJudge(verdict="consistent")
    finder = ContradictionFinder(gm, judge=judge, db_client=fake_db)

    finder.check_session("session-a")
    assert len(judge.calls) == 1

    finder.check_session("session-a")
    assert len(judge.calls) == 1  # not called again


def test_a_failed_judge_call_is_retried_on_the_next_run(fake_db):
    """A None verdict (Gemini outage, bad schema, etc.) must not be
    mistaken for a real 'consistent'/'unrelated' answer - otherwise a
    transient failure would permanently skip a pair forever."""
    gm = _make_manager(fake_db)
    a, b = _claim("Claim A", [1.0, 0.0]), _claim("Claim B", [1.0, 0.0])
    gm.add_node(a)
    gm.add_node(b)
    failing_judge = _CountingJudge(verdict=None)
    finder = ContradictionFinder(gm, judge=failing_judge, db_client=fake_db)
    finder.check_session("session-a")
    assert len(failing_judge.calls) == 1

    working_judge = _CountingJudge(verdict="consistent")
    finder_retry = ContradictionFinder(gm, judge=working_judge, db_client=fake_db)
    finder_retry.check_session("session-a")

    assert len(working_judge.calls) == 1  # retried, not skipped


def test_existing_contradicts_edge_is_not_rechecked(fake_db):
    gm = _make_manager(fake_db)
    a, b = _claim("Claim A", [1.0, 0.0]), _claim("Claim B", [1.0, 0.0])
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge(
        Edge(
            id=str(uuid.uuid4()),
            source_id=a.id,
            target_id=b.id,
            type=EdgeType.CONTRADICTS,
            provenance=ProvenanceTag.INFERRED,
            source_quote="already known to disagree",
        )
    )
    judge = _CountingJudge(verdict="contradicts")
    finder = ContradictionFinder(gm, judge=judge)

    results = finder.check_session("session-a")

    assert results == []
    assert judge.calls == []


def test_claims_from_a_different_session_are_never_compared(fake_db):
    gm = _make_manager(fake_db)
    a = _claim("Claim A", [1.0, 0.0], session_id="session-a")
    b = _claim("Claim B", [1.0, 0.0], session_id="session-b")
    gm.add_node(a)
    gm.add_node(b)
    judge = _CountingJudge(verdict="contradicts")
    finder = ContradictionFinder(gm, judge=judge)

    results = finder.check_session("session-a")

    assert results == []
    assert judge.calls == []


def test_max_llm_calls_caps_how_many_pairs_get_judged(fake_db):
    gm = _make_manager(fake_db)
    claims = [_claim(f"Claim {i}", [1.0, 0.0]) for i in range(4)]
    for c in claims:
        gm.add_node(c)
    judge = _CountingJudge(verdict="consistent")
    finder = ContradictionFinder(gm, judge=judge)

    finder.check_session("session-a", max_llm_calls=2)

    assert len(judge.calls) == 2
