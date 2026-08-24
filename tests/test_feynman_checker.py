from __future__ import annotations

import uuid

from agent.feynman_checker import FeynmanChecker, _VerdictPayload, pick_check_nodes
from agent.graph_manager import GraphManager
from agent.schema import Edge, EdgeType, Node, NodeType, ProvenanceTag


def _make_manager(fake_db) -> GraphManager:
    return GraphManager(project_id="test-project", db_client=fake_db)


def _node(
    name: str,
    node_type: NodeType,
    session_id: str = "session-a",
    description: str | None = None,
) -> Node:
    return Node(
        id=str(uuid.uuid4()),
        type=node_type,
        name=name,
        description=description if description is not None else f"{name} is a real, substantive description for testing.",
        session_id=session_id,
    )


def _link(gm: GraphManager, source: Node, target: Node, paper_id: str) -> None:
    gm.add_edge(
        Edge(
            id=str(uuid.uuid4()),
            source_id=source.id,
            target_id=target.id,
            type=EdgeType.USES,
            provenance=ProvenanceTag.EXTRACTED,
            source_paper_id=paper_id,
            session_id="session-a",
        )
    )


class _FakeJudge:
    def __init__(self, verdict: str | None = "strong", explanation: str = "Good."):
        self.calls: list[tuple[str, str, str]] = []
        self._verdict = verdict
        self._explanation = explanation

    def __call__(self, node_name: str, node_description: str, explanation: str):
        self.calls.append((node_name, node_description, explanation))
        if self._verdict is None:
            return None
        return _VerdictPayload(verdict=self._verdict, explanation=self._explanation)


def test_pick_check_nodes_excludes_paper_nodes_and_scopes_to_the_right_paper(fake_db):
    gm = _make_manager(fake_db)
    paper = _node("Paper A", NodeType.PAPER)
    concept = _node("Concept A", NodeType.CONCEPT)
    other_paper_concept = _node("Concept B", NodeType.CONCEPT)
    gm.add_node(paper)
    gm.add_node(concept)
    gm.add_node(other_paper_concept)
    _link(gm, paper, concept, paper_id="paper-a")
    _link(gm, paper, other_paper_concept, paper_id="paper-b")

    prompts = pick_check_nodes(gm, paper_id="paper-a", session_id="session-a")

    node_ids = {p.node_id for p in prompts}
    assert concept.id in node_ids
    assert paper.id not in node_ids
    assert other_paper_concept.id not in node_ids


def test_pick_check_nodes_scopes_to_session(fake_db):
    gm = _make_manager(fake_db)
    paper = _node("Paper A", NodeType.PAPER, session_id="session-a")
    other_session_concept = _node("Concept X", NodeType.CONCEPT, session_id="session-b")
    gm.add_node(paper)
    gm.add_node(other_session_concept)
    _link(gm, paper, other_session_concept, paper_id="paper-a")

    prompts = pick_check_nodes(gm, paper_id="paper-a", session_id="session-a")

    assert prompts == []


def test_pick_check_nodes_ranks_by_degree_within_the_paper(fake_db):
    gm = _make_manager(fake_db)
    paper = _node("Paper A", NodeType.PAPER)
    central = _node("Central Method", NodeType.METHOD)
    peripheral = _node("Peripheral Metric", NodeType.METRIC)
    extra = _node("Extra Concept", NodeType.CONCEPT)
    gm.add_node(paper)
    gm.add_node(central)
    gm.add_node(peripheral)
    gm.add_node(extra)
    _link(gm, paper, central, paper_id="paper-a")
    _link(gm, central, extra, paper_id="paper-a")
    _link(gm, paper, peripheral, paper_id="paper-a")

    prompts = pick_check_nodes(gm, paper_id="paper-a", session_id="session-a", count=1)

    assert len(prompts) == 1
    assert prompts[0].node_id == central.id
    assert "Central Method" in prompts[0].question


def test_pick_check_nodes_skips_nodes_with_a_placeholder_description(fake_db):
    """A node described only as "<name> method" has nothing real for the
    judge to grade against - confirmed live to make the judge silently fall
    back to Gemini's own background knowledge instead of graph evidence.
    Excluded even though it has higher degree than the real-description
    node, which must still be picked."""
    gm = _make_manager(fake_db)
    paper = _node("Paper A", NodeType.PAPER)
    thin = _node("PrefixEmbed", NodeType.METHOD, description="PrefixEmbed method")
    real = _node(
        "Central Method",
        NodeType.METHOD,
        description="A method that prepends trainable tokens to every transformer layer's input.",
    )
    extra = _node("Extra Concept", NodeType.CONCEPT)
    gm.add_node(paper)
    gm.add_node(thin)
    gm.add_node(real)
    gm.add_node(extra)
    _link(gm, paper, thin, paper_id="paper-a")
    _link(gm, thin, extra, paper_id="paper-a")  # gives `thin` the higher degree
    _link(gm, paper, real, paper_id="paper-a")

    prompts = pick_check_nodes(gm, paper_id="paper-a", session_id="session-a")

    node_ids = {p.node_id for p in prompts}
    assert real.id in node_ids
    assert thin.id not in node_ids


def test_check_returns_the_judges_verdict_with_a_graph_citation(fake_db):
    gm = _make_manager(fake_db)
    paper = _node("Paper A", NodeType.PAPER)
    concept = _node("Concept A", NodeType.CONCEPT)
    gm.add_node(paper)
    gm.add_node(concept)
    _link(gm, paper, concept, paper_id="paper-a")
    judge = _FakeJudge(verdict="weak", explanation="Missed the key mechanism.")
    checker = FeynmanChecker(gm, judge=judge)

    result = checker.check(
        concept.id, "It's a thing that helps.", paper_id="paper-a", session_id="session-a"
    )

    assert result is not None
    assert result.verdict == "weak"
    assert result.explanation == "Missed the key mechanism."
    assert result.citation is not None
    assert result.citation.node_ids == [concept.id]
    assert judge.calls == [
        ("Concept A", "Concept A is a real, substantive description for testing.", "It's a thing that helps.")
    ]


def test_check_returns_none_when_the_judge_call_fails(fake_db):
    gm = _make_manager(fake_db)
    paper = _node("Paper A", NodeType.PAPER)
    concept = _node("Concept A", NodeType.CONCEPT)
    gm.add_node(paper)
    gm.add_node(concept)
    _link(gm, paper, concept, paper_id="paper-a")
    checker = FeynmanChecker(gm, judge=_FakeJudge(verdict=None))

    result = checker.check(
        concept.id, "Some explanation.", paper_id="paper-a", session_id="session-a"
    )

    assert result is None


def test_check_returns_none_for_an_unknown_node(fake_db):
    gm = _make_manager(fake_db)
    checker = FeynmanChecker(gm, judge=_FakeJudge())

    result = checker.check(
        "does-not-exist", "Some explanation.", paper_id="paper-a", session_id="session-a"
    )

    assert result is None


def test_check_rejects_a_node_from_a_different_session(fake_db):
    """The exact live-confirmed vulnerability: a node_id scraped from one
    session, submitted against a different session_id, must never be
    graded - it would otherwise leak that node's real name/description with
    no authorization check at all."""
    gm = _make_manager(fake_db)
    paper = _node("Paper A", NodeType.PAPER, session_id="session-a")
    foreign_concept = _node("Foreign Concept", NodeType.CONCEPT, session_id="session-b")
    gm.add_node(paper)
    gm.add_node(foreign_concept)
    _link(gm, paper, foreign_concept, paper_id="paper-a")
    judge = _FakeJudge()
    checker = FeynmanChecker(gm, judge=judge)

    result = checker.check(
        foreign_concept.id, "attacker probe", paper_id="paper-a", session_id="session-a"
    )

    assert result is None
    assert judge.calls == []  # never even reaches the judge


def test_check_rejects_a_node_that_does_not_belong_to_the_given_paper(fake_db):
    """Same node, same session - but submitted alongside a paper_id it has
    no real edge connection to. Must not leak content from a paper the
    caller didn't actually ask about."""
    gm = _make_manager(fake_db)
    paper_a = _node("Paper A", NodeType.PAPER)
    paper_b = _node("Paper B", NodeType.PAPER)
    concept = _node("Concept A", NodeType.CONCEPT)
    gm.add_node(paper_a)
    gm.add_node(paper_b)
    gm.add_node(concept)
    _link(gm, paper_a, concept, paper_id="paper-a")  # concept only belongs to paper-a
    judge = _FakeJudge()
    checker = FeynmanChecker(gm, judge=judge)

    result = checker.check(
        concept.id, "attacker probe", paper_id="paper-b", session_id="session-a"
    )

    assert result is None
    assert judge.calls == []


def test_check_rejects_a_paper_node_directly(fake_db):
    """pick_check_nodes never offers a PAPER node as a question ("what is
    this paper" isn't a real recall test), but check() is a separate entry
    point that took whatever node_id it was given - confirmed live that
    submitting the paper's own node_id directly still got graded. Must be
    rejected the same way pick_check_nodes already excludes it."""
    gm = _make_manager(fake_db)
    paper = _node("Paper A", NodeType.PAPER)
    gm.add_node(paper)
    judge = _FakeJudge()
    checker = FeynmanChecker(gm, judge=judge)

    result = checker.check(
        paper.id, "This is the paper itself.", paper_id="paper-a", session_id="session-a"
    )

    assert result is None
    assert judge.calls == []
