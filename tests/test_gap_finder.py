from __future__ import annotations

import uuid
from types import SimpleNamespace

from agent.gap_finder import GapFinder, GeminiExplainer
from agent.graph_manager import GraphManager
from agent.schema import Edge, EdgeType, Node, NodeType, ProvenanceTag


def _node(name: str) -> Node:
    return Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT, name=name)


def _edge(source_id: str, target_id: str) -> Edge:
    return Edge(
        id=str(uuid.uuid4()),
        source_id=source_id,
        target_id=target_id,
        type=EdgeType.SUPPORTS,
        provenance=ProvenanceTag.EXTRACTED,
        source_quote="quote",
    )


def _populated_graph(fake_db) -> tuple[GraphManager, Node, Node, Node]:
    """a and b share a neighbor (shared) but have no direct edge to each
    other — the exact shape find_sparse_pairs is meant to surface."""
    gm = GraphManager(project_id="test-project", db_client=fake_db)
    a, b, shared = _node("Sparse Concept A"), _node("Sparse Concept B"), _node(
        "Shared Neighbor"
    )
    gm.add_node(a)
    gm.add_node(b)
    gm.add_node(shared)
    gm.add_edge(_edge(shared.id, a.id))
    gm.add_edge(_edge(shared.id, b.id))
    return gm, a, b, shared


def _stub_explain(name_a: str, name_b: str, evidence: list[str]) -> str:
    return f"stub explanation: {name_a} / {name_b} / {evidence}"


def test_find_candidates_surfaces_sparse_pair_with_evidence(fake_db):
    gm, a, b, shared = _populated_graph(fake_db)
    gf = GapFinder(gm, explain_fn=_stub_explain)

    candidates = gf.find_candidates()

    assert len(candidates) == 1
    candidate = candidates[0]
    assert {candidate.node_a_id, candidate.node_b_id} == {a.id, b.id}
    assert candidate.common_neighbor_ids == [shared.id]
    assert candidate.explanation is not None


def test_find_candidates_topology_decides_not_llm(fake_db):
    """Score is purely a function of common-neighbor count — no model call
    is involved in deciding which candidates surface, only in phrasing."""
    gm, a, b, shared = _populated_graph(fake_db)
    gf = GapFinder(gm, explain_fn=_stub_explain)

    candidates = gf.find_candidates()
    assert candidates[0].score == 1.0  # exactly one common neighbor


def test_feedback_boosts_future_ranking(fake_db):
    gm, a, b, shared = _populated_graph(fake_db)
    other = _node("Unrelated Concept")
    other2 = _node("Another Unrelated Concept")
    gm.add_node(other)
    gm.add_node(other2)
    gm.add_edge(_edge(shared.id, other.id))
    gm.add_edge(_edge(shared.id, other2.id))

    gf = GapFinder(gm, explain_fn=_stub_explain)
    before = gf.find_candidates(limit=10)
    ab_score_before = next(
        c.score for c in before if {c.node_a_id, c.node_b_id} == {a.id, b.id}
    )

    gf.record_feedback(a.id, b.id, interesting=True)

    after = gf.find_candidates(limit=10)
    ab_score_after = next(
        c.score for c in after if {c.node_a_id, c.node_b_id} == {a.id, b.id}
    )
    assert ab_score_after > ab_score_before


def test_not_interesting_feedback_lowers_future_ranking(fake_db):
    gm, a, b, shared = _populated_graph(fake_db)
    gf = GapFinder(gm, explain_fn=_stub_explain)

    before = gf.find_candidates()[0].score
    gf.record_feedback(a.id, b.id, interesting=False)
    after = gf.find_candidates()[0].score

    assert after < before


def test_feedback_writes_event_to_firestore(fake_db):
    gm, a, b, shared = _populated_graph(fake_db)
    gf = GapFinder(gm, explain_fn=_stub_explain, db_client=fake_db)

    gf.record_feedback(a.id, b.id, interesting=True)

    events = list(fake_db.collection("feedback_events").stream())
    assert len(events) == 1
    assert events[0].to_dict()["type"] == "gap_rating"
    assert events[0].to_dict()["interesting"] is True


class _FakeModels:
    def __init__(self, text: str | None = None, error: Exception | None = None):
        self._text = text
        self._error = error
        self.last_call: dict | None = None

    def generate_content(self, *, model: str, contents: str):
        self.last_call = {"model": model, "contents": contents}
        if self._error is not None:
            raise self._error
        return SimpleNamespace(text=self._text)


class _FakeGenaiClient:
    def __init__(self, text: str | None = None, error: Exception | None = None):
        self.models = _FakeModels(text=text, error=error)


def test_gemini_explainer_returns_model_text():
    fake_client = _FakeGenaiClient(text="  A genuinely interesting gap.  ")
    explainer = GeminiExplainer(client=fake_client)

    result = explainer("Chain of Thought", "Reasoning", ["Prompting"])

    assert result == "A genuinely interesting gap."
    assert fake_client.models.last_call["model"] == "gemini-2.5-flash"
    assert "Chain of Thought" in fake_client.models.last_call["contents"]
    assert "Reasoning" in fake_client.models.last_call["contents"]
    assert "Prompting" in fake_client.models.last_call["contents"]


def test_gemini_explainer_falls_back_on_empty_response():
    fake_client = _FakeGenaiClient(text="   ")
    explainer = GeminiExplainer(client=fake_client)

    result = explainer("A", "B", [])

    assert "share context" in result  # the deterministic template's phrasing


def test_gemini_explainer_falls_back_on_exception():
    fake_client = _FakeGenaiClient(error=RuntimeError("quota exceeded"))
    explainer = GeminiExplainer(client=fake_client)

    result = explainer("A", "B", ["Evidence"])

    assert "share context" in result  # never raises, degrades to the template


def test_gap_finder_uses_gemini_explainer_end_to_end(fake_db):
    gm, a, b, shared = _populated_graph(fake_db)
    fake_client = _FakeGenaiClient(text="Real Gemini explanation.")
    gf = GapFinder(gm, explain_fn=GeminiExplainer(client=fake_client))

    candidates = gf.find_candidates()

    assert candidates[0].explanation == "Real Gemini explanation."
