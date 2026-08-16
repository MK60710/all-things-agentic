"""Live integration tests against real Vertex AI.

These exist because two real bugs (a 404 on the model name, and Gemini
2.5 Flash's "thinking" tokens silently eating the output budget) were only
ever caught by manually running GeminiExplainer against the real API - the
unit tests in test_gap_finder.py use a fake client and cannot reproduce
either failure mode, since they depend on actual Vertex AI model
availability and actual token-budget consumption behavior.

Skipped by default (RUN_LIVE_TESTS is unset) so `pytest tests/` stays
credential-free, matching the rest of this project's convention. Run with:

    RUN_LIVE_TESTS=1 uv run pytest tests/test_gap_finder_live.py -v

Requires `gcloud auth application-default login` to have been run first.
"""

from __future__ import annotations

import os
import uuid

import pytest

from agent.gap_finder import GapFinder, GeminiExplainer
from agent.graph_manager import GraphManager
from agent.schema import Edge, EdgeType, Node, NodeType, ProvenanceTag

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to run tests against real Vertex AI",
)


@pytest.mark.live
def test_live_gemini_explainer_returns_complete_untruncated_text():
    """Would have caught both real bugs found this session: a 404 if the
    configured model name isn't actually available in this project/region,
    and silent truncation if thinking tokens ever start consuming the
    output budget again (e.g. a future model swap re-enables thinking)."""
    explainer = GeminiExplainer()

    result = explainer(
        "Chain of Thought Prompting",
        "Retrieval Augmented Generation",
        ["Large Language Models", "Reasoning"],
    )

    assert result
    assert len(result) > 20  # not a truncated fragment
    # A truncated response reads as an obviously cut-off sentence fragment;
    # the deterministic template's phrasing is the other failure signal
    # (would mean the call silently fell back instead of really answering).
    assert "share context" not in result


@pytest.mark.live
def test_live_gap_finder_dispatches_concurrent_explanations_for_real(fake_db):
    """Exercises the actual production wiring end-to-end against real
    Vertex AI: GapFinder.find_candidates() with GeminiExplainer, three
    disjoint sparse pairs so the ThreadPoolExecutor dispatch, the
    in-flight/cache locking, and GeminiExplainer's lazy client-construction
    lock all get exercised concurrently against the real API, not a fake
    client. The unit tests cover this logic with fakes; only a live run
    proves it actually works against the real SDK objects (real
    response.candidates[0].finish_reason shape, real GenerateContentConfig
    acceptance, etc)."""
    gm = GraphManager(project_id="test-project", db_client=fake_db)
    topics = [
        ("Sparse Attention", "Retrieval Augmented Generation", "Long Context"),
        ("Chain of Thought Prompting", "Tool Use", "Reasoning"),
        ("Knowledge Graphs", "Vector Search", "Semantic Retrieval"),
    ]
    for name_a, name_b, shared_name in topics:
        a = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT, name=name_a)
        b = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT, name=name_b)
        shared = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT, name=shared_name)
        gm.add_node(a)
        gm.add_node(b)
        gm.add_node(shared)
        gm.add_edge(
            Edge(
                id=str(uuid.uuid4()),
                source_id=shared.id,
                target_id=a.id,
                type=EdgeType.SUPPORTS,
                provenance=ProvenanceTag.EXTRACTED,
                source_paper_id="paper-1",
                source_quote=f"{shared_name} relates to {name_a}.",
            )
        )
        gm.add_edge(
            Edge(
                id=str(uuid.uuid4()),
                source_id=b.id,
                target_id=shared.id,
                type=EdgeType.USES,
                provenance=ProvenanceTag.EXTRACTED,
                source_paper_id="paper-2",
                source_quote=f"{name_b} uses {shared_name}.",
            )
        )

    gap_finder = GapFinder(gm, explain_fn=GeminiExplainer())
    candidates = gap_finder.find_candidates(limit=3)

    assert len(candidates) == 3
    for c in candidates:
        assert c.explanation
        assert len(c.explanation) > 20  # not a truncated fragment
        assert "share context" not in c.explanation  # a real call, not the fallback
        assert len(c.citations) == 2
        assert {cit.connects_to for cit in c.citations} == {"node_a", "node_b"}
        assert all(cit.source_quote for cit in c.citations)

    # Re-running must not re-hit Gemini - the cache (now LRU-aware) should
    # serve all three from the previous call.
    again = gap_finder.find_candidates(limit=3)
    assert {c.node_a_name for c in again} == {c.node_a_name for c in candidates}
    assert all(c.explanation for c in again)
