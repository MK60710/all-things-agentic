from __future__ import annotations

import threading
import uuid

import pytest

from agent.clarification_orchestrator import ClarificationOrchestrator
from agent.graph_manager import GraphManager, _cosine_similarity
from agent.schema import (
    Edge,
    EdgeType,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
    Node,
    NodeType,
    ProvenanceTag,
)


def _make_manager(fake_db) -> GraphManager:
    return GraphManager(project_id="test-project", db_client=fake_db)


def _node(name: str, embedding: list[float] | None = None, owner_uid: str = "owner-1") -> Node:
    return Node(
        id=str(uuid.uuid4()),
        type=NodeType.CONCEPT,
        name=name,
        entity_embedding=embedding,
        owner_uid=owner_uid,
    )


def test_add_node_is_idempotent(fake_db):
    gm = _make_manager(fake_db)
    node = _node("Retrieval Augmented Generation")
    gm.add_node(node)
    gm.add_node(node)  # simulate a retry
    assert gm.graph.number_of_nodes() == 1


def test_add_edge_is_idempotent(fake_db):
    gm = _make_manager(fake_db)
    a, b = _node("Paper A"), _node("Paper B")
    gm.add_node(a)
    gm.add_node(b)
    edge = Edge(
        id=str(uuid.uuid4()),
        source_id=a.id,
        target_id=b.id,
        type=EdgeType.EXTENDS,
        provenance=ProvenanceTag.EXTRACTED,
        source_quote="A extends the method proposed in B.",
    )
    gm.add_edge(edge)
    gm.add_edge(edge)
    assert gm.graph.number_of_edges() == 1


def test_get_neighbors(fake_db):
    gm = _make_manager(fake_db)
    a, b = _node("Concept A"), _node("Concept B")
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge(
        Edge(
            id=str(uuid.uuid4()),
            source_id=a.id,
            target_id=b.id,
            type=EdgeType.SUPPORTS,
            provenance=ProvenanceTag.EXTRACTED,
            source_quote="quote",
        )
    )
    assert b.id in gm.get_neighbors(a.id)
    assert a.id in gm.get_neighbors(b.id)


def test_search_nodes_ranks_by_token_overlap(fake_db):
    gm = _make_manager(fake_db)
    strong = _node("Sparse Attention Mechanism")
    weak = _node("Mechanism For Data Loading")
    gm.add_node(strong)
    gm.add_node(weak)

    hits = gm.search_nodes("sparse attention mechanism", min_score=0.0)

    assert [hit.node_id for hit in hits[:2]] == [strong.id, weak.id]
    assert hits[0].score > hits[1].score


def test_search_nodes_min_score_gates_low_relevance_matches(fake_db):
    gm = _make_manager(fake_db)
    node = _node("Mechanism For Data Loading")
    gm.add_node(node)

    # Only "mechanism" overlaps out of 4 query tokens (score 0.25) - below
    # a reasonable relevance bar, unlike the old any-overlap-wins behavior.
    hits = gm.search_nodes(
        "sparse attention scoring mechanism", min_score=0.4
    )

    assert hits == []


def test_search_nodes_reflects_updated_node_data(fake_db):
    """The per-node token cache must not serve stale tokens after an
    add_node upsert changes a node's name/description."""
    gm = _make_manager(fake_db)
    node_id = str(uuid.uuid4())
    gm.add_node(Node(id=node_id, type=NodeType.CONCEPT, name="Placeholder"))
    assert gm.search_nodes("gradient clipping", min_score=0.5) == []

    gm.add_node(
        Node(id=node_id, type=NodeType.CONCEPT, name="Gradient Clipping")
    )

    hits = gm.search_nodes("gradient clipping", min_score=0.5)
    assert [hit.node_id for hit in hits] == [node_id]


def test_get_incident_edges_deduplicates_and_resolves_names(fake_db):
    gm = _make_manager(fake_db)
    a, b = _node("Method A"), _node("Metric B")
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge(
        Edge(
            id=str(uuid.uuid4()),
            source_id=a.id,
            target_id=b.id,
            type=EdgeType.USES,
            provenance=ProvenanceTag.EXTRACTED,
            source_paper_id="paper-1",
            source_quote="quote",
        )
    )

    edges_from_a = gm.get_incident_edges(a.id)
    edges_from_b = gm.get_incident_edges(b.id)

    assert len(edges_from_a) == 1
    assert edges_from_a[0].source_name == "Method A"
    assert edges_from_a[0].target_name == "Metric B"
    assert edges_from_b == edges_from_a


def test_export_session_graph_only_returns_nodes_tagged_to_that_session(fake_db):
    gm = _make_manager(fake_db)
    a = _node("Session A Concept")
    a.session_id = "session-a"
    b = _node("Session B Concept")
    b.session_id = "session-b"
    gm.add_node(a)
    gm.add_node(b)

    export = gm.export_session_graph("session-a")

    assert [n.node_id for n in export.nodes] == [a.id]


def test_export_session_graph_includes_edges_between_session_nodes(fake_db):
    gm = _make_manager(fake_db)
    a, b = _node("Method A"), _node("Metric B")
    a.session_id = b.session_id = "session-a"
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge(
        Edge(
            id=str(uuid.uuid4()),
            source_id=a.id,
            target_id=b.id,
            type=EdgeType.USES,
            provenance=ProvenanceTag.EXTRACTED,
            source_quote="quote",
            session_id="session-a",
        )
    )

    export = gm.export_session_graph("session-a")

    assert len(export.edges) == 1
    assert export.edges[0].source_id == a.id
    assert export.edges[0].target_id == b.id


def test_export_session_graph_drops_edge_reaching_outside_the_session(fake_db):
    """Strict scoping: an edge with only one endpoint in the session must
    not appear, even if the edge itself or the other node exists in the
    graph - the Graph Explorer never reaches into the shared graph."""
    gm = _make_manager(fake_db)
    a = _node("Session A Concept")
    a.session_id = "session-a"
    b = _node("Shared Graph Concept")
    b.session_id = "session-b"
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge(
        Edge(
            id=str(uuid.uuid4()),
            source_id=a.id,
            target_id=b.id,
            type=EdgeType.USES,
            provenance=ProvenanceTag.EXTRACTED,
            source_quote="quote",
            session_id="session-a",
        )
    )

    export = gm.export_session_graph("session-a")

    assert export.edges == []


def test_export_session_graph_excludes_same_as_merge_edges(fake_db):
    gm = _make_manager(fake_db)
    a, b = _node("Duplicate Name"), _node("duplicate name")
    a.session_id = b.session_id = "session-a"
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge(
        Edge(
            id=str(uuid.uuid4()),
            source_id=b.id,
            target_id=a.id,
            type=EdgeType.SAME_AS,
            provenance=ProvenanceTag.INFERRED,
            session_id="session-a",
        )
    )

    export = gm.export_session_graph("session-a")

    assert export.edges == []


def test_export_session_graph_derives_and_dedupes_node_citations(fake_db):
    """A node's citations come from its own in-session edges' real
    provenance (paper_id/section), not a separate field on the node
    itself - and a node touched by several edges into the same
    paper/section must not produce repeat entries."""
    gm = _make_manager(fake_db)
    a, b, c = _node("Method A"), _node("Metric B"), _node("Concept C")
    a.session_id = b.session_id = c.session_id = "session-a"
    gm.add_node(a)
    gm.add_node(b)
    gm.add_node(c)
    gm.add_edge(
        Edge(
            id=str(uuid.uuid4()),
            source_id=a.id,
            target_id=b.id,
            type=EdgeType.USES,
            provenance=ProvenanceTag.EXTRACTED,
            source_quote="A uses B.",
            source_paper_id="paper-1",
            source_section="Method",
            session_id="session-a",
        )
    )
    gm.add_edge(
        Edge(
            id=str(uuid.uuid4()),
            source_id=a.id,
            target_id=c.id,
            type=EdgeType.SUPPORTS,
            provenance=ProvenanceTag.EXTRACTED,
            source_quote="A supports C.",
            source_paper_id="paper-1",
            source_section="Method",  # same (paper, section) as the edge above
            session_id="session-a",
        )
    )

    export = gm.export_session_graph("session-a")

    node_a = next(n for n in export.nodes if n.node_id == a.id)
    assert [(cit.paper_id, cit.section) for cit in node_a.citations] == [
        ("paper-1", "Method")
    ]


def test_export_session_graph_node_citations_skip_inferred_edges(fake_db):
    """An INFERRED edge (no real paper/section behind it, e.g. a
    Contradiction Finder edge) must not show up as a fabricated
    citation."""
    gm = _make_manager(fake_db)
    a, b = _node("Claim A"), _node("Claim B")
    a.session_id = b.session_id = "session-a"
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge(
        Edge(
            id=str(uuid.uuid4()),
            source_id=a.id,
            target_id=b.id,
            type=EdgeType.CONTRADICTS,
            provenance=ProvenanceTag.INFERRED,
            source_quote="They disagree.",
            session_id="session-a",
        )
    )

    export = gm.export_session_graph("session-a")

    assert all(n.citations == [] for n in export.nodes)


def test_get_node_returns_none_for_unknown_id(fake_db):
    gm = _make_manager(fake_db)
    assert gm.get_node("does-not-exist") is None


def test_canonicalize_string_match_auto_merges(fake_db):
    gm = _make_manager(fake_db)
    existing = _node("Chain of Thought")
    gm.add_node(existing)
    result = gm.canonicalize("chain of thought", owner_uid="owner-1")  # casing differs
    assert result.decision == "auto_merge"
    assert result.matched_node_id == existing.id


def test_canonicalize_high_similarity_auto_merges(fake_db):
    gm = _make_manager(fake_db)
    existing = _node("Few-Shot Prompting", embedding=[1.0, 0.0])
    gm.add_node(existing)
    result = gm.canonicalize("In-Context Learning", embedding=[1.0, 0.0], owner_uid="owner-1")
    assert result.decision == "auto_merge"


def test_canonicalize_middle_band_needs_clarification(fake_db):
    gm = _make_manager(fake_db)
    existing = _node("Fine-Tuning", embedding=[1.0, 0.0])
    gm.add_node(existing)
    # cosine similarity = 0.8, between LOW (0.75) and HIGH (0.92)
    result = gm.canonicalize("Parameter Tuning", embedding=[0.8, 0.6], owner_uid="owner-1")
    assert result.decision == "needs_clarification"
    assert result.matched_node_id == existing.id


def test_canonicalize_low_similarity_is_new(fake_db):
    gm = _make_manager(fake_db)
    existing = _node("Transformer", embedding=[1.0, 0.0])
    gm.add_node(existing)
    result = gm.canonicalize("Reinforcement Learning", embedding=[0.0, 1.0], owner_uid="owner-1")
    assert result.decision == "new"


def test_canonicalize_unicode_math_variant_auto_merges(fake_db):
    """"𝑄" (U+1D444, Mathematical Italic Capital Q) is a different
    codepoint than plain "Q" - without NFKC normalization it gets stripped
    by _normalize_name entirely instead of matching, landing this pair in
    needs_clarification instead of the exact-match auto_merge tier."""
    gm = _make_manager(fake_db)
    existing = _node("Cochran's Q statistic")
    gm.add_node(existing)
    result = gm.canonicalize("Cochran's \U0001d444 statistic", owner_uid="owner-1")
    assert result.decision == "auto_merge"
    assert result.matched_node_id == existing.id


def test_canonicalize_bare_abbreviation_matches_spelled_out_form(fake_db):
    gm = _make_manager(fake_db)
    existing = _node("moral operational design domain (moral ODD)")
    gm.add_node(existing)
    result = gm.canonicalize("moral ODD", owner_uid="owner-1")
    assert result.decision == "auto_merge"
    assert result.matched_node_id == existing.id


def test_canonicalize_spelled_out_form_matches_existing_bare_abbreviation(fake_db):
    gm = _make_manager(fake_db)
    existing = _node("moral ODD")
    gm.add_node(existing)
    result = gm.canonicalize("moral operational design domain (moral ODD)", owner_uid="owner-1")
    assert result.decision == "auto_merge"
    assert result.matched_node_id == existing.id


def test_canonicalize_does_not_match_two_differently_qualified_parenthetical_forms(
    fake_db,
):
    """Two spelled-out forms that share an abbreviation but each carry
    their own distinguishing prefix are a real judgment call, not an
    auto-merge - this must fall through to the normal embedding path
    rather than picking one silently."""
    gm = _make_manager(fake_db)
    existing = _node(
        "explicit moral operational design domain (moral ODD)",
        embedding=[1.0, 0.0],
    )
    gm.add_node(existing)
    result = gm.canonicalize(
        "moral operational design domain (moral ODD)", embedding=[0.0, 1.0]
    , owner_uid="owner-1")
    assert result.decision == "new"


def test_canonicalize_bare_abbreviation_ambiguous_across_two_forms_falls_through(
    fake_db,
):
    """A bare abbreviation that matches more than one differently-qualified
    spelled-out form in the graph must not be silently decided by
    iteration order - it falls through to the embedding path like any
    other ambiguous name."""
    gm = _make_manager(fake_db)
    gm.add_node(
        _node(
            "moral operational design domain (moral ODD)",
            embedding=[1.0, 0.0],
        )
    )
    gm.add_node(
        _node(
            "explicit moral operational design domain (moral ODD)",
            embedding=[1.0, 0.0],
        )
    )
    result = gm.canonicalize("moral ODD", embedding=[0.0, 1.0], owner_uid="owner-1")
    assert result.decision == "new"


def test_cosine_similarity_rejects_dimension_mismatch():
    assert _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0


def test_resolve_alias_merge_writes_same_as_edge(fake_db):
    gm = _make_manager(fake_db)
    canonical, alias = _node("LLM Agent"), _node("Language Model Agent")
    gm.add_node(canonical)
    gm.add_node(alias)
    gm.resolve_alias(canonical.id, alias.id, owner_uid="owner-1")
    gm.resolve_alias(canonical.id, alias.id, owner_uid="owner-1")
    edges = list(gm.graph.get_edge_data(alias.id, canonical.id).values())
    assert len(edges) == 1
    assert edges[0]["type"] == EdgeType.SAME_AS.value
    assert edges[0]["provenance"] == ProvenanceTag.INFERRED.value


def test_resolve_alias_distinct_prevents_recollision(fake_db):
    gm = _make_manager(fake_db)
    a, b = _node("Concept A", embedding=[1.0, 0.0]), _node(
        "Concept B", embedding=[0.8, 0.6]
    )
    gm.add_node(a)
    gm.add_node(b)
    gm.resolve_alias(a.id, b.id, owner_uid="owner-1", distinct=True)
    assert tuple(sorted((a.id, b.id))) in gm._known_distinct

    rehydrated = _make_manager(fake_db)
    assert tuple(sorted((a.id, b.id))) in rehydrated._known_distinct


def test_find_sparse_pairs_requires_common_neighbor(fake_db):
    gm = _make_manager(fake_db)
    a, b, shared = _node("A"), _node("B"), _node("Shared Neighbor")
    gm.add_node(a)
    gm.add_node(b)
    gm.add_node(shared)
    for target in (a, b):
        gm.add_edge(
            Edge(
                id=str(uuid.uuid4()),
                source_id=shared.id,
                target_id=target.id,
                type=EdgeType.SUPPORTS,
                provenance=ProvenanceTag.EXTRACTED,
                source_quote="quote",
            )
        )
    pairs = gm.find_sparse_pairs()
    assert (a.id, b.id) in pairs or (b.id, a.id) in pairs


def test_find_sparse_pairs_sees_a_node_shared_across_sessions_from_either_one(fake_db):
    """Regression test for the live-confirmed bug: a node created by one
    session and later reused by a second session's ingest (via
    canonicalization) must be visible to session-scoped candidate
    generation from BOTH sessions, not just the first."""
    gm = _make_manager(fake_db)
    a, b, shared = _node("A"), _node("B"), _node("Shared Neighbor")
    a.session_id = "session-a"
    b.session_id = "session-a"
    gm.add_node(a)
    gm.add_node(b)
    gm.add_node(shared)
    gm.add_node(Node(**{**shared.model_dump(), "session_id": "session-a"}))
    gm.add_node(Node(**{**shared.model_dump(), "session_id": "session-b"}))
    for target in (a, b):
        gm.add_edge(
            Edge(
                id=str(uuid.uuid4()),
                source_id=shared.id,
                target_id=target.id,
                type=EdgeType.SUPPORTS,
                provenance=ProvenanceTag.EXTRACTED,
                source_quote="quote",
                session_id="session-a",
            )
        )

    pairs_a = gm.find_sparse_pairs(session_id="session-a")
    assert (a.id, b.id) in pairs_a or (b.id, a.id) in pairs_a

    # session-b only owns the shared node itself (not a/b), so it has no
    # sparse pairs of its own here - the real assertion is that the
    # shared node's session filter doesn't silently exclude it either.
    candidates_b = [
        n
        for n, data in gm.graph.nodes(data=True)
        if "session-b" in data.get("session_ids", [])
    ]
    assert shared.id in candidates_b


def test_rehydrate_loads_existing_data(fake_db):
    gm = _make_manager(fake_db)
    node = _node("Persisted Concept")
    gm.add_node(node)

    # New GraphManager instance, same fake_db — simulates Cloud Run cold start.
    gm2 = _make_manager(fake_db)
    assert node.id in gm2.graph


def test_apply_extraction_result_is_idempotent(fake_db):
    gm = _make_manager(fake_db)
    extraction = ExtractionResult(
        paper_id="paper-1",
        entities=[
            ExtractedEntity(
                name="Chain of Thought",
                type=NodeType.CONCEPT,
                description="Reasoning style",
            ),
            ExtractedEntity(
                name="Reasoning",
                type=NodeType.CONCEPT,
                description="Inference process",
            ),
        ],
        relations=[
            ExtractedRelation(
                source_entity="Chain of Thought",
                relation=EdgeType.SUPPORTS,
                target_entity="Reasoning",
                source_quote="Chain of thought prompts can support reasoning.",
                source_section="Introduction",
            )
        ],
        chunks=["chunk one"],
    )

    first = gm.apply_extraction_result(extraction, paper_name="Paper Title", owner_uid="owner-1")
    second = gm.apply_extraction_result(extraction, paper_name="Paper Title", owner_uid="owner-1")

    assert first.paper_node_id != "paper-1"
    assert len(first.node_writes) == 2
    assert len(first.edge_writes) == 3  # two MENTIONS edges plus SUPPORTS
    assert gm.graph.number_of_nodes() == 3
    assert gm.graph.number_of_edges() == 3
    first_supports = next(edge for edge in first.edge_writes if edge.relation == "SUPPORTS")
    second_supports = next(edge for edge in second.edge_writes if edge.relation == "SUPPORTS")
    assert second_supports.edge_id == first_supports.edge_id


def test_apply_extraction_result_tags_new_nodes_and_edges_with_session_id(fake_db):
    gm = _make_manager(fake_db)
    extraction = ExtractionResult(
        paper_id="paper-session-tag",
        entities=[
            ExtractedEntity(name="Chain of Thought", type=NodeType.CONCEPT, description=""),
            ExtractedEntity(name="Reasoning", type=NodeType.CONCEPT, description=""),
        ],
        relations=[
            ExtractedRelation(
                source_entity="Chain of Thought",
                relation=EdgeType.SUPPORTS,
                target_entity="Reasoning",
                source_quote="Chain of thought prompts can support reasoning.",
            )
        ],
    )

    report = gm.apply_extraction_result(
        extraction, paper_name="Paper Title", session_id="session-a"
    , owner_uid="owner-1")

    assert gm.graph.nodes[report.paper_node_id]["session_ids"] == ["session-a"]
    for write in report.node_writes:
        assert gm.graph.nodes[write.node_id]["session_ids"] == ["session-a"]
    edge_write = report.edge_writes[0]
    edge_data = gm.graph.edges[edge_write.source_id, edge_write.target_id, edge_write.edge_id]
    assert edge_data["session_ids"] == ["session-a"]


def test_apply_extraction_result_reused_node_adds_the_new_session_alongside_the_original(fake_db):
    """A node created by one session and later merged into by a different
    session's ingest must become visible to BOTH sessions, not just the
    first - confirmed live as a real bug where a well-known entity reused
    from an earlier session stayed invisible to whichever session reused
    it, silently starving Gap Finder/Contradiction Finder/Feynman Check
    of real data. scripts/clear_session.py's own cleanup now strips just
    one session's membership rather than deleting a still-shared node."""
    gm = _make_manager(fake_db)
    first = ExtractionResult(
        paper_id="paper-one",
        entities=[
            ExtractedEntity(name="Chain of Thought", type=NodeType.CONCEPT, description="")
        ],
        relations=[],
    )
    gm.apply_extraction_result(first, session_id="session-a", owner_uid="owner-1")

    second = ExtractionResult(
        paper_id="paper-two",
        entities=[
            ExtractedEntity(name="chain of thought", type=NodeType.CONCEPT, description="")
        ],  # casing differs - exact-match auto_merge into the session-a node
        relations=[],
    )
    report = gm.apply_extraction_result(second, session_id="session-b", owner_uid="owner-1")

    reused_node_id = report.node_writes[0].node_id
    assert report.node_writes[0].reused_existing_node is True
    assert gm.graph.nodes[reused_node_id]["session_ids"] == ["session-a", "session-b"]


def test_concurrent_ingests_of_the_same_new_entity_do_not_create_duplicate_nodes(fake_db, monkeypatch):
    """Two real threads, each ingesting a different paper that introduces
    the exact same new entity name, must not each independently decide
    "new" and create two separate nodes for the same real-world entity -
    the exact race apply_extraction_result's own docstring used to
    document as an accepted, unresolved gap.

    canonicalize is patched to sleep briefly right after computing its
    decision - GIL/thread scheduling alone doesn't reliably interleave
    two tight, no-I/O method calls in-process, so without an artificial
    pause here this test can pass even when the lock fix is missing
    (verified: it did, before this patch was added). The pause forces a
    real race window instead of depending on scheduling luck."""
    gm = _make_manager(fake_db)
    real_canonicalize = GraphManager.canonicalize

    def slow_canonicalize(self, *args, **kwargs):
        result = real_canonicalize(self, *args, **kwargs)
        threading.Event().wait(0.05)
        return result

    monkeypatch.setattr(GraphManager, "canonicalize", slow_canonicalize)

    def ingest(paper_id: str):
        extraction = ExtractionResult(
            paper_id=paper_id,
            entities=[
                ExtractedEntity(
                    name="Retrieval-Augmented Generation",
                    type=NodeType.CONCEPT,
                    description="",
                )
            ],
            relations=[],
        )
        gm.apply_extraction_result(extraction, session_id="session-a", owner_uid="owner-1")

    t1 = threading.Thread(target=ingest, args=("paper-one",))
    t2 = threading.Thread(target=ingest, args=("paper-two",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    matching_nodes = [
        node_id
        for node_id, data in gm.graph.nodes(data=True)
        if data.get("name") == "Retrieval-Augmented Generation"
    ]
    assert len(matching_nodes) == 1


def test_apply_extraction_result_paper_node_reingest_adds_the_new_session(fake_db):
    """A paper's node id is deterministic from paper_id alone - re-ingesting
    the same paper_id under a different session (e.g. the same arXiv id
    added twice) must make it visible from BOTH sessions, not steal it
    from the first or leave it invisible to the second - confirmed live
    as a real bug where the second session got nothing."""
    gm = _make_manager(fake_db)
    extraction = ExtractionResult(paper_id="paper-reingest", entities=[], relations=[])

    first = gm.apply_extraction_result(
        extraction, paper_name="Paper Title", session_id="session-a"
    , owner_uid="owner-1")
    gm.apply_extraction_result(extraction, paper_name="Paper Title", session_id="session-b", owner_uid="owner-1")

    assert gm.graph.nodes[first.paper_node_id]["session_ids"] == ["session-a", "session-b"]


def test_relation_endpoint_uses_declared_entity_type(fake_db):
    gm = _make_manager(fake_db)
    extraction = ExtractionResult(
        paper_id="paper/unsafe",
        entities=[
            ExtractedEntity(name="BERT", type=NodeType.MODEL, description="Model")
        ],
        relations=[
            ExtractedRelation(
                source_entity=" bert ",
                source_type=NodeType.MODEL,
                relation=EdgeType.USES,
                target_entity="Attention",
                target_type=NodeType.CONCEPT,
                source_quote="BERT uses attention.",
            )
        ],
    )

    report = gm.apply_extraction_result(extraction, owner_uid="owner-1")

    model_id = report.node_writes[0].node_id
    uses_edge = next(edge for edge in report.edge_writes if edge.relation == "USES")
    assert uses_edge.source_id == model_id
    assert "/" not in report.paper_node_id


def test_ambiguous_relation_endpoint_is_skipped_not_paper_fatal(fake_db):
    """An ambiguous untyped relation endpoint used to raise ValueError
    uncaught, aborting apply_extraction_result mid-write - by that point
    other nodes in the same paper are already durably committed to
    Firestore via add_node, so the exception discarded the rest of the
    paper's relations with no report of what already landed. Must be
    isolated to just the one ambiguous relation, the same per-unit
    resilience already applied to per-window extraction failures in
    gemini_extractor.py."""
    gm = _make_manager(fake_db)
    extraction = ExtractionResult(
        paper_id="paper-1",
        entities=[
            ExtractedEntity(
                name="Attention", type=NodeType.METHOD, description="Method"
            ),
            ExtractedEntity(
                name="Attention", type=NodeType.CONCEPT, description="Concept"
            ),
            ExtractedEntity(
                name="Reasoning", type=NodeType.CONCEPT, description="Concept"
            ),
        ],
        relations=[
            ExtractedRelation(
                source_entity="Attention",
                relation=EdgeType.SUPPORTS,
                target_entity="Reasoning",
                source_quote="Attention supports reasoning.",
            ),
            ExtractedRelation(
                source_entity="Attention",
                source_type=NodeType.CONCEPT,
                relation=EdgeType.SUPPORTS,
                target_entity="Reasoning",
                target_type=NodeType.CONCEPT,
                source_quote="Attention as a concept supports reasoning.",
            ),
        ],
    )

    report = gm.apply_extraction_result(extraction, owner_uid="owner-1")

    # The ambiguous relation (untyped "Attention" matches two nodes) is
    # skipped and counted, not raised.
    assert report.skipped_relations == 1
    # The second, unambiguous (typed) relation still gets written - one
    # bad relation doesn't take the rest of the paper down with it.
    assert len(report.edge_writes) == 4  # three MENTIONS edges plus SUPPORTS
    assert sum(edge.relation == "SUPPORTS" for edge in report.edge_writes) == 1
    assert len(report.node_writes) == 3


def test_search_nodes_excludes_merged_alias_nodes(fake_db):
    """After resolve_alias merges an alias into a canonical node, the alias
    must stop showing up as its own independent search hit - otherwise a
    query that already got disambiguated once keeps looking ambiguous
    forever, since search_nodes has no other way to know the two node ids
    now refer to the same real-world entity."""
    gm = _make_manager(fake_db)
    canonical = _node("Fine-Tuning")
    alias = _node("Parameter Tuning")
    gm.add_node(canonical)
    gm.add_node(alias)
    assert len(gm.search_nodes("fine tuning parameter", min_score=0.0)) == 2

    gm.resolve_alias(canonical.id, alias.id, owner_uid="owner-1")

    hits = gm.search_nodes("fine tuning parameter", min_score=0.0)
    assert [hit.node_id for hit in hits] == [canonical.id]


def test_canonicalize_exact_match_excludes_merged_alias_nodes(fake_db):
    """A merged-away alias node's name must not win canonicalize()'s exact-
    string-match pass - otherwise a later entity matching that name gets
    routed to the dead alias instead of the real canonical node, the same
    class of bug search_nodes had before it gained this exclusion."""
    gm = _make_manager(fake_db)
    canonical = _node("Fine-Tuning")
    alias = _node("Parameter Tuning")
    gm.add_node(canonical)
    gm.add_node(alias)
    gm.resolve_alias(canonical.id, alias.id, owner_uid="owner-1")

    result = gm.canonicalize("Parameter Tuning", owner_uid="owner-1")

    assert result.decision == "new"  # not auto_merge against the dead alias


def test_canonicalize_embedding_match_excludes_merged_alias_nodes(fake_db):
    gm = _make_manager(fake_db)
    canonical = _node("LLM Agent", embedding=[1.0, 0.0])
    alias = _node("Language Model Agent", embedding=[1.0, 0.0])
    gm.add_node(canonical)
    gm.add_node(alias)
    gm.resolve_alias(canonical.id, alias.id, owner_uid="owner-1")

    result = gm.canonicalize("Autonomous Agent", embedding=[1.0, 0.0], owner_uid="owner-1")

    # The alias had a perfect-similarity embedding but is merged away -
    # canonicalize must not return it as the match.
    assert result.matched_node_id != alias.id
    assert result.matched_node_id == canonical.id


def test_get_incident_edges_includes_merged_aliases_edges(fake_db):
    """After a merge, the alias's own real extracted edges (which
    resolve_alias never moves or copies onto the canonical node) must
    still be reachable through the canonical node - otherwise they become
    permanently invisible to graph evidence once search_nodes stops
    returning the alias as its own hit."""
    gm = _make_manager(fake_db)
    canonical = _node("GPT-4")
    alias = _node("GPT4")
    other = _node("Some Benchmark")
    gm.add_node(canonical)
    gm.add_node(alias)
    gm.add_node(other)
    gm.add_edge(
        Edge(
            id=str(uuid.uuid4()),
            source_id=alias.id,
            target_id=other.id,
            type=EdgeType.EVALUATES_ON,
            provenance=ProvenanceTag.EXTRACTED,
            source_quote="GPT4 is evaluated on Some Benchmark.",
        )
    )

    gm.resolve_alias(canonical.id, alias.id, owner_uid="owner-1")

    edges = gm.get_incident_edges(canonical.id)
    assert len(edges) == 1
    assert edges[0].source_id == alias.id
    assert edges[0].target_id == other.id
    # The SAME_AS merge marker itself is not useful evidence and must not
    # be surfaced as if it were a real extracted relation.
    assert all(edge.relation != "SAME_AS" for edge in edges)


def test_search_nodes_still_returns_pairs_marked_distinct(fake_db):
    """distinct=True must not exclude a node from search - only an actual
    merge (SAME_AS edge) does. Two nodes correctly marked as different
    things should both keep showing up."""
    gm = _make_manager(fake_db)
    a, b = _node("Concept A"), _node("Concept B")
    gm.add_node(a)
    gm.add_node(b)

    gm.resolve_alias(a.id, b.id, owner_uid="owner-1", distinct=True)

    hits = gm.search_nodes("concept", min_score=0.0)
    assert {hit.node_id for hit in hits} == {a.id, b.id}


def test_needs_clarification_does_not_reask_a_pair_already_marked_distinct(fake_db):
    """If the exact (candidate, provisional) pair apply_extraction_result
    is about to ask a question for is already recorded as known-distinct
    (e.g. a person already answered "no, genuinely different" for it in an
    earlier attempt that didn't complete for unrelated reasons), it must
    not register a duplicate identical question - resolve_alias's own
    docstring promises the same question isn't asked again."""
    gm = _make_manager(fake_db)
    existing = _node("Fine-Tuning", embedding=[1.0, 0.0])
    gm.add_node(existing)
    orchestrator = ClarificationOrchestrator(graph_manager=gm)

    extraction = ExtractionResult(
        paper_id="paper-1",
        entities=[
            ExtractedEntity(
                name="Parameter Tuning",
                type=NodeType.CONCEPT,
                description="A tuning approach",
            )
        ],
        relations=[],
    )
    embedding_fn = lambda entity: [0.8, 0.6]  # needs_clarification band

    # Pre-seed _known_distinct with exactly the pair this extraction is
    # about to land on - the same deterministic id apply_extraction_result
    # itself would compute for this paper_id/type/name.
    provisional_id = gm._stable_node_id(
        "paper-1", "Parameter Tuning", NodeType.CONCEPT
    )
    gm._known_distinct.add(tuple(sorted((existing.id, provisional_id))))

    report = gm.apply_extraction_result(
        extraction, embedding_fn=embedding_fn, clarification=orchestrator
    , owner_uid="owner-1")

    # The node itself is still created (canonicalize() still says
    # needs_clarification) - only the question is suppressed.
    assert report.node_writes[0].node_id == provisional_id
    assert report.node_writes[0].decision == "needs_clarification"
    assert orchestrator.pending() == []


def test_needs_clarification_still_creates_node_and_registers_question(fake_db):
    """The extraction-side clarification hook: a middle-band canonicalization
    match must still create the provisional node (so the batch never
    stalls waiting on a person) but also register a question when an
    orchestrator is passed in."""
    gm = _make_manager(fake_db)
    existing = _node("Fine-Tuning", embedding=[1.0, 0.0])
    gm.add_node(existing)
    orchestrator = ClarificationOrchestrator(graph_manager=gm)

    def embedding_fn(entity):
        # cosine similarity to "Fine-Tuning" = 0.8, in the needs_clarification band
        return [0.8, 0.6]

    extraction = ExtractionResult(
        paper_id="paper-1",
        entities=[
            ExtractedEntity(
                name="Parameter Tuning",
                type=NodeType.CONCEPT,
                description="A tuning approach",
            )
        ],
        relations=[],
    )

    report = gm.apply_extraction_result(
        extraction, embedding_fn=embedding_fn, clarification=orchestrator
    , owner_uid="owner-1")

    provisional_id = report.node_writes[0].node_id
    assert provisional_id != existing.id
    assert provisional_id in gm.graph  # node was still created, batch didn't stall

    pending = orchestrator.pending()
    assert len(pending) == 1
    assert pending[0].kind == "entity_merge"
    assert pending[0].provisional_node_id == provisional_id
    assert pending[0].candidate_node_id == existing.id


def test_needs_clarification_without_orchestrator_behaves_as_before(fake_db):
    """clarification is optional - omitting it must not change behavior for
    existing callers (no question registered, no error)."""
    gm = _make_manager(fake_db)
    existing = _node("Fine-Tuning", embedding=[1.0, 0.0])
    gm.add_node(existing)

    extraction = ExtractionResult(
        paper_id="paper-1",
        entities=[
            ExtractedEntity(
                name="Parameter Tuning",
                type=NodeType.CONCEPT,
                description="A tuning approach",
            )
        ],
        relations=[],
    )

    report = gm.apply_extraction_result(
        extraction, embedding_fn=lambda entity: [0.8, 0.6]
    , owner_uid="owner-1")

    assert report.node_writes[0].node_id != existing.id
    assert report.node_writes[0].decision == "needs_clarification"


def test_extracted_paper_entity_reuses_source_paper_node(fake_db):
    gm = _make_manager(fake_db)
    extraction = ExtractionResult(
        paper_id="paper-1",
        entities=[
            ExtractedEntity(
                name="Paper Title", type=NodeType.PAPER, description="Paper"
            ),
            ExtractedEntity(
                name="Method X", type=NodeType.METHOD, description="Method"
            ),
        ],
        relations=[
            ExtractedRelation(
                source_entity="Paper Title",
                source_type=NodeType.PAPER,
                relation=EdgeType.PROPOSES,
                target_entity="Method X",
                target_type=NodeType.METHOD,
                source_quote="Paper Title proposes Method X.",
            )
        ],
    )

    report = gm.apply_extraction_result(extraction, paper_name="Paper Title", owner_uid="owner-1")

    assert gm.graph.number_of_nodes() == 2
    assert report.node_writes[0].node_id == report.paper_node_id
    assert report.node_writes[0].reused_existing_node is True
    assert report.edge_writes[0].source_id == report.paper_node_id


def test_concurrent_search_and_mutation_does_not_crash(fake_db):
    """networkx's MultiDiGraph is not thread-safe - the FastAPI service
    runs sync route handlers concurrently against one shared GraphManager
    instance, so a concurrent add_node during search_nodes'
    self.graph.nodes(data=True) iteration could previously raise
    "RuntimeError: dictionary changed size during iteration" and turn a
    normal request into a 500. Hammers both concurrently and asserts no
    thread raised."""
    import threading

    gm = _make_manager(fake_db)
    for i in range(20):
        gm.add_node(_node(f"Seed Node {i}"))

    errors: list[Exception] = []
    stop = threading.Event()

    def searcher() -> None:
        while not stop.is_set():
            try:
                gm.search_nodes("seed node", min_score=0.0)
            except Exception as exc:  # pragma: no cover - failure path only
                errors.append(exc)
                return

    def writer() -> None:
        for i in range(200):
            try:
                gm.add_node(_node(f"Concurrent Node {i}"))
            except Exception as exc:  # pragma: no cover - failure path only
                errors.append(exc)
                return

    threads = [threading.Thread(target=searcher) for _ in range(4)]
    writer_thread = threading.Thread(target=writer)
    for t in threads:
        t.start()
    writer_thread.start()
    writer_thread.join()
    stop.set()
    for t in threads:
        t.join()

    assert not errors, f"concurrent access raised: {errors}"


def test_remove_by_session_deletes_owned_nodes_and_edges_live_and_in_firestore(
    fake_db,
):
    gm = _make_manager(fake_db)
    owned = _node("Session Node")
    owned.session_id = "session-a"
    other = _node("Other Session Node")
    gm.add_node(owned)
    gm.add_node(other)
    edge = Edge(
        id=str(uuid.uuid4()),
        source_id=owned.id,
        target_id=other.id,
        type=EdgeType.USES,
        provenance=ProvenanceTag.EXTRACTED,
        session_id="session-a",
    )
    gm.add_edge(edge)

    removed = gm.remove_by_session("session-a")

    assert removed == {owned.id}
    assert owned.id not in gm.graph
    assert other.id in gm.graph
    assert gm.graph.number_of_edges() == 0
    # A fresh manager rehydrating from the same fake_db must not resurrect
    # the removed node/edge - proves the Firestore docs were deleted too,
    # not just the in-memory graph.
    rehydrated = _make_manager(fake_db)
    assert owned.id not in rehydrated.graph
    assert other.id in rehydrated.graph
    assert rehydrated.graph.number_of_edges() == 0


def test_remove_by_session_removes_a_different_sessions_edge_to_a_dying_node(
    fake_db,
):
    """An edge that belongs to a *different* session but touches a node
    being removed must also go - otherwise it dangles, and worse,
    resurrects the deleted node as a bare stub on the next rehydrate
    (add_edge auto-creates missing endpoint nodes)."""
    gm = _make_manager(fake_db)
    dying = _node("Dying Node")
    dying.session_id = "session-a"
    survivor = _node("Survivor Node")
    survivor.session_id = "session-b"
    gm.add_node(dying)
    gm.add_node(survivor)
    edge = Edge(
        id=str(uuid.uuid4()),
        source_id=survivor.id,
        target_id=dying.id,
        type=EdgeType.USES,
        provenance=ProvenanceTag.EXTRACTED,
        session_id="session-b",
    )
    gm.add_edge(edge)

    gm.remove_by_session("session-a")

    assert survivor.id in gm.graph
    assert dying.id not in gm.graph
    assert gm.graph.number_of_edges() == 0
    rehydrated = _make_manager(fake_db)
    assert survivor.id in rehydrated.graph
    assert dying.id not in rehydrated.graph


def test_remove_by_session_strips_membership_but_keeps_a_node_shared_with_another_session(
    fake_db,
):
    """A node genuinely shared across two sessions (e.g. an entity reused
    via canonicalization from an earlier ingest) must survive one of
    those sessions being deleted - only removed once its last owning
    session goes away. Same for an edge shared the same way."""
    gm = _make_manager(fake_db)
    shared = _node("Shared Node")
    gm.add_node(shared)
    gm.add_node(Node(**{**shared.model_dump(), "session_id": "session-a"}))
    gm.add_node(Node(**{**shared.model_dump(), "session_id": "session-b"}))
    other = _node("Other Node")
    other.session_id = "session-a"
    gm.add_node(other)
    edge = Edge(
        id=str(uuid.uuid4()),
        source_id=shared.id,
        target_id=other.id,
        type=EdgeType.USES,
        provenance=ProvenanceTag.EXTRACTED,
    )
    gm.add_edge(edge)
    gm.add_edge(Edge(**{**edge.model_dump(), "session_id": "session-a"}))
    gm.add_edge(Edge(**{**edge.model_dump(), "session_id": "session-b"}))

    removed_first = gm.remove_by_session("session-a")

    # "other" was only ever session-a's, so it (and the now-endpoint-less
    # edge) are gone - but the shared node survives, just with session-a
    # stripped from its membership.
    assert removed_first == {other.id}
    assert shared.id in gm.graph
    assert gm.graph.nodes[shared.id]["session_ids"] == ["session-b"]
    assert other.id not in gm.graph

    removed_second = gm.remove_by_session("session-b")

    assert removed_second == {shared.id}
    assert shared.id not in gm.graph
