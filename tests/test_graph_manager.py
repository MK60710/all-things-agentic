from __future__ import annotations

import uuid

import pytest

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


def _node(name: str, embedding: list[float] | None = None) -> Node:
    return Node(
        id=str(uuid.uuid4()),
        type=NodeType.CONCEPT,
        name=name,
        entity_embedding=embedding,
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


def test_get_node_returns_none_for_unknown_id(fake_db):
    gm = _make_manager(fake_db)
    assert gm.get_node("does-not-exist") is None


def test_canonicalize_string_match_auto_merges(fake_db):
    gm = _make_manager(fake_db)
    existing = _node("Chain of Thought")
    gm.add_node(existing)
    result = gm.canonicalize("chain of thought")  # casing differs
    assert result.decision == "auto_merge"
    assert result.matched_node_id == existing.id


def test_canonicalize_high_similarity_auto_merges(fake_db):
    gm = _make_manager(fake_db)
    existing = _node("Few-Shot Prompting", embedding=[1.0, 0.0])
    gm.add_node(existing)
    result = gm.canonicalize("In-Context Learning", embedding=[1.0, 0.0])
    assert result.decision == "auto_merge"


def test_canonicalize_middle_band_needs_clarification(fake_db):
    gm = _make_manager(fake_db)
    existing = _node("Fine-Tuning", embedding=[1.0, 0.0])
    gm.add_node(existing)
    # cosine similarity = 0.8, between LOW (0.75) and HIGH (0.92)
    result = gm.canonicalize("Parameter Tuning", embedding=[0.8, 0.6])
    assert result.decision == "needs_clarification"
    assert result.matched_node_id == existing.id


def test_canonicalize_low_similarity_is_new(fake_db):
    gm = _make_manager(fake_db)
    existing = _node("Transformer", embedding=[1.0, 0.0])
    gm.add_node(existing)
    result = gm.canonicalize("Reinforcement Learning", embedding=[0.0, 1.0])
    assert result.decision == "new"


def test_cosine_similarity_rejects_dimension_mismatch():
    assert _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0


def test_resolve_alias_merge_writes_same_as_edge(fake_db):
    gm = _make_manager(fake_db)
    canonical, alias = _node("LLM Agent"), _node("Language Model Agent")
    gm.add_node(canonical)
    gm.add_node(alias)
    gm.resolve_alias(canonical.id, alias.id)
    gm.resolve_alias(canonical.id, alias.id)
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
    gm.resolve_alias(a.id, b.id, distinct=True)
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

    first = gm.apply_extraction_result(extraction, paper_name="Paper Title")
    second = gm.apply_extraction_result(extraction, paper_name="Paper Title")

    assert first.paper_node_id != "paper-1"
    assert len(first.node_writes) == 2
    assert len(first.edge_writes) == 1
    assert gm.graph.number_of_nodes() == 3
    assert gm.graph.number_of_edges() == 1
    assert second.edge_writes[0].edge_id == first.edge_writes[0].edge_id


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

    report = gm.apply_extraction_result(extraction)

    model_id = report.node_writes[0].node_id
    assert report.edge_writes[0].source_id == model_id
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

    report = gm.apply_extraction_result(extraction)

    # The ambiguous relation (untyped "Attention" matches two nodes) is
    # skipped and counted, not raised.
    assert report.skipped_relations == 1
    # The second, unambiguous (typed) relation still gets written - one
    # bad relation doesn't take the rest of the paper down with it.
    assert len(report.edge_writes) == 1
    assert len(report.node_writes) == 3


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

    report = gm.apply_extraction_result(extraction, paper_name="Paper Title")

    assert gm.graph.number_of_nodes() == 2
    assert report.node_writes[0].node_id == report.paper_node_id
    assert report.node_writes[0].reused_existing_node is True
    assert report.edge_writes[0].source_id == report.paper_node_id
