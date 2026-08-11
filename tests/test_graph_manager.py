from __future__ import annotations

import uuid

from agent.graph_manager import GraphManager
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


def test_canonicalize_mismatched_embedding_dims_is_not_a_match(fake_db):
    """_cosine_similarity must reject mismatched-length embeddings instead
    of silently truncating via zip() and returning a false high score."""
    gm = _make_manager(fake_db)
    existing = _node("Transformer", embedding=[1.0, 0.0, 0.0, 0.0])
    gm.add_node(existing)
    result = gm.canonicalize("Something Else", embedding=[1.0, 0.0])
    assert result.decision == "new"
    assert result.score == 0.0


def test_resolve_alias_merge_writes_same_as_edge(fake_db):
    gm = _make_manager(fake_db)
    canonical, alias = _node("LLM Agent"), _node("Language Model Agent")
    gm.add_node(canonical)
    gm.add_node(alias)
    gm.resolve_alias(canonical.id, alias.id)
    edges = list(gm.graph.get_edge_data(alias.id, canonical.id).values())
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

    # paper_node_id is a stable hash of paper_id, not paper_id itself (so an
    # attacker-influenced paper_id can't be used raw as a graph/Firestore
    # id) - assert it's deterministic across calls instead of a fixed string.
    assert first.paper_node_id == second.paper_node_id
    assert len(first.node_writes) == 2
    assert len(first.edge_writes) == 1
    assert gm.graph.number_of_nodes() == 3
    assert gm.graph.number_of_edges() == 1
    assert second.edge_writes[0].edge_id == first.edge_writes[0].edge_id


def test_paper_node_id_is_not_raw_paper_id(fake_db):
    """paper_id must be hashed, not used raw as a graph/Firestore id."""
    gm = _make_manager(fake_db)
    extraction = ExtractionResult(
        paper_id="paper/with/slashes",
        entities=[],
        relations=[],
        chunks=[],
    )
    report = gm.apply_extraction_result(extraction)
    assert report.paper_node_id != "paper/with/slashes"
    assert "/" not in report.paper_node_id


def test_relation_endpoint_matches_existing_node_of_different_type(fake_db):
    """A relation referencing an entity by name should link to an already-
    existing node even when that node isn't a CONCEPT, instead of minting a
    disconnected duplicate CONCEPT node (see _resolve_relation_endpoint)."""
    gm = _make_manager(fake_db)
    first = ExtractionResult(
        paper_id="paper-1",
        entities=[
            ExtractedEntity(name="BERT", type=NodeType.MODEL, description="A model"),
        ],
        relations=[],
        chunks=[],
    )
    gm.apply_extraction_result(first)
    bert_node_id = next(
        n for n, data in gm.graph.nodes(data=True) if data["name"] == "BERT"
    )

    second = ExtractionResult(
        paper_id="paper-2",
        entities=[],
        relations=[
            ExtractedRelation(
                source_entity="BERT",
                relation=EdgeType.USES,
                target_entity="Attention",
                source_quote="BERT uses attention.",
            )
        ],
        chunks=[],
    )
    report = gm.apply_extraction_result(second)

    assert report.edge_writes[0].source_id == bert_node_id
    assert gm.graph.nodes[bert_node_id]["type"] == NodeType.MODEL.value


def test_same_name_different_type_entities_keep_first_mapping(fake_db):
    """Two entities sharing a name but differing in type must not silently
    overwrite each other's id in entity_to_node_id; behavior should be
    deterministic (first occurrence wins) rather than iteration-order-
    dependent."""
    gm = _make_manager(fake_db)
    extraction = ExtractionResult(
        paper_id="paper-1",
        entities=[
            ExtractedEntity(name="Attention", type=NodeType.METHOD, description="A"),
            ExtractedEntity(name="Attention", type=NodeType.CONCEPT, description="B"),
        ],
        relations=[
            ExtractedRelation(
                source_entity="Attention",
                relation=EdgeType.SUPPORTS,
                target_entity="Attention",
                source_quote="Attention supports attention.",
            )
        ],
        chunks=[],
    )
    report = gm.apply_extraction_result(extraction)

    assert gm.graph.number_of_nodes() == 3  # paper + 2 distinct Attention nodes
    method_node_id = next(
        n
        for n, data in gm.graph.nodes(data=True)
        if data["name"] == "Attention" and data["type"] == NodeType.METHOD.value
    )
    assert report.edge_writes[0].source_id == method_node_id
    assert report.edge_writes[0].target_id == method_node_id
