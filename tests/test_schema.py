from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schema import EdgeType, ExtractedRelation, NodeType, ProvenanceTag


def test_node_type_has_seven_members():
    assert len(NodeType) == 7


def test_edge_type_has_eight_members():
    assert len(EdgeType) == 8


def test_extracted_relation_requires_source_quote():
    relation = ExtractedRelation(
        source_entity="Paper A",
        relation=EdgeType.PROPOSES,
        target_entity="Method X",
        source_quote="We propose Method X to address...",
    )
    assert relation.source_quote
    assert relation.relation == EdgeType.PROPOSES


def test_extracted_relation_rejects_blank_source_quote():
    with pytest.raises(ValidationError):
        ExtractedRelation(
            source_entity="Paper A",
            relation=EdgeType.PROPOSES,
            target_entity="Method X",
            source_quote="   ",
        )


def test_provenance_tag_values():
    assert ProvenanceTag.EXTRACTED.value == "EXTRACTED"
    assert ProvenanceTag.INFERRED.value == "INFERRED"
