"""networkx-as-engine, Firestore-as-durability graph manager.

CQRS pattern: the in-memory networkx graph is the fast command/query path
for a live session; Firestore is the durable, eventually-consistent log
behind it. Tools are split read-only vs write so an LLM-issued tool call
can't trigger an unintended write through a read-path agent.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx

try:
    from google.cloud import firestore
except ImportError:  # pragma: no cover - exercised only outside test doubles
    firestore = None  # type: ignore[assignment]

from agent.schema import (
    Edge,
    EdgeType,
    ExtractionResult,
    ExtractedEntity,
    Node,
    NodeType,
    ProvenanceTag,
)

# Two-tier canonicalization thresholds. Above HIGH: auto-merge silently.
# Below LOW: treat as genuinely new. In between: the concrete trigger
# condition for a Clarification Orchestrator question.
CANONICALIZATION_HIGH = 0.92
CANONICALIZATION_LOW = 0.75


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


@dataclass
class CanonicalizationResult:
    decision: str  # "auto_merge" | "new" | "needs_clarification"
    matched_node_id: str | None = None
    score: float | None = None


@dataclass
class NodeWriteResult:
    entity_name: str
    node_id: str
    decision: str
    score: float | None = None
    reused_existing_node: bool = False


@dataclass
class EdgeWriteResult:
    relation: str
    edge_id: str
    source_id: str
    target_id: str


@dataclass
class GraphIngestionReport:
    paper_id: str
    paper_node_id: str
    node_writes: list[NodeWriteResult]
    edge_writes: list[EdgeWriteResult]


class GraphManager:
    def __init__(
        self,
        project_id: str,
        database: str = "(default)",
        db_client: Any | None = None,
    ):
        self.graph = nx.MultiDiGraph()
        if db_client is not None:
            self._db = db_client
        else:
            if firestore is None:
                raise ModuleNotFoundError(
                    "google.cloud.firestore is not installed; inject db_client for tests"
                )
            self._db = firestore.Client(project=project_id, database=database)
        self._known_distinct: set[tuple[str, str]] = set()
        self._rehydrate()

    def _rehydrate(self) -> None:
        """Load existing nodes/edges from Firestore into the in-memory graph
        (session resume on Cloud Run cold start)."""
        for doc in self._db.collection("nodes").stream():
            data = doc.to_dict()
            self.graph.add_node(doc.id, **data)
        for doc in self._db.collection("edges").stream():
            data = doc.to_dict()
            self.graph.add_edge(
                data["source_id"], data["target_id"], key=doc.id, **data
            )

    # ---- Write tools ----

    def add_node(self, node: Node) -> str:
        """Idempotent upsert on node.id — safe to retry."""
        self._db.collection("nodes").document(node.id).set(
            node.model_dump(mode="json"), merge=True
        )
        self.graph.add_node(node.id, **node.model_dump(mode="json"))
        return node.id

    def add_edge(self, edge: Edge) -> str:
        """Idempotent upsert on edge.id — safe to retry."""
        self._db.collection("edges").document(edge.id).set(
            edge.model_dump(mode="json"), merge=True
        )
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.id,
            **edge.model_dump(mode="json"),
        )
        return edge.id

    def resolve_alias(
        self, canonical_id: str, alias_id: str, distinct: bool = False
    ) -> None:
        """Apply a user correction on an ambiguous canonicalization match.

        distinct=False merges alias_id into canonical_id via a SAME_AS edge
        (INFERRED — no single source quote by definition). distinct=True
        records the pair as known-distinct so the same question isn't asked
        again later in the batch.
        """
        if distinct:
            self._known_distinct.add(tuple(sorted((canonical_id, alias_id))))
            return
        edge = Edge(
            id=str(uuid.uuid4()),
            source_id=alias_id,
            target_id=canonical_id,
            type=EdgeType.SAME_AS,
            provenance=ProvenanceTag.INFERRED,
        )
        self.add_edge(edge)

    # ---- Read-only tools ----

    def get_neighbors(self, node_id: str) -> list[str]:
        if node_id not in self.graph:
            return []
        return list(self.graph.successors(node_id)) + list(
            self.graph.predecessors(node_id)
        )

    def _stable_node_id(
        self, paper_id: str, name: str, node_type: NodeType
    ) -> str:
        normalized = _normalize_name(name)
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL, f"node:{paper_id}:{node_type.value}:{normalized}"
            )
        )

    def _stable_edge_id(
        self,
        paper_id: str,
        source_id: str,
        target_id: str,
        relation: EdgeType,
        source_quote: str,
    ) -> str:
        quote_key = re.sub(r"\s+", " ", source_quote).strip()
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"edge:{paper_id}:{source_id}:{target_id}:{relation.value}:{quote_key}",
            )
        )

    def find_sparse_pairs(
        self, node_type: NodeType | None = None, limit: int = 10
    ) -> list[tuple[str, str]]:
        """Candidate pairs with no direct edge but at least one common
        neighbor — topology decides, not LLM guessing."""
        candidates = [
            n
            for n, data in self.graph.nodes(data=True)
            if node_type is None or data.get("type") == node_type.value
        ]
        undirected = self.graph.to_undirected()
        pairs: list[tuple[str, str, int]] = []
        for i, a in enumerate(candidates):
            for b in candidates[i + 1 :]:
                if undirected.has_edge(a, b):
                    continue
                if tuple(sorted((a, b))) in self._known_distinct:
                    continue
                common = len(
                    list(nx.common_neighbors(undirected, a, b))
                )
                if common > 0:
                    pairs.append((a, b, common))
        pairs.sort(key=lambda p: p[2], reverse=True)
        return [(a, b) for a, b, _ in pairs[:limit]]

    # ---- Canonicalization ----

    def canonicalize(
        self,
        name: str,
        embedding: list[float] | None = None,
        node_type: NodeType | None = None,
    ) -> CanonicalizationResult:
        """Two-tier match: cheap string match first, then embedding
        similarity. Three-way routing on the result."""
        normalized = _normalize_name(name)
        for node_id, data in self.graph.nodes(data=True):
            if node_type is not None and data.get("type") != node_type.value:
                continue
            if _normalize_name(data.get("name", "")) == normalized:
                return CanonicalizationResult("auto_merge", node_id, 1.0)

        if embedding is None:
            return CanonicalizationResult("new")

        best_id, best_score = None, 0.0
        for node_id, data in self.graph.nodes(data=True):
            if node_type is not None and data.get("type") != node_type.value:
                continue
            candidate_embedding = data.get("entity_embedding")
            if not candidate_embedding:
                continue
            score = _cosine_similarity(embedding, candidate_embedding)
            if score > best_score:
                best_id, best_score = node_id, score

        if best_score >= CANONICALIZATION_HIGH:
            return CanonicalizationResult("auto_merge", best_id, best_score)
        if best_score >= CANONICALIZATION_LOW:
            return CanonicalizationResult("needs_clarification", best_id, best_score)
        return CanonicalizationResult("new", best_id, best_score)

    def apply_extraction_result(
        self,
        extraction: ExtractionResult,
        *,
        paper_name: str | None = None,
        embedding_fn: Callable[[ExtractedEntity], list[float] | None] | None = None,
    ) -> GraphIngestionReport:
        """Persist structured extraction output with stable, retry-safe IDs."""

        # paper_id is caller-supplied (an ADK tool call could eventually
        # drive this per the module docstring), so it's hashed the same way
        # as every other id here rather than used raw as a graph/Firestore id.
        paper_node = Node(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"node:paper:{extraction.paper_id}")),
            type=NodeType.PAPER,
            name=paper_name or extraction.paper_id,
            description="Source paper",
        )
        self.add_node(paper_node)

        entity_to_node_id: dict[str, str] = {}
        node_writes: list[NodeWriteResult] = []
        for entity in extraction.entities:
            embedding = embedding_fn(entity) if embedding_fn is not None else None
            canonical = self.canonicalize(
                entity.name, embedding=embedding, node_type=entity.type
            )
            if canonical.decision == "auto_merge" and canonical.matched_node_id:
                node_id = canonical.matched_node_id
                reused = True
            else:
                node_id = self._stable_node_id(
                    extraction.paper_id, entity.name, entity.type
                )
                self.add_node(
                    Node(
                        id=node_id,
                        type=entity.type,
                        name=entity.name,
                        description=entity.description,
                        entity_embedding=embedding,
                    )
                )
                reused = False
            # Relations only carry entity *names* (see ExtractedRelation), not
            # types, so this map can't fully disambiguate two same-name,
            # different-type entities from a single extraction. Keep the
            # first occurrence deterministically rather than letting a later
            # duplicate silently overwrite it depending on iteration order.
            entity_to_node_id.setdefault(entity.name, node_id)
            node_writes.append(
                NodeWriteResult(
                    entity_name=entity.name,
                    node_id=node_id,
                    decision=canonical.decision,
                    score=canonical.score,
                    reused_existing_node=reused,
                )
            )

        edge_writes: list[EdgeWriteResult] = []
        for relation in extraction.relations:
            source_id = self._resolve_relation_endpoint(
                extraction.paper_id, relation.source_entity, entity_to_node_id
            )
            target_id = self._resolve_relation_endpoint(
                extraction.paper_id, relation.target_entity, entity_to_node_id
            )
            edge = Edge(
                id=self._stable_edge_id(
                    extraction.paper_id,
                    source_id,
                    target_id,
                    relation.relation,
                    relation.source_quote,
                ),
                source_id=source_id,
                target_id=target_id,
                type=relation.relation,
                provenance=ProvenanceTag.EXTRACTED,
                source_paper_id=extraction.paper_id,
                source_section=relation.source_section,
                source_quote=relation.source_quote,
            )
            self.add_edge(edge)
            edge_writes.append(
                EdgeWriteResult(
                    relation=relation.relation.value,
                    edge_id=edge.id,
                    source_id=source_id,
                    target_id=target_id,
                )
            )

        return GraphIngestionReport(
            paper_id=extraction.paper_id,
            paper_node_id=paper_node.id,
            node_writes=node_writes,
            edge_writes=edge_writes,
        )

    def _resolve_relation_endpoint(
        self,
        paper_id: str,
        name: str,
        entity_to_node_id: dict[str, str],
    ) -> str:
        if name in entity_to_node_id:
            return entity_to_node_id[name]

        # No type is known for an implicit relation endpoint (relations only
        # carry names), so search across all node types rather than
        # restricting to CONCEPT, or an existing node of a different type
        # (e.g. a MODEL) could never be matched here.
        canonical = self.canonicalize(name, node_type=None)
        if canonical.decision == "auto_merge" and canonical.matched_node_id:
            return canonical.matched_node_id

        node_id = self._stable_node_id(paper_id, name, NodeType.CONCEPT)
        self.add_node(
            Node(
                id=node_id,
                type=NodeType.CONCEPT,
                name=name,
                description="Implicit relation endpoint from extraction",
            )
        )
        entity_to_node_id[name] = node_id
        return node_id
