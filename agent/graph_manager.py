"""networkx-as-engine, Firestore-as-durability graph manager.

CQRS pattern: the in-memory networkx graph is the fast command/query path
for a live session; Firestore is the durable, eventually-consistent log
behind it. Tools are split read-only vs write so an LLM-issued tool call
can't trigger an unintended write through a read-path agent.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import networkx as nx

if TYPE_CHECKING:
    from agent.clarification_orchestrator import ClarificationOrchestrator

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
from agent.text_utils import search_tokens

logger = logging.getLogger(__name__)

# Two-tier canonicalization thresholds. Above HIGH: auto-merge silently.
# Below LOW: treat as genuinely new. In between: the concrete trigger
# condition for a Clarification Orchestrator question.
CANONICALIZATION_HIGH = 0.92
CANONICALIZATION_LOW = 0.75


def _session_ids(data: dict) -> list[str]:
    """A node/edge's real session membership, read from the persisted
    "session_ids" list this module writes (see GraphManager.add_node/
    add_edge). Falls back to the old single "session_id" field for data
    written before multi-session membership existed - this is what lets
    ~90% of the graph's pre-existing legacy-tagged nodes keep working
    exactly as before with no migration script, and self-heal into the
    new list-based model for free the next time anything re-touches
    them."""
    session_ids = data.get("session_ids")
    if session_ids:
        return session_ids
    legacy = data.get("session_id")
    return [legacy] if legacy else []


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
    # NFKC first: folds Unicode compatibility variants (e.g. the
    # Mathematical Alphanumeric Symbols block - "𝑄" U+1D444 italic-math
    # capital Q) down to their plain ASCII letters before the regex strips
    # anything outside literal a-z0-9. Without this, an entity name that
    # differs from an existing node only by a math-italic codepoint drops
    # that codepoint entirely instead of matching it, missing the exact-
    # match tier and landing in the fuzzy needs_clarification band instead.
    normalized = unicodedata.normalize("NFKC", name)
    return re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()


_ABBREVIATION_RE = re.compile(r"^(.+?)\s*\(([^()]+)\)$")


def _abbreviation_of(name: str) -> str | None:
    """If `name` is "Full Form (Abbrev)", return normalize(Abbrev) - the
    bare-abbreviation side of the pair. Used only to auto-match a spelled-
    out name against another entity whose name IS that bare abbreviation
    (e.g. "moral ODD" against "moral operational design domain (moral
    ODD)"). Deliberately not used to match two parenthetical forms against
    each other - two names that each carry their own distinguishing prefix
    (e.g. "explicit moral ODD (...)" vs "moral ODD (...)") stay a real
    needs_clarification question instead of being silently merged."""
    match = _ABBREVIATION_RE.match(name.strip())
    if match is None:
        return None
    return _normalize_name(match.group(2))


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
class NodeSearchHit:
    node_id: str
    score: float
    name: str
    type: str
    description: str


@dataclass
class IncidentEdge:
    edge_id: str
    source_id: str
    target_id: str
    source_name: str
    target_name: str
    relation: str
    source_quote: str
    source_paper_id: str | None
    source_section: str | None


@dataclass
class SessionGraphNode:
    node_id: str
    name: str
    type: str
    description: str


@dataclass
class SessionGraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str


@dataclass
class SessionGraphExport:
    nodes: list[SessionGraphNode]
    edges: list[SessionGraphEdge]


@dataclass
class GraphIngestionReport:
    paper_id: str
    paper_node_id: str
    node_writes: list[NodeWriteResult]
    edge_writes: list[EdgeWriteResult]
    # Relations whose endpoint couldn't be resolved (e.g. an ambiguous
    # untyped name matching more than one existing node) and were skipped
    # rather than raising - see apply_extraction_result. Default 0 so
    # existing callers/tests don't need to know about this field.
    skipped_relations: int = 0


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
        self._node_token_cache: dict[str, set[str]] = {}
        # networkx's MultiDiGraph is not thread-safe - the FastAPI service
        # (service/) runs sync route handlers concurrently against one
        # shared GraphManager instance, so a concurrent write during a
        # read's iteration (e.g. self.graph.nodes(data=True) in
        # search_nodes/canonicalize) can raise "dictionary changed size
        # during iteration" and turn a normal request into a 500.
        # Reentrant because apply_extraction_result calls add_node/
        # canonicalize internally, from the same thread, while already
        # holding this lock.
        self._lock = threading.RLock()
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
        for doc in self._db.collection("canonicalization_corrections").stream():
            data = doc.to_dict()
            if data.get("decision") == "distinct":
                self._known_distinct.add(
                    tuple(sorted((data["first_node_id"], data["second_node_id"])))
                )

    # ---- Write tools ----

    def add_node(self, node: Node) -> str:
        """Idempotent upsert on node.id — safe to retry.

        Also session-accumulating: a node reused across multiple ingests
        (via canonicalization, or the same paper_id re-ingested into a
        different session) belongs to every session that has genuinely
        used it, not just whichever session touched it first or last -
        confirmed live as a real bug where re-ingesting an already-known
        paper into a new session left it invisible there. The merged
        list is computed once here so every caller (paper nodes, entity
        nodes, implicit relation-endpoint nodes) gets this for free.
        """
        with self._lock:
            existing = self.graph.nodes.get(node.id)
            merged_session_ids = sorted(
                set(_session_ids(existing or {})) | set(_session_ids(node.model_dump(mode="json")))
            )
            payload = node.model_dump(mode="json")
            payload["session_ids"] = merged_session_ids
            self._db.collection("nodes").document(node.id).set(payload, merge=True)
            self.graph.add_node(node.id, **payload)
            # add_node is an upsert (name/description can change on retry
            # with richer data) - drop any stale cached tokens so
            # search_nodes below re-tokenizes from the current data on
            # next lookup.
            self._node_token_cache.pop(node.id, None)
            return node.id

    def add_edge(self, edge: Edge) -> str:
        """Idempotent upsert on edge.id — safe to retry. Session-
        accumulating in the same way and for the same reason as
        add_node above."""
        with self._lock:
            existing_edge = self.graph.get_edge_data(
                edge.source_id, edge.target_id, key=edge.id
            )
            merged_session_ids = sorted(
                set(_session_ids(existing_edge or {})) | set(_session_ids(edge.model_dump(mode="json")))
            )
            payload = edge.model_dump(mode="json")
            payload["session_ids"] = merged_session_ids
            self._db.collection("edges").document(edge.id).set(payload, merge=True)
            self.graph.add_edge(
                edge.source_id,
                edge.target_id,
                key=edge.id,
                **payload,
            )
            return edge.id

    def remove_by_session(self, session_id: str) -> set[str]:
        """Removes this session's membership from every node/edge it
        touches - a node/edge still genuinely shared with another
        session survives (just has this session_id stripped from its
        membership list); one that becomes ownerless is fully deleted,
        from both the live in-memory graph and Firestore. Returns the
        ids of nodes that were actually fully deleted, so a caller
        (session delete) can clean up anything else keyed off them,
        like pending clarification questions - a surviving shared node
        must not have its clarification questions removed just because
        one of its owning sessions went away.

        Edges are collected up front, before any removal - both every
        edge touching a node that's about to be fully deleted
        (regardless of which session added that edge - an edge to a
        node that's about to disappear is dead either way, and leaving
        its Firestore doc behind would resurrect the node as a bare,
        data-less stub on the next _rehydrate(), since add_edge
        auto-creates missing endpoint nodes) and every edge that becomes
        ownerless once this session's membership is stripped from it,
        even between nodes it doesn't own (reused via canonicalization,
        so wouldn't otherwise be touched).
        """
        with self._lock:
            touched_node_ids = {
                node_id
                for node_id, data in self.graph.nodes(data=True)
                if session_id in _session_ids(data)
            }

            fully_removed_node_ids: set[str] = set()
            for node_id in touched_node_ids:
                data = self.graph.nodes[node_id]
                remaining = [s for s in _session_ids(data) if s != session_id]
                if remaining:
                    updated = {**data, "session_ids": remaining}
                    self.graph.nodes[node_id]["session_ids"] = remaining
                    self._db.collection("nodes").document(node_id).set(
                        updated, merge=True
                    )
                else:
                    fully_removed_node_ids.add(node_id)

            edges_to_remove: dict[str, tuple[str, str, str]] = {}
            for node_id in fully_removed_node_ids:
                for source, target, key in self.graph.in_edges(node_id, keys=True):
                    edges_to_remove[key] = (source, target, key)
                for source, target, key in self.graph.out_edges(node_id, keys=True):
                    edges_to_remove[key] = (source, target, key)

            edges_to_update: list[tuple[str, str, str, dict]] = []
            for source, target, key, data in self.graph.edges(keys=True, data=True):
                if key in edges_to_remove or session_id not in _session_ids(data):
                    continue
                remaining = [s for s in _session_ids(data) if s != session_id]
                if remaining:
                    edges_to_update.append(
                        (source, target, key, {**data, "session_ids": remaining})
                    )
                else:
                    edges_to_remove[key] = (source, target, key)

            for source, target, key in edges_to_remove.values():
                if self.graph.has_edge(source, target, key):
                    self.graph.remove_edge(source, target, key)
                self._db.collection("edges").document(key).delete()

            for source, target, key, updated in edges_to_update:
                self.graph[source][target][key]["session_ids"] = updated["session_ids"]
                self._db.collection("edges").document(key).set(updated, merge=True)

            for node_id in fully_removed_node_ids:
                if node_id in self.graph:
                    self.graph.remove_node(node_id)
                self._db.collection("nodes").document(node_id).delete()
                self._node_token_cache.pop(node_id, None)

            return fully_removed_node_ids

    def resolve_alias(
        self, canonical_id: str, alias_id: str, distinct: bool = False
    ) -> None:
        """Apply a user correction on an ambiguous canonicalization match.

        distinct=False merges alias_id into canonical_id via a SAME_AS edge
        (INFERRED — no single source quote by definition). distinct=True
        records the pair as known-distinct so the same question isn't asked
        again later in the batch.
        """
        with self._lock:
            if distinct:
                pair = tuple(sorted((canonical_id, alias_id)))
                self._known_distinct.add(pair)
                correction_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"canonicalization:distinct:{pair[0]}:{pair[1]}",
                    )
                )
                self._db.collection("canonicalization_corrections").document(
                    correction_id
                ).set(
                    {
                        "decision": "distinct",
                        "first_node_id": pair[0],
                        "second_node_id": pair[1],
                    },
                    merge=True,
                )
                return
            edge = Edge(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"same_as:{alias_id}:{canonical_id}",
                    )
                ),
                source_id=alias_id,
                target_id=canonical_id,
                type=EdgeType.SAME_AS,
                provenance=ProvenanceTag.INFERRED,
            )
            self.add_edge(edge)

    # ---- Read-only tools ----

    def get_neighbors(self, node_id: str) -> list[str]:
        with self._lock:
            if node_id not in self.graph:
                return []
            return list(self.graph.successors(node_id)) + list(
                self.graph.predecessors(node_id)
            )

    def _node_tokens(self, node_id: str, data: dict[str, Any]) -> set[str]:
        cached = self._node_token_cache.get(node_id)
        if cached is not None:
            return cached
        tokens = search_tokens(f"{data.get('name', '')} {data.get('description', '')}")
        self._node_token_cache[node_id] = tokens
        return tokens

    def _is_merged_alias(self, node_id: str) -> bool:
        """True if node_id has been resolved into another node via
        resolve_alias(distinct=False) - a SAME_AS edge out of it means it's
        superseded by its canonical target, not an independent entity
        anymore."""
        return any(
            data.get("type") == EdgeType.SAME_AS.value
            for _, _, data in self.graph.out_edges(node_id, data=True)
        )

    def search_nodes(
        self, query: str, *, limit: int = 8, min_score: float = 0.0
    ) -> list[NodeSearchHit]:
        """Lexical relevance search over node name+description.

        The read-side counterpart to ChunkIndex.search's scoring/min_score
        pattern in retrieval.py - callers get a real quality gate instead
        of reaching into `.graph` themselves and treating any token overlap
        as a match. Tokenization is cached per node (invalidated in
        add_node) since node text is static between graph mutations and
        this scan is O(number of nodes) per call. Nodes already merged
        into another node (resolve_alias(distinct=False)) are excluded -
        otherwise a resolved entity_merge question keeps resurfacing the
        same ambiguity on every later query, since the alias node would
        still show up as an independent, separately-scored hit. The merge
        check runs after the (cached) token-overlap check, not before -
        it's an uncached edge scan, so only paying it for nodes that
        already cleared the cheap check keeps the common case (no overlap)
        fast.
        """
        query_tokens = search_tokens(query)
        if not query_tokens:
            return []
        with self._lock:
            hits: list[NodeSearchHit] = []
            for node_id, data in self.graph.nodes(data=True):
                node_tokens = self._node_tokens(node_id, data)
                if not node_tokens:
                    continue
                overlap = query_tokens & node_tokens
                if not overlap:
                    continue
                if self._is_merged_alias(node_id):
                    continue
                score = len(overlap) / len(query_tokens)
                if score >= min_score:
                    hits.append(
                        NodeSearchHit(
                            node_id=node_id,
                            score=score,
                            name=data.get("name", node_id),
                            type=data.get("type", "UNKNOWN"),
                            description=data.get("description", ""),
                        )
                    )
            hits.sort(key=lambda hit: (-hit.score, hit.node_id))
            return hits[:limit]

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            if node_id not in self.graph:
                return None
            return dict(self.graph.nodes[node_id])

    def _merged_aliases_of(self, node_id: str) -> list[str]:
        """Node ids that were merged into node_id via
        resolve_alias(distinct=False) - i.e. nodes with a SAME_AS edge
        pointing at node_id. Used by get_incident_edges so a canonical
        node's real extracted edges aren't the only ones surfaced after a
        merge; the alias's own edges (which resolve_alias never moves or
        copies) stay reachable too, through the node that now represents
        both."""
        return [
            source_id
            for source_id, _, data in self.graph.in_edges(node_id, data=True)
            if data.get("type") == EdgeType.SAME_AS.value
        ]

    def get_incident_edges(self, node_id: str) -> list[IncidentEdge]:
        """All edges touching node_id, deduplicated (MultiDiGraph can
        return the same edge from both an in- and out-edge query if it's a
        self-loop), plus - if other nodes were merged into node_id via
        resolve_alias - their edges too. Without this, a merge only adds a
        SAME_AS edge; the alias's real extracted relations stay stored
        under its own node_id and would otherwise become permanently
        unreachable from graph evidence once search_nodes stops returning
        the (now-merged) alias as its own hit."""
        with self._lock:
            if node_id not in self.graph:
                return []
            node_ids = [node_id, *self._merged_aliases_of(node_id)]
            combined = []
            for nid in node_ids:
                combined += list(self.graph.in_edges(nid, keys=True, data=True))
                combined += list(self.graph.out_edges(nid, keys=True, data=True))
            seen: set[str] = set()
            edges: list[IncidentEdge] = []
            for source_id, target_id, edge_id, data in combined:
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                if data.get("type") == EdgeType.SAME_AS.value:
                    # The merge marker itself, not a real extracted
                    # relation - already accounted for via
                    # _merged_aliases_of, not useful as evidence.
                    continue
                edges.append(
                    IncidentEdge(
                        edge_id=edge_id,
                        source_id=source_id,
                        target_id=target_id,
                        source_name=self.graph.nodes.get(source_id, {}).get(
                            "name", source_id
                        ),
                        target_name=self.graph.nodes.get(target_id, {}).get(
                            "name", target_id
                        ),
                        relation=data.get("type", "UNKNOWN"),
                        source_quote=data.get("source_quote") or "",
                        source_paper_id=data.get("source_paper_id"),
                        source_section=data.get("source_section"),
                    )
                )
            return edges

    def export_session_graph(self, session_id: str) -> SessionGraphExport:
        """All nodes/edges tagged with session_id, as a flat list for a
        client-side force layout (Graph Explorer). Strict scoping: an edge
        is only included when BOTH endpoints belong to this session - it
        never reaches into the shared graph, unlike get_incident_edges
        which follows a single node's real edges regardless of who added
        them. Same session_id in _session_ids(data) membership check as
        remove_by_session/find_sparse_pairs."""
        with self._lock:
            nodes = [
                SessionGraphNode(
                    node_id=node_id,
                    name=data.get("name", node_id),
                    type=data.get("type", "UNKNOWN"),
                    description=data.get("description", ""),
                )
                for node_id, data in self.graph.nodes(data=True)
                if session_id in _session_ids(data)
            ]
            node_ids = {n.node_id for n in nodes}
            seen: set[str] = set()
            edges: list[SessionGraphEdge] = []
            for source_id, target_id, edge_id, data in self.graph.edges(
                keys=True, data=True
            ):
                if (
                    edge_id in seen
                    or source_id not in node_ids
                    or target_id not in node_ids
                ):
                    continue
                if data.get("type") == EdgeType.SAME_AS.value:
                    # The merge marker itself, not a real relation - same
                    # exclusion as get_incident_edges.
                    continue
                seen.add(edge_id)
                edges.append(
                    SessionGraphEdge(
                        edge_id=edge_id,
                        source_id=source_id,
                        target_id=target_id,
                        relation=data.get("type", "UNKNOWN"),
                    )
                )
            return SessionGraphExport(nodes=nodes, edges=edges)

    def _stable_node_id(
        self, paper_id: str, name: str, node_type: NodeType
    ) -> str:
        normalized = _normalize_name(name)
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL, f"node:{paper_id}:{node_type.value}:{normalized}"
            )
        )

    def _stable_paper_node_id(self, paper_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"paper:{paper_id}"))

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
        self,
        node_type: NodeType | None = None,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """Candidate pairs with no direct edge but at least one common
        neighbor — topology decides, not LLM guessing.

        session_id, when given, restricts candidates to nodes tagged with
        that session (same tagging remove_by_session already relies on) -
        otherwise gap suggestions draw from every paper ever ingested in
        any session, not just what's actually in this one."""
        with self._lock:
            candidates = [
                n
                for n, data in self.graph.nodes(data=True)
                if (node_type is None or data.get("type") == node_type.value)
                and (session_id is None or session_id in _session_ids(data))
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
        similarity. Three-way routing on the result.

        Both loops skip already-merged alias nodes (resolve_alias
        (distinct=False) target) - without this, a later entity could
        match a node that itself got superseded by an earlier merge, and
        canonicalize/apply_extraction_result would auto-merge or raise a
        clarification question against a dead alias instead of the real
        canonical node, same class of bug search_nodes had before it
        gained the same exclusion.
        """
        with self._lock:
            normalized = _normalize_name(name)
            new_abbreviation = _abbreviation_of(name)
            abbreviation_matches: list[str] = []
            for node_id, data in self.graph.nodes(data=True):
                if node_type is not None and data.get("type") != node_type.value:
                    continue
                if self._is_merged_alias(node_id):
                    continue
                existing_name = data.get("name", "")
                if _normalize_name(existing_name) == normalized:
                    return CanonicalizationResult("auto_merge", node_id, 1.0)
                # Bare-abbreviation match: one side is exactly "Full Form
                # (Abbrev)" and the other is exactly "Abbrev" - as strong a
                # signal as the exact-string match above. Collected rather
                # than returned immediately: if the abbreviation happens to
                # be shared by more than one differently-qualified existing
                # name (e.g. "moral ODD" abbreviates both "moral
                # operational design domain (moral ODD)" and "explicit
                # moral operational design domain (moral ODD)"), which one
                # a bare mention actually means is genuinely ambiguous and
                # must fall through to the normal embedding-similarity /
                # needs_clarification path below, not be silently decided
                # by graph iteration order.
                if new_abbreviation is not None and _normalize_name(existing_name) == new_abbreviation:
                    abbreviation_matches.append(node_id)
                elif normalized == _abbreviation_of(existing_name):
                    abbreviation_matches.append(node_id)

            if len(abbreviation_matches) == 1:
                return CanonicalizationResult("auto_merge", abbreviation_matches[0], 1.0)

            if embedding is None:
                return CanonicalizationResult("new")

            best_id, best_score = None, 0.0
            for node_id, data in self.graph.nodes(data=True):
                if node_type is not None and data.get("type") != node_type.value:
                    continue
                if self._is_merged_alias(node_id):
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
        clarification: "ClarificationOrchestrator | None" = None,
        session_id: str | None = None,
    ) -> GraphIngestionReport:
        """Persist structured extraction output with stable, retry-safe IDs.

        Each individual graph read/write this method calls (add_node,
        canonicalize, add_edge) is independently lock-protected, which is
        enough to prevent the crash this service is actually exposed to -
        a concurrent iteration/mutation on self.graph raising "dictionary
        changed size during iteration". This method does not hold one lock
        across its own entire body, so two concurrent
        apply_extraction_result calls can still interleave between their
        individual locked steps (e.g. both could canonicalize a similar
        entity as "new" before either has written it) - a real but lower-
        severity race than the crash risk, and one Firestore writes
        wouldn't be transactional against either way. Acceptable given
        this service's documented single-instance, --concurrency=4 deploy
        profile (see service/state.py) - revisit if that profile changes.
        """

        # A paper's node id is deterministic from paper_id alone (see
        # _stable_paper_node_id) - re-ingesting the same paper_id (e.g.
        # the same arXiv id added into a second session) reuses the same
        # node either way. add_node itself accumulates session
        # membership on write, so this call is enough to make the paper
        # (and, below, every reused entity) visible from both sessions -
        # no special-casing needed here anymore.
        paper_node_id = self._stable_paper_node_id(extraction.paper_id)
        paper_node = Node(
            id=paper_node_id,
            type=NodeType.PAPER,
            name=paper_name or extraction.paper_id,
            description="Source paper",
            session_id=session_id,
        )
        self.add_node(paper_node)

        entity_to_node_id: dict[tuple[str, NodeType], str] = {}
        node_writes: list[NodeWriteResult] = []
        for entity in extraction.entities:
            if entity.type == NodeType.PAPER:
                entity_to_node_id[
                    (_normalize_name(entity.name), entity.type)
                ] = paper_node.id
                node_writes.append(
                    NodeWriteResult(
                        entity_name=entity.name,
                        node_id=paper_node.id,
                        decision="auto_merge",
                        score=1.0,
                        reused_existing_node=True,
                    )
                )
                continue
            embedding = embedding_fn(entity) if embedding_fn is not None else None
            canonical = self.canonicalize(
                entity.name, embedding=embedding, node_type=entity.type
            )
            if canonical.decision == "auto_merge" and canonical.matched_node_id:
                node_id = canonical.matched_node_id
                reused = True
                # Reusing an existing node id is not itself a write - a
                # real add_node call is needed here too, or this
                # session's membership never gets recorded on it at all.
                # Confirmed live and by a failing test: the single most
                # common case this whole fix targets (a well-known
                # entity reused across sessions via canonicalization)
                # was silently skipping accumulation entirely. Keeps the
                # existing node's own name/description/type - this is
                # the same real-world entity, not new content to merge
                # in, so nothing about it should change except which
                # sessions can now see it.
                existing_data = self.graph.nodes[node_id]
                self.add_node(
                    Node(
                        id=node_id,
                        type=NodeType(existing_data["type"]),
                        name=existing_data.get("name", entity.name),
                        description=existing_data.get("description", ""),
                        entity_embedding=existing_data.get("entity_embedding"),
                        session_id=session_id,
                    )
                )
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
                        session_id=session_id,
                    )
                )
                reused = False
                already_distinct = canonical.matched_node_id is not None and tuple(
                    sorted((canonical.matched_node_id, node_id))
                ) in self._known_distinct
                if (
                    canonical.decision == "needs_clarification"
                    and canonical.matched_node_id
                    and clarification is not None
                    and not already_distinct
                ):
                    # The node is created either way so the batch never
                    # stalls on a person - this just tags the provisional
                    # node so a later answer can merge it via
                    # resolve_alias, same fix-it mechanism used for
                    # manual corrections today. already_distinct guards a
                    # retry of this exact pair (deterministic node_id) from
                    # re-asking a question a person already answered
                    # "no, genuinely different" for.
                    matched_data = self.graph.nodes.get(
                        canonical.matched_node_id, {}
                    )
                    clarification.register_entity_merge_question(
                        provisional_node_id=node_id,
                        entity_name=entity.name,
                        candidate_node_id=canonical.matched_node_id,
                        candidate_name=matched_data.get(
                            "name", canonical.matched_node_id
                        ),
                        candidate_description=matched_data.get("description", ""),
                        score=canonical.score,
                    )
            entity_to_node_id[(_normalize_name(entity.name), entity.type)] = node_id
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
        skipped_relations = 0
        for relation in extraction.relations:
            try:
                source_id = self._resolve_relation_endpoint(
                    extraction.paper_id,
                    relation.source_entity,
                    relation.source_type,
                    entity_to_node_id,
                    session_id=session_id,
                )
                target_id = self._resolve_relation_endpoint(
                    extraction.paper_id,
                    relation.target_entity,
                    relation.target_type,
                    entity_to_node_id,
                    session_id=session_id,
                )
            except ValueError:
                # An untyped relation endpoint whose name matches more than
                # one existing node in the graph is genuinely ambiguous
                # (_resolve_relation_endpoint raises rather than guessing).
                # By this point earlier entities/nodes in this same paper
                # have already been durably written to Firestore via
                # add_node - letting this exception propagate would abort
                # the rest of the paper's relations with no report of what
                # already landed. Skip just this one relation instead, the
                # same per-unit isolation already applied to per-window
                # extraction failures in gemini_extractor.py.
                skipped_relations += 1
                logger.warning(
                    "GraphManager: relation endpoint ambiguous, skipping "
                    "relation",
                    exc_info=True,
                )
                continue
            edge_id = self._stable_edge_id(
                extraction.paper_id,
                source_id,
                target_id,
                relation.relation,
                relation.source_quote,
            )
            # Deterministic edge id, same re-ingest-collision reasoning as
            # the paper node above - add_edge accumulates session
            # membership on write, so re-ingesting this relation from a
            # different session correctly adds it there too.
            edge = Edge(
                id=edge_id,
                source_id=source_id,
                target_id=target_id,
                type=relation.relation,
                provenance=ProvenanceTag.EXTRACTED,
                source_paper_id=extraction.paper_id,
                source_section=relation.source_section,
                source_quote=relation.source_quote,
                session_id=session_id,
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
            skipped_relations=skipped_relations,
        )

    def _resolve_relation_endpoint(
        self,
        paper_id: str,
        name: str,
        node_type: NodeType | None,
        entity_to_node_id: dict[tuple[str, NodeType], str],
        *,
        session_id: str | None = None,
    ) -> str:
        normalized = _normalize_name(name)
        if node_type is not None:
            exact = entity_to_node_id.get((normalized, node_type))
            if exact is not None:
                return exact
        else:
            matches = [
                node_id
                for (entity_name, _), node_id in entity_to_node_id.items()
                if entity_name == normalized
            ]
            if len(set(matches)) == 1:
                return matches[0]
            if len(set(matches)) > 1:
                raise ValueError(
                    f"ambiguous relation endpoint {name!r}; source_type/target_type is required"
                )

        canonical = self.canonicalize(name, node_type=node_type)
        if canonical.decision == "auto_merge" and canonical.matched_node_id:
            # Same accumulation gap as the main entity loop above: reusing
            # an existing node id is not itself a write, so this
            # session's membership must be recorded explicitly here too.
            existing_data = self.graph.nodes[canonical.matched_node_id]
            self.add_node(
                Node(
                    id=canonical.matched_node_id,
                    type=NodeType(existing_data["type"]),
                    name=existing_data.get("name", name),
                    description=existing_data.get("description", ""),
                    entity_embedding=existing_data.get("entity_embedding"),
                    session_id=session_id,
                )
            )
            return canonical.matched_node_id

        resolved_type = node_type or NodeType.CONCEPT
        node_id = self._stable_node_id(paper_id, name, resolved_type)
        self.add_node(
            Node(
                id=node_id,
                type=resolved_type,
                name=name,
                description="Implicit relation endpoint from extraction",
                session_id=session_id,
            )
        )
        entity_to_node_id[(normalized, resolved_type)] = node_id
        return node_id
