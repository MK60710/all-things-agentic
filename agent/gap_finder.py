"""Gap-Finder: topology decides candidates, Gemini only explains why.

Candidate pairs come from Graph Manager's find_sparse_pairs — a deterministic
common-neighbors computation, not LLM guessing. User feedback re-weights
future rankings via a simple per-node interest score: no ML model, just a
demonstrable signal that a "not interesting" rating changes what gets
surfaced next (the Day 16 verification checkpoint in the plan).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone

import networkx as nx
from pydantic import BaseModel

from agent.graph_manager import GraphManager
from agent.schema import NodeType

ExplainFn = Callable[[str, str, list[str]], str]


class GapCandidate(BaseModel):
    node_a_id: str
    node_b_id: str
    node_a_name: str
    node_b_name: str
    common_neighbor_ids: list[str]
    score: float
    explanation: str | None = None


def _default_explain(name_a: str, name_b: str, evidence: list[str]) -> str:
    """Placeholder until wired to a real Gemini call inside the ADK agent."""
    shared = ", ".join(evidence) if evidence else "no shared neighbors"
    return (
        f"{name_a} and {name_b} share context ({shared}) but have no direct "
        f"connection in the graph — worth checking if that's a real gap."
    )


class GapFinder:
    def __init__(
        self,
        graph_manager: GraphManager,
        explain_fn: ExplainFn = _default_explain,
        db_client=None,
    ):
        self._gm = graph_manager
        self._explain_fn = explain_fn
        self._db = db_client
        self._node_boost: dict[str, float] = {}

    def find_candidates(
        self, node_type: NodeType | None = None, limit: int = 5
    ) -> list[GapCandidate]:
        pairs = self._gm.find_sparse_pairs(node_type=node_type, limit=limit * 3)
        undirected = self._gm.graph.to_undirected()

        candidates = []
        for a_id, b_id in pairs:
            a_data = self._gm.graph.nodes[a_id]
            b_data = self._gm.graph.nodes[b_id]
            common_ids = list(nx.common_neighbors(undirected, a_id, b_id))
            base_score = float(len(common_ids))
            boost = self._node_boost.get(a_id, 0.0) + self._node_boost.get(
                b_id, 0.0
            )
            candidates.append(
                GapCandidate(
                    node_a_id=a_id,
                    node_b_id=b_id,
                    node_a_name=a_data.get("name", a_id),
                    node_b_name=b_data.get("name", b_id),
                    common_neighbor_ids=common_ids,
                    score=base_score + boost,
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        top = candidates[:limit]
        for c in top:
            evidence_names = [
                self._gm.graph.nodes[n].get("name", n)
                for n in c.common_neighbor_ids
            ]
            c.explanation = self._explain_fn(
                c.node_a_name, c.node_b_name, evidence_names
            )
        return top

    def record_feedback(
        self, node_a_id: str, node_b_id: str, interesting: bool
    ) -> None:
        """Applied immediately to future rankings — the concrete mechanism
        behind "adapts to the user's thinking", not just a log entry."""
        delta = 1.0 if interesting else -1.0
        self._node_boost[node_a_id] = self._node_boost.get(node_a_id, 0.0) + delta
        self._node_boost[node_b_id] = self._node_boost.get(node_b_id, 0.0) + delta

        if self._db is not None:
            event_id = str(uuid.uuid4())
            self._db.collection("feedback_events").document(event_id).set(
                {
                    "type": "gap_rating",
                    "node_a_id": node_a_id,
                    "node_b_id": node_b_id,
                    "interesting": interesting,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
