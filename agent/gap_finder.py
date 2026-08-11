"""Gap-Finder: topology decides candidates, Gemini only explains why.

Candidate pairs come from Graph Manager's find_sparse_pairs — a deterministic
common-neighbors computation, not LLM guessing. User feedback re-weights
future rankings via a simple per-node interest score: no ML model, just a
demonstrable signal that a "not interesting" rating changes what gets
surfaced next (the Day 16 verification checkpoint in the plan).
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

import networkx as nx
from google import genai
from pydantic import BaseModel

from agent.graph_manager import GraphManager
from agent.schema import NodeType

logger = logging.getLogger(__name__)

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


class GeminiExplainer:
    """Calls Gemini via Vertex AI to explain a candidate research gap.

    Authenticates via Application Default Credentials (project IAM), not an
    API key, matching the rest of this project's auth convention. Falls
    back to the deterministic template on any failure - a Gemini outage,
    quota limit, or auth issue must never break gap-finding itself, since
    the candidates themselves already come from graph topology, not the
    model.
    """

    def __init__(
        self,
        client: genai.Client | None = None,
        model: str = "gemini-2.5-flash",
        project: str | None = None,
        location: str | None = None,
    ):
        self._model = model
        self._client = client or genai.Client(
            vertexai=True,
            project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    def __call__(self, name_a: str, name_b: str, evidence: list[str]) -> str:
        shared = ", ".join(evidence) if evidence else "no shared neighbors"
        prompt = (
            "You are helping a researcher spot gaps in a knowledge graph "
            "built from academic papers. Two entities share context but "
            "have no direct connection recorded between them.\n\n"
            f"Entity A: {name_a}\n"
            f"Entity B: {name_b}\n"
            f"Shared context (common neighbors): {shared}\n\n"
            "In 1-2 sentences, explain why this might be a real, "
            "worth-checking research gap, or say if it looks more like a "
            "coincidence."
        )
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=prompt
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception:
            # Never let a Gemini outage/quota/config issue break gap-finding
            # itself (candidates already come from graph topology), but log
            # it - a silently-swallowed wrong model name is exactly what
            # slipped through here during development.
            logger.warning(
                "GeminiExplainer call failed, falling back to template",
                exc_info=True,
            )
        return _default_explain(name_a, name_b, evidence)


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
