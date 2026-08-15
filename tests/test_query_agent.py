from __future__ import annotations

from types import SimpleNamespace

from agent.graph_manager import GraphManager
from agent.query_agent import QueryAgent
from agent.retrieval import ChunkIndex
from agent.schema import Edge, EdgeType, Node, NodeType, ProvenanceTag


def _graph(fake_db):
    graph = GraphManager(project_id="test", db_client=fake_db)
    graph.add_node(Node(id="method", type=NodeType.METHOD, name="Memory Method"))
    graph.add_node(Node(id="metric", type=NodeType.METRIC, name="Recall Metric"))
    graph.add_edge(
        Edge(
            id="edge-1",
            source_id="method",
            target_id="metric",
            type=EdgeType.USES,
            provenance=ProvenanceTag.EXTRACTED,
            source_paper_id="paper-1",
            source_section="Evaluation",
            source_quote="The memory method improves recall.",
        )
    )
    return graph


def test_graph_evidence_is_preferred_and_cited(fake_db):
    agent = QueryAgent(ChunkIndex(), _graph(fake_db))

    result = agent.answer("How does the memory method use recall?")

    assert result.retrieval_mode == "graph"
    assert result.citations[0].paper_id == "paper-1"
    assert "improves recall" in result.answer
    assert agent.metrics == {"graph_hits": 1, "vector_fallbacks": 0}


def test_chunk_retrieval_is_used_when_graph_has_no_match():
    index = ChunkIndex()
    index.upsert_paper(
        "paper-2",
        ["A retrieval method improves answer quality on the benchmark."],
    )
    agent = QueryAgent(index)

    result = agent.answer("What improves answer quality?")

    assert result.retrieval_mode == "vector"
    assert result.citations[0].paper_id == "paper-2"
    assert agent.metrics == {"graph_hits": 0, "vector_fallbacks": 1}


def test_empty_retrieval_returns_no_results():
    result = QueryAgent(ChunkIndex()).answer("What is unrelated?")

    assert result.retrieval_mode == "no_results"
    assert result.citations == []


def test_gemini_receives_retrieved_context():
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="Grounded answer")

    client = SimpleNamespace(models=FakeModels())
    index = ChunkIndex()
    index.upsert_paper("paper-3", ["The paper evaluates memory retrieval."])
    agent = QueryAgent(index, client=client)

    result = agent.answer("What does the paper evaluate?")

    assert result.answer == "Grounded answer"
    assert "The paper evaluates memory retrieval." in calls[0]["contents"]
    assert calls[0]["config"].temperature == 0


def test_gemini_empty_response_is_safe():
    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="")

    index = ChunkIndex()
    index.upsert_paper("paper-4", ["Stored evidence."])
    result = QueryAgent(
        index,
        client=SimpleNamespace(models=FakeModels()),
    ).answer("What evidence is stored?")

    assert result.answer == "The stored research did not provide a usable answer."
