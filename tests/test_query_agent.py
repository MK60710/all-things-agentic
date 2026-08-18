from __future__ import annotations

from types import SimpleNamespace

from agent.clarification_orchestrator import ClarificationOrchestrator
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


def test_low_relevance_graph_match_falls_back_to_chunk(fake_db):
    """A single generic shared token shouldn't lock in a low-relevance
    graph answer over a much more specific chunk match (the failure mode
    behind min_graph_score)."""
    graph = _graph(fake_db)
    index = ChunkIndex()
    index.upsert_paper(
        "paper-2",
        ["A retrieval method improves answer quality on the benchmark."],
    )
    agent = QueryAgent(index, graph)

    result = agent.answer("What improves answer quality on the benchmark?")

    assert result.retrieval_mode == "vector"
    assert result.citations[0].paper_id == "paper-2"


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


def test_vector_citations_keep_the_top_scoring_hit():
    """assemble_context orders hits by document position, not score - the
    citation list must not silently drop the best match on truncation."""
    index = ChunkIndex()
    index.upsert_paper(
        "paper-3",
        [
            "Unrelated background material about lab equipment.",
            "A graph neural network improves node classification accuracy.",
        ],
    )
    agent = QueryAgent(index, max_citations=1)

    result = agent.answer("What improves node classification accuracy?")

    assert result.retrieval_mode == "vector"
    assert len(result.citations) == 1
    assert "graph neural network" in result.citations[0].text


def test_empty_retrieval_returns_no_results():
    result = QueryAgent(ChunkIndex()).answer("What is unrelated?")

    assert result.retrieval_mode == "no_results"
    assert result.citations == []


def test_no_client_fallback_does_not_leak_prompt_markup():
    """Without a Gemini client, the answer must be the citation text, not
    the raw <source_metadata>-wrapped, angle-bracket-escaped context blob
    built for Gemini's consumption."""
    index = ChunkIndex()
    index.upsert_paper(
        "paper-5", ["The model scores 5 < 10 on the held-out benchmark."]
    )
    result = QueryAgent(index).answer("What does the model score?")

    assert "<source_metadata>" not in result.answer
    assert "&lt;" not in result.answer
    assert "5 < 10" in result.answer


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


def test_gemini_empty_response_falls_back_to_evidence_summary():
    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="")

    index = ChunkIndex()
    index.upsert_paper("paper-4", ["Stored evidence."])
    result = QueryAgent(
        index,
        client=SimpleNamespace(models=FakeModels()),
    ).answer("What evidence is stored?")

    assert result.answer == "Based on the stored research:\n\nStored evidence."


def test_gemini_call_failure_falls_back_instead_of_crashing():
    class FakeModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("quota exceeded")

    index = ChunkIndex()
    index.upsert_paper("paper-6", ["Stored evidence about retries."])
    result = QueryAgent(
        index,
        client=SimpleNamespace(models=FakeModels()),
    ).answer("What is stored?")

    assert result.retrieval_mode == "vector"
    assert "Stored evidence about retries." in result.answer


def _ambiguous_graph(fake_db) -> GraphManager:
    """Two distinct nodes that both contain the query's only real token,
    so search_nodes scores them identically - the concrete trigger for
    QueryAgent's ambiguity check."""
    graph = GraphManager(project_id="test", db_client=fake_db)
    graph.add_node(
        Node(
            id="method",
            type=NodeType.METHOD,
            name="Attention Mechanism",
            description="A method.",
        )
    )
    graph.add_node(
        Node(
            id="concept",
            type=NodeType.CONCEPT,
            name="Attention Economy",
            description="A concept.",
        )
    )
    return graph


def test_ambiguous_graph_matches_return_clarifying_question_not_a_guess(fake_db):
    class FakeModels:
        def generate_content(self, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("Gemini must not be called for an ambiguous query")

    agent = QueryAgent(
        ChunkIndex(),
        _ambiguous_graph(fake_db),
        client=SimpleNamespace(models=FakeModels()),
    )

    result = agent.answer("attention")

    assert result.retrieval_mode == "ambiguous"
    assert {c.node_id for c in result.candidates} == {"method", "concept"}
    assert result.citations == []
    assert result.clarification_question_id is None  # no orchestrator was given


def test_ambiguous_query_registers_a_pending_question_when_orchestrator_given(fake_db):
    orchestrator = ClarificationOrchestrator()
    agent = QueryAgent(
        ChunkIndex(), _ambiguous_graph(fake_db), clarification=orchestrator
    )

    result = agent.answer("attention")

    assert result.clarification_question_id is not None
    pending = orchestrator.pending()
    assert len(pending) == 1
    assert pending[0].kind == "query_disambiguation"
    assert pending[0].id == result.clarification_question_id
    assert {opt.id for opt in pending[0].options} == {"method", "concept"}


def test_low_confidence_graph_match_is_flagged_not_hidden(fake_db):
    """The in-between case from the Part 5 plan: a graph match that clears
    min_graph_score but is still a soft one (2 of 5 query tokens) must not
    look identical to a clean match - confidence should say so."""
    graph = GraphManager(project_id="test", db_client=fake_db)
    graph.add_node(
        Node(id="n1", type=NodeType.METHOD, name="Sparse Retrieval System")
    )
    agent = QueryAgent(ChunkIndex(), graph)

    result = agent.answer("sparse retrieval mechanism gradient clipping")

    assert result.retrieval_mode == "graph"
    assert result.confidence == "low"


def test_confident_graph_match_is_not_flagged(fake_db):
    graph = GraphManager(project_id="test", db_client=fake_db)
    graph.add_node(
        Node(id="n1", type=NodeType.METHOD, name="Memory Retrieval Method")
    )
    agent = QueryAgent(ChunkIndex(), graph)

    result = agent.answer("memory retrieval method")

    assert result.retrieval_mode == "graph"
    assert result.confidence == "confident"


def test_low_confidence_vector_match_is_flagged_not_hidden():
    """Same in-between case on the chunk-retrieval side - only 1 of 3
    query tokens present caps the score below 0.6 regardless of the
    vector-similarity component, so this must always land as low."""
    index = ChunkIndex()
    index.upsert_paper("paper-1", ["Something about gradient descent."])
    agent = QueryAgent(index)

    result = agent.answer("gradient unrelated tangent")

    assert result.retrieval_mode == "vector"
    assert result.confidence == "low"


def test_confident_vector_match_is_not_flagged():
    """Full lexical overlap guarantees score >= 0.65 regardless of the
    vector-similarity component, so this must always land as confident."""
    index = ChunkIndex()
    index.upsert_paper("paper-1", ["A paper about gradient descent methods."])
    agent = QueryAgent(index)

    result = agent.answer("gradient descent methods")

    assert result.retrieval_mode == "vector"
    assert result.confidence == "confident"


def test_answer_scans_search_nodes_only_once_per_query(fake_db):
    """_check_query_ambiguity and _graph_evidence used to each call
    GraphManager.search_nodes independently with identical arguments -
    search_nodes has no request-level memoization (only per-node
    tokenization is cached), so every graph-backed query paid its full
    O(number of nodes) scan cost twice. They must now share one scan."""
    graph = _graph(fake_db)
    calls = []
    original = graph.search_nodes

    def counting_search_nodes(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    graph.search_nodes = counting_search_nodes
    agent = QueryAgent(ChunkIndex(), graph)

    agent.answer("How does the memory method use recall?")

    assert len(calls) == 1


def test_gemini_response_text_property_raising_falls_back():
    """The google-genai SDK can raise from the response.text property
    itself (e.g. a safety-filtered response with no candidates), not just
    return an empty string - the try/except must cover the property
    access, not only the generate_content call."""

    class RaisingResponse:
        @property
        def text(self):
            raise ValueError("no candidates")

    class FakeModels:
        def generate_content(self, **kwargs):
            return RaisingResponse()

    index = ChunkIndex()
    index.upsert_paper("paper-7", ["Stored evidence about safety blocks."])
    result = QueryAgent(
        index,
        client=SimpleNamespace(models=FakeModels()),
    ).answer("What is stored?")

    assert "Stored evidence about safety blocks." in result.answer
