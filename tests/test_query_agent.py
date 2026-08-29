from __future__ import annotations

from types import SimpleNamespace

from agent.clarification_orchestrator import ClarificationOrchestrator
from agent.graph_manager import GraphManager, NodeSearchHit
from agent.query_agent import QueryAgent
from agent.retrieval import ChunkIndex
from agent.schema import Edge, EdgeType, ExtractionChunk, Node, NodeType, ProvenanceTag


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


def test_graph_citation_recovers_a_real_page_number_from_the_matching_chunk(fake_db):
    """IncidentEdge never stores a page number (only chunks do) - a graph
    -mode citation used to always show a blank page, inconsistent with a
    chunk-mode citation for the same paper. This is the fix: match the
    edge's own quoted text against the paper's real chunks to recover the
    page it actually came from."""
    graph = _graph(fake_db)
    index = ChunkIndex()
    index.upsert_paper(
        "paper-1",
        [ExtractionChunk(text="The memory method improves recall.", ordinal=0, page_start=4, page_end=4)],
    )
    agent = QueryAgent(index, graph)

    result = agent.answer("How does the memory method use recall?")

    assert result.retrieval_mode == "graph"
    assert result.citations[0].page_start == 4
    assert result.citations[0].page_end == 4


def test_graph_citation_leaves_page_blank_when_quote_matches_no_chunk(fake_db):
    """A quote that was paraphrased during extraction (or whose paper was
    never chunk-indexed at all) has no real page to recover - must stay
    None rather than guessing a wrong page from an unrelated chunk."""
    graph = _graph(fake_db)
    index = ChunkIndex()
    index.upsert_paper("paper-1", ["Completely unrelated chunk text."])
    agent = QueryAgent(index, graph)

    result = agent.answer("How does the memory method use recall?")

    assert result.citations[0].page_start is None
    assert result.citations[0].page_end is None


def test_node_fallback_citation_borrows_provenance_from_an_incident_edge(fake_db):
    """_citation_from_node fires when a matched node's incident edges were
    all already cited via an earlier hit in the same query (or it has
    none) - NodeSearchHit itself has no paper_id/section/page, so before
    this fix the citation rendered with nothing but a name/description
    and no source to check, even for a node with a real incident edge.
    This is the fix: borrow paper/section/page from the first incident
    edge that has one, purely for display."""
    graph = _graph(fake_db)
    agent = QueryAgent(ChunkIndex(), graph)
    hit = NodeSearchHit(node_id="metric", score=1.0, name="Recall Metric", type="METRIC", description="A metric.")

    citation = agent._citation_from_node(hit)

    assert citation.paper_id == "paper-1"
    assert citation.section == "Evaluation"
    assert citation.node_ids == ["metric"]


def test_node_fallback_citation_stays_blank_for_a_node_with_no_incident_edges(fake_db):
    """A genuinely isolated node (no incident edges at all) has nothing to
    borrow provenance from - must stay blank rather than erroring."""
    graph = _graph(fake_db)
    graph.add_node(Node(id="isolated", type=NodeType.CONCEPT, name="Isolated Concept"))
    agent = QueryAgent(ChunkIndex(), graph)
    hit = NodeSearchHit(node_id="isolated", score=1.0, name="Isolated Concept", type="CONCEPT", description="Alone.")

    citation = agent._citation_from_node(hit)

    assert citation.paper_id is None
    assert citation.section is None
    assert citation.page_start is None


def test_low_relevance_graph_match_falls_back_to_chunk(fake_db):
    """A single generic shared token shouldn't lock in a low-relevance
    graph answer over a much more specific chunk match (the failure mode
    behind min_graph_score) - and the fallback must be its own clearly
    separate retrieval_mode, not blended into a graph answer."""
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
    assert result.citations[0].source_kind == "chunk"


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


def test_broad_paper_question_receives_paper_prose_not_graph_entities(fake_db):
    """A paper-scoped summary must not be reduced to the paper's extracted
    entity names. The full question context should include prose from across
    the indexed paper, including later sections when available."""
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="Grounded paper summary")

    index = ChunkIndex()
    index.upsert_paper(
        "paper-1",
        [
            ExtractionChunk(
                text="The introduction studies contextual noncompliance in language models.",
                ordinal=0,
                section="Introduction",
            ),
            ExtractionChunk(
                text="The authors evaluate five categories with a 1000-prompt benchmark.",
                ordinal=1,
                section="Experiments",
            ),
            ExtractionChunk(
                text="LoRA improves noncompliance while preserving general capabilities.",
                ordinal=2,
                section="Conclusion",
            ),
        ],
    )
    agent = QueryAgent(
        index,
        _graph(fake_db),
        client=SimpleNamespace(models=FakeModels()),
    )

    result = agent.answer("Summarize the paper", paper_ids={"paper-1"})

    assert result.retrieval_mode == "vector"
    assert result.answer == "Grounded paper summary"
    assert "contextual noncompliance" in calls[0]["contents"]
    assert "1000-prompt benchmark" in calls[0]["contents"]
    assert "preserving general capabilities" in calls[0]["contents"]


def test_paper_scoped_relationship_question_still_uses_graph(fake_db):
    agent = QueryAgent(ChunkIndex(), _graph(fake_db))

    result = agent.answer(
        "What is the relationship between the memory method and recall?",
        paper_ids={"paper-1"},
    )

    assert result.retrieval_mode == "graph"
    assert result.citations[0].source_kind == "graph"


def test_goal_is_added_to_system_instruction_when_provided():
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="Grounded answer")

    client = SimpleNamespace(models=FakeModels())
    index = ChunkIndex()
    index.upsert_paper("paper-3b", ["The paper evaluates memory retrieval."])
    agent = QueryAgent(index, client=client)

    agent.answer("What does the paper evaluate?", goal="benchmarking retrieval methods")

    instruction = calls[0]["config"].system_instruction
    assert "benchmarking retrieval methods" in instruction


def test_no_goal_leaves_system_instruction_unchanged():
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="Grounded answer")

    client = SimpleNamespace(models=FakeModels())
    index = ChunkIndex()
    index.upsert_paper("paper-3c", ["The paper evaluates memory retrieval."])
    agent = QueryAgent(index, client=client)

    agent.answer("What does the paper evaluate?")

    instruction = calls[0]["config"].system_instruction
    assert "current goal" not in instruction


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


def test_greeting_style_non_question_returns_no_results_not_noise_citations():
    """Confirmed live: "HI" scores exactly 0.0 (zero lexical overlap, zero
    vector signal) against real paper chunks, but assemble_context's own
    min_score used to always default to 0.0 - so a bare "HI" still always
    surfaced whatever its top-N nearest chunks happened to be as if they
    were real evidence, leaving Gemini to notice and disclaim it in the
    answer text while the (irrelevant) citations were already shown."""
    index = ChunkIndex()
    index.upsert_paper("paper-1", ["Something about gradient descent."])
    agent = QueryAgent(index)

    result = agent.answer("HI")

    assert result.retrieval_mode == "no_results"
    assert result.citations == []


def test_confident_vector_match_is_not_flagged():
    """Full lexical overlap guarantees score >= 0.65 regardless of the
    vector-similarity component, so this must always land as confident."""
    index = ChunkIndex()
    index.upsert_paper("paper-1", ["A paper about gradient descent methods."])
    agent = QueryAgent(index)

    result = agent.answer("gradient descent methods")

    assert result.retrieval_mode == "vector"
    assert result.confidence == "confident"


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


def test_forced_node_id_bypasses_ambiguity_check(fake_db):
    """The frontend's 'click to select a candidate' flow - a query that
    would normally be ambiguous must resolve directly to the chosen node
    instead of asking the same clarifying question again."""
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="Grounded answer about the method.")

    agent = QueryAgent(
        ChunkIndex(),
        _ambiguous_graph(fake_db),
        client=SimpleNamespace(models=FakeModels()),
    )

    result = agent.answer("attention", node_id="method")

    assert result.retrieval_mode == "graph"
    assert result.answer == "Grounded answer about the method."
    assert [c.node_ids for c in result.citations] == [["method"]]
    assert len(calls) == 1


def test_forced_node_id_falls_back_to_search_when_id_unknown(fake_db):
    """An unresolvable node_id (stale click, deleted node) must not crash
    or silently return nothing useful - falls through to the normal
    text-search path instead."""
    agent = QueryAgent(ChunkIndex(), _ambiguous_graph(fake_db))

    result = agent.answer("attention", node_id="does-not-exist")

    assert result.retrieval_mode == "ambiguous"


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


def test_low_confidence_graph_match_stays_graph_only(fake_db):
    """A weak-but-real graph match (clears min_graph_score, just softly)
    must stay graph-only, not get silently enriched with raw chunk text -
    a graph answer's citations are always 100% graph evidence, never a mix
    the user can't tell apart. The fact that only lives in the paper's raw
    text (92% accuracy) is real graph-uncovered content; a query with no
    graph hits at all is what should surface it, via the clearly separate
    retrieval_mode="vector" path, not this one."""
    graph = GraphManager(project_id="test", db_client=fake_db)
    graph.add_node(
        Node(id="n1", type=NodeType.METHOD, name="Sparse Retrieval System")
    )
    index = ChunkIndex()
    index.upsert_paper(
        "paper-1",
        ["The sparse retrieval mechanism achieves 92% accuracy on the benchmark."],
    )
    agent = QueryAgent(index, graph)

    result = agent.answer("sparse retrieval mechanism gradient clipping")

    assert result.retrieval_mode == "graph"
    assert result.confidence == "low"
    assert all(citation.source_kind == "graph" for citation in result.citations)
    assert not any("92%" in citation.text for citation in result.citations)


def test_confident_graph_match_is_not_flagged(fake_db):
    graph = GraphManager(project_id="test", db_client=fake_db)
    graph.add_node(
        Node(id="n1", type=NodeType.METHOD, name="Memory Retrieval Method")
    )
    agent = QueryAgent(ChunkIndex(), graph)

    result = agent.answer("memory retrieval method")

    assert result.retrieval_mode == "graph"
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


def test_record_feedback_boosts_a_node_above_a_higher_scoring_rival(fake_db):
    """The concrete "adapts" behavior: negative feedback on the currently
    winning node must be able to flip which node search_nodes' ambiguity
    check treats as the clear top match on a later, identical query -
    a log entry with no effect on ranking wouldn't satisfy this."""
    graph = GraphManager(project_id="test", db_client=fake_db)
    graph.add_node(
        Node(id="strong", type=NodeType.METHOD, name="Retrieval Augmented Method")
    )
    graph.add_node(
        Node(id="weak", type=NodeType.METHOD, name="Retrieval Method")
    )
    agent = QueryAgent(ChunkIndex(), graph)

    before = agent.answer("retrieval augmented method")
    assert before.citations[0].node_ids == ["strong"]

    agent.record_feedback("strong", helpful=False)
    agent.record_feedback("weak", helpful=True)
    agent.record_feedback("weak", helpful=True)

    after = agent.answer("retrieval augmented method")
    assert after.citations[0].node_ids == ["weak"]


def test_record_feedback_writes_a_durable_event_when_db_client_given(fake_db):
    graph = _graph(fake_db)
    agent = QueryAgent(ChunkIndex(), graph, db_client=fake_db)

    agent.record_feedback("method", helpful=True)

    events = list(fake_db.collection("feedback_events").stream())
    assert len(events) == 1
    data = events[0].to_dict()
    assert data["type"] == "query_rating"
    assert data["node_id"] == "method"
    assert data["helpful"] is True


def test_node_boost_rehydrates_from_feedback_events_on_construction(fake_db):
    """Regression: feedback_events was written for durability but never
    read back, so 'capture feedback so it adapts' only held for the
    lifetime of one running process - a restart silently forgot every
    rating a user had given. A fresh QueryAgent pointed at the same
    db_client must already reflect prior feedback, with no record_feedback
    call on the new instance."""
    graph = GraphManager(project_id="test", db_client=fake_db)
    graph.add_node(
        Node(id="strong", type=NodeType.METHOD, name="Retrieval Augmented Method")
    )
    graph.add_node(Node(id="weak", type=NodeType.METHOD, name="Retrieval Method"))
    first_process = QueryAgent(ChunkIndex(), graph, db_client=fake_db)
    first_process.record_feedback("strong", helpful=False)
    first_process.record_feedback("weak", helpful=True)
    first_process.record_feedback("weak", helpful=True)

    restarted = QueryAgent(ChunkIndex(), graph, db_client=fake_db)

    after_restart = restarted.answer("retrieval augmented method")
    assert after_restart.citations[0].node_ids == ["weak"]


def test_record_feedback_without_db_client_does_not_raise():
    agent = QueryAgent(ChunkIndex())
    agent.record_feedback("some-node", helpful=True)  # must not raise


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


def test_paper_scope_prevents_cross_paper_graph_and_chunk_evidence(fake_db):
    index = ChunkIndex()
    index.upsert_paper("paper-2", ["Paper two discusses a separate recall baseline."])
    agent = QueryAgent(index, _graph(fake_db))

    result = agent.answer("What does recall mean?", paper_ids={"paper-2"})

    assert result.retrieval_mode == "vector"
    assert {citation.paper_id for citation in result.citations} == {"paper-2"}


def test_paper_scoped_answer_passes_conversation_history_to_gemini(fake_db):
    """Regression: the /chat route's paper-scoped branch used to call
    QueryAgent.answer() with no history param at all - a follow-up like
    "how does that compare to prior work?" had nothing to resolve "that"
    against, since every call started a fresh conversation. Once history
    is threaded through, _answer_with_gemini must build a real multi-turn
    `contents` list (matching GeneralChatAgent's pattern), not silently
    drop it."""
    from agent.general_chat import ChatTurn

    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="Grounded follow-up answer")

    agent = QueryAgent(
        ChunkIndex(),
        _graph(fake_db),
        client=SimpleNamespace(models=FakeModels()),
    )
    history = [
        ChatTurn(role="user", text="What does the memory method use recall for?"),
        ChatTurn(role="assistant", text="It uses recall to evaluate retrieval quality."),
    ]

    result = agent.answer(
        "How does that compare to the metric?", history=history
    )

    assert result.answer == "Grounded follow-up answer"
    contents = calls[0]["contents"]
    assert isinstance(contents, list)
    assert len(contents) == 3  # 2 history turns + the current question
    assert contents[0].role == "user"
    assert contents[1].role == "model"  # "assistant" maps to "model"
    assert contents[2].role == "user"
    assert "How does that compare to the metric?" in contents[2].parts[0].text


def test_no_history_keeps_the_original_single_string_contents(fake_db):
    """Without history, contents must stay a plain string (the original
    shape) rather than always wrapping in a list - avoids changing
    behavior for every existing non-chat caller of QueryAgent.answer()."""
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="Answer")

    agent = QueryAgent(
        ChunkIndex(),
        _graph(fake_db),
        client=SimpleNamespace(models=FakeModels()),
    )

    agent.answer("How does the memory method use recall?")

    assert isinstance(calls[0]["contents"], str)


def test_paper_scoped_answer_only_walks_incident_edges_once_per_node(fake_db):
    """Regression: paper_ids filtering in answer() and citation-building
    in _graph_evidence each independently called
    GraphManager.get_incident_edges(hit.node_id) for the same node -
    every paper-scoped query walked each candidate node's edges twice."""
    graph = _graph(fake_db)
    calls = []
    original = graph.get_incident_edges

    def counting_get_incident_edges(node_id):
        calls.append(node_id)
        return original(node_id)

    graph.get_incident_edges = counting_get_incident_edges
    agent = QueryAgent(ChunkIndex(), graph)

    agent.answer("How does the memory method use recall?", paper_ids={"paper-1"})

    assert calls.count("method") == 1
