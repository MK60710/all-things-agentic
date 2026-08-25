from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent.clarification_orchestrator import ClarificationOrchestrator
from agent.contradiction_finder import ContradictionFinder
from agent.document_ingestion import PdfTextExtractor
from agent.extraction_agent import ChunkOnlyStructuredExtractor, ExtractionAgent
from agent.feynman_checker import FeynmanChecker
from agent.gap_finder import GapFinder
from agent.general_chat import GeneralChatAgent
from agent.graph_manager import GraphManager
from agent.schema import Edge, EdgeType, Node, NodeType, ProvenanceTag
from agent.paper_guide import GuideSection, PaperGuide, PaperGuideAgent
from agent.query_agent import QueryAgent
from agent.research_store import ResearchStore
from agent.retrieval import ChunkIndex
from agent.schema import Node, NodeType
from service.app import app
from service.deps import get_state
from service.state import AppState
from service.storage import PaperStore, SessionStore, UploadTokenStore


def _no_op_judge(claim_a: str, claim_b: str):
    """Stands in for GeminiContradictionJudge in fixtures that don't test
    contradiction-checking behavior itself - never makes a live call,
    unlike constructing a real GeminiContradictionJudge with no client,
    which would lazily build a real genai.Client from whatever
    GOOGLE_CLOUD_PROJECT happens to be set in the ambient environment."""
    return None


def _no_op_feynman_judge(node_name: str, node_description: str, explanation: str):
    """Same reasoning as _no_op_judge, for GeminiFeynmanJudge."""
    return None


@pytest.fixture
def app_state(fake_db, tmp_path) -> AppState:
    graph = GraphManager(project_id="test", db_client=fake_db)
    chunks = ChunkIndex(db_client=fake_db)
    clarification = ClarificationOrchestrator(graph_manager=graph)
    return AppState(
        graph=graph,
        chunks=chunks,
        clarification=clarification,
        # No project=/client= -> stays client-less, never attempts a live
        # Vertex AI call, same pattern test_query_agent.py/test_gap_finder.py
        # already use.
        query_agent=QueryAgent(chunks, graph, clarification=clarification, db_client=fake_db),
        general_chat=GeneralChatAgent(),
        paper_guide=PaperGuideAgent(),
        gap_finder=GapFinder(graph, db_client=fake_db),
        contradiction_finder=ContradictionFinder(graph, judge=_no_op_judge, db_client=fake_db),
        feynman_checker=FeynmanChecker(graph, judge=_no_op_feynman_judge),
        extraction_agent=ExtractionAgent(
            document_extractor=PdfTextExtractor(allowed_root=str(tmp_path)),
            structured_extractor=ChunkOnlyStructuredExtractor(),
        ),
        research_store=ResearchStore(chunks, graph),
        upload_root=str(tmp_path),
        paper_store=PaperStore(fake_db),
        upload_tokens=UploadTokenStore(fake_db),
        session_store=SessionStore(fake_db),
    )


@pytest.fixture
def client(app_state, monkeypatch):
    # The real lifespan still runs on TestClient context entry regardless
    # of dependency_overrides, and build_state() would otherwise construct
    # a real firestore.Client and call GraphManager._rehydrate() - a real
    # network call - during test setup. Patch build_state itself so the
    # lifespan's real work never happens; get_state is also overridden so
    # every route handler gets app_state regardless.
    monkeypatch.setattr("service.app.build_state", lambda: app_state)
    app.dependency_overrides[get_state] = lambda: app_state
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_state, None)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_no_results(client):
    response = client.post("/query", json={"query": "nonexistent topic xyz"})
    assert response.status_code == 200
    assert response.json()["retrieval_mode"] == "no_results"


def test_clarifications_empty(client):
    assert client.get("/clarifications").json() == []


def test_clarifications_404_for_unknown_id(client):
    assert client.get("/clarifications/does-not-exist").status_code == 404


def test_clarifications_session_filter_excludes_a_different_sessions_question(
    client, app_state
):
    """With named sessions coexisting, an entity-merge question created by
    one session's ingest must not surface to a different session - it's
    ambiguity about that session's own data, not a shared prompt."""
    candidate = Node(id="candidate-node", type=NodeType.CONCEPT, name="Existing Concept")
    app_state.graph.add_node(candidate)
    provisional = Node(
        id="provisional-node",
        type=NodeType.CONCEPT,
        name="New Concept",
        session_id="session-a",
    )
    app_state.graph.add_node(provisional)
    app_state.clarification.register_entity_merge_question(
        provisional_node_id=provisional.id,
        entity_name="New Concept",
        candidate_node_id=candidate.id,
        candidate_name="Existing Concept",
    )

    assert client.get("/clarifications", params={"session_id": "session-b"}).json() == []
    same_session = client.get("/clarifications", params={"session_id": "session-a"}).json()
    assert len(same_session) == 1
    assert len(client.get("/clarifications").json()) == 1  # unscoped call still returns it


def test_gaps_empty_graph(client):
    assert client.get("/gaps").json() == []


def test_sessions_create_and_list_round_trip(client):
    response = client.post("/sessions", json={"name": "AI session"})
    assert response.status_code == 200
    created = response.json()
    assert created["name"] == "AI session"
    assert created["id"]

    listed = client.get("/sessions").json()
    assert any(s["id"] == created["id"] and s["name"] == "AI session" for s in listed)


def test_rename_session_updates_name(client):
    created = client.post("/sessions", json={"name": "Untitled session"}).json()

    response = client.patch(f"/sessions/{created['id']}", json={"name": "Biology"})

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == created["id"]
    assert body["name"] == "Biology"
    assert body["created_at"] == created["created_at"]

    listed = client.get("/sessions").json()
    assert any(s["id"] == created["id"] and s["name"] == "Biology" for s in listed)


def test_session_goal_round_trips_and_survives_rename(client):
    created = client.post(
        "/sessions", json={"name": "AI session", "goal": "benchmark retrieval methods"}
    ).json()
    assert created["goal"] == "benchmark retrieval methods"

    listed = client.get("/sessions").json()
    match = next(s for s in listed if s["id"] == created["id"])
    assert match["goal"] == "benchmark retrieval methods"

    renamed = client.patch(f"/sessions/{created['id']}", json={"name": "Biology"}).json()
    assert renamed["goal"] == "benchmark retrieval methods"


def test_session_without_goal_defaults_to_none(client):
    created = client.post("/sessions", json={"name": "No goal session"}).json()
    assert created["goal"] is None


def test_session_graph_endpoint_returns_only_that_sessions_nodes(client, app_state):
    created = client.post("/sessions", json={"name": "Graph session"}).json()
    session_id = created["id"]
    a = Node(id="node-a", type=NodeType.CONCEPT, name="In Session", session_id=session_id)
    b = Node(id="node-b", type=NodeType.CONCEPT, name="Other Session", session_id="other-session")
    app_state.graph.add_node(a)
    app_state.graph.add_node(b)
    app_state.graph.add_edge(
        Edge(
            id="edge-1",
            source_id=a.id,
            target_id=a.id,
            type=EdgeType.SUPPORTS,
            provenance=ProvenanceTag.EXTRACTED,
            source_quote="quote",
            session_id=session_id,
        )
    )

    response = client.get(f"/sessions/{session_id}/graph")

    assert response.status_code == 200
    body = response.json()
    assert [n["node_id"] for n in body["nodes"]] == ["node-a"]
    assert len(body["edges"]) == 1


def test_session_graph_endpoint_404s_for_unknown_session(client):
    response = client.get("/sessions/does-not-exist/graph")
    assert response.status_code == 404


def test_bibliography_endpoint_returns_real_bibtex_for_the_sessions_papers(client, app_state):
    created = client.post("/sessions", json={"name": "Bib session"}).json()
    session_id = created["id"]
    app_state.paper_store.save(
        "arxiv-2106.09685", title="LoRA", authors="Edward J. Hu", status="ready", session_id=session_id
    )

    response = client.get(f"/sessions/{session_id}/bibliography")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-bibtex")
    assert "@misc{arxiv_2106_09685," in response.text
    assert "title = {LoRA}" in response.text


def test_bibliography_endpoint_only_includes_this_sessions_papers(client, app_state):
    created = client.post("/sessions", json={"name": "Bib session"}).json()
    session_id = created["id"]
    app_state.paper_store.save("paper-mine", title="Mine", status="ready", session_id=session_id)
    app_state.paper_store.save("paper-other", title="Other", status="ready", session_id="a-different-session")

    response = client.get(f"/sessions/{session_id}/bibliography")

    assert "paper_mine" in response.text.replace("-", "_")
    assert "Other" not in response.text


def test_bibliography_endpoint_degrades_gracefully_for_a_session_with_no_papers(client):
    created = client.post("/sessions", json={"name": "Empty bib session"}).json()
    session_id = created["id"]

    response = client.get(f"/sessions/{session_id}/bibliography")

    assert response.status_code == 200
    assert "No papers" in response.text


def test_bibliography_endpoint_404s_for_unknown_session(client):
    response = client.get("/sessions/does-not-exist/bibliography")
    assert response.status_code == 404


def test_contradictions_check_endpoint_returns_empty_when_nothing_to_compare(client):
    created = client.post("/sessions", json={"name": "Contradictions session"}).json()

    response = client.post(f"/sessions/{created['id']}/contradictions/check")

    assert response.status_code == 200
    assert response.json() == []


def test_contradictions_check_endpoint_404s_for_unknown_session(client):
    response = client.post("/sessions/does-not-exist/contradictions/check")
    assert response.status_code == 404


def test_feynman_prompts_endpoint_404s_for_unknown_paper(client):
    response = client.get("/papers/does-not-exist/feynman/prompts?session_id=session-a")
    assert response.status_code == 404


def test_feynman_prompts_endpoint_returns_testable_nodes_for_the_paper(client, app_state):
    app_state.paper_store.save("paper-a", title="Paper A", status="ready", session_id="session-a")
    paper = Node(id="paper-a", type=NodeType.PAPER, name="Paper A", description="Source paper", session_id="session-a")
    concept = Node(id="concept-a", type=NodeType.CONCEPT, name="Concept A", description="A concept with a real, substantive description.", session_id="session-a")
    app_state.graph.add_node(paper)
    app_state.graph.add_node(concept)
    app_state.graph.add_edge(
        Edge(
            id="edge-a",
            source_id="paper-a",
            target_id="concept-a",
            type=EdgeType.USES,
            provenance=ProvenanceTag.EXTRACTED,
            source_paper_id="paper-a",
            session_id="session-a",
        )
    )

    response = client.get("/papers/paper-a/feynman/prompts?session_id=session-a")

    assert response.status_code == 200
    prompts = response.json()
    assert len(prompts) == 1
    assert prompts[0]["node_id"] == "concept-a"


def test_feynman_check_endpoint_404s_for_unknown_paper(client):
    response = client.post(
        "/papers/does-not-exist/feynman/check",
        json={"node_id": "concept-a", "session_id": "session-a", "explanation": "It helps."},
    )
    assert response.status_code == 404


def _link_node_to_paper(app_state, node_id: str, paper_id: str, session_id: str) -> None:
    app_state.graph.add_edge(
        Edge(
            id=f"edge-{node_id}-{paper_id}",
            source_id=paper_id,
            target_id=node_id,
            type=EdgeType.USES,
            provenance=ProvenanceTag.EXTRACTED,
            source_paper_id=paper_id,
            session_id=session_id,
        )
    )


def test_feynman_check_endpoint_502s_when_the_judge_fails(client, app_state):
    """app_state's feynman_checker uses _no_op_feynman_judge, which always
    returns None - mirrors how a real Gemini outage should surface as a
    clear error, not a fabricated verdict."""
    app_state.paper_store.save("paper-a", title="Paper A", status="ready", session_id="session-a")
    paper = Node(id="paper-a", type=NodeType.PAPER, name="Paper A", description="Source paper", session_id="session-a")
    concept = Node(id="concept-a", type=NodeType.CONCEPT, name="Concept A", description="A concept", session_id="session-a")
    app_state.graph.add_node(paper)
    app_state.graph.add_node(concept)
    _link_node_to_paper(app_state, "concept-a", "paper-a", "session-a")

    response = client.post(
        "/papers/paper-a/feynman/check",
        json={"node_id": "concept-a", "session_id": "session-a", "explanation": "It helps."},
    )

    assert response.status_code == 502


def test_feynman_check_endpoint_rejects_a_node_from_a_different_session(client, app_state):
    """Router-level regression test for the live-confirmed cross-session
    leak: a node_id that belongs to a different session must be rejected
    before it ever reaches the judge, not just filtered client-side."""
    app_state.paper_store.save("paper-a", title="Paper A", status="ready", session_id="session-a")
    paper = Node(id="paper-a", type=NodeType.PAPER, name="Paper A", description="Source paper", session_id="session-a")
    foreign_concept = Node(
        id="foreign-concept",
        type=NodeType.CONCEPT,
        name="Foreign Concept",
        description="Belongs to a different session entirely.",
        session_id="session-b",
    )
    app_state.graph.add_node(paper)
    app_state.graph.add_node(foreign_concept)
    _link_node_to_paper(app_state, "foreign-concept", "paper-a", "session-b")

    response = client.post(
        "/papers/paper-a/feynman/check",
        json={"node_id": "foreign-concept", "session_id": "session-a", "explanation": "attacker probe"},
    )

    assert response.status_code == 502


def test_feynman_check_endpoint_rejects_an_oversized_explanation(client, app_state):
    """Confirmed live: with no cap, a 5MB explanation was accepted and took
    17+ seconds before failing - a free cost/DoS surface on a paid,
    per-call Gemini endpoint. Must 422 before it ever reaches the graph
    lookup or the judge, not fail slowly downstream."""
    app_state.paper_store.save("paper-a", title="Paper A", status="ready", session_id="session-a")

    response = client.post(
        "/papers/paper-a/feynman/check",
        json={"node_id": "concept-a", "session_id": "session-a", "explanation": "x" * 4001},
    )

    assert response.status_code == 422


def test_rename_session_404_for_unknown_session(client):
    response = client.patch("/sessions/does-not-exist", json={"name": "Anything"})

    assert response.status_code == 404


def test_delete_session_cascades_papers_graph_and_clarifications(client, app_state):
    """Full cascade: deleting a session removes its papers, the graph
    nodes/edges it added, its chunks, and any clarification question
    referencing one of those nodes - not just the session record."""
    created = client.post("/sessions", json={"name": "To delete"}).json()
    session_id = created["id"]

    app_state.paper_store.save(
        "paper-a", title="Paper A", status="ready", session_id=session_id
    )
    app_state.chunks.upsert_paper("paper-a", ["Some paper text about apples."])
    provisional = Node(
        id="provisional-node",
        type=NodeType.CONCEPT,
        name="New Concept",
        session_id=session_id,
    )
    candidate = Node(id="candidate-node", type=NodeType.CONCEPT, name="Existing Concept")
    app_state.graph.add_node(provisional)
    app_state.graph.add_node(candidate)
    app_state.clarification.register_entity_merge_question(
        provisional_node_id=provisional.id,
        entity_name="New Concept",
        candidate_node_id=candidate.id,
        candidate_name="Existing Concept",
    )

    response = client.delete(f"/sessions/{session_id}")

    assert response.status_code == 204
    assert not any(s["id"] == session_id for s in client.get("/sessions").json())
    assert not any(p["id"] == "paper-a" for p in app_state.paper_store.list())
    assert app_state.chunks.paper_chunks("paper-a") == []
    assert "provisional-node" not in app_state.graph.graph
    assert app_state.clarification.pending() == []
    # A node from a different session that never touched this one is untouched.
    assert "candidate-node" in app_state.graph.graph


def test_delete_session_404_for_unknown_session(client):
    response = client.delete("/sessions/does-not-exist")

    assert response.status_code == 404


def test_delete_session_leaves_a_paper_shared_with_another_session_intact(client, app_state):
    """A paper (and its graph node) added to two sessions must survive
    deleting one of them - only removed once every session sharing it is
    gone. Regression test for the exact live-confirmed bug: re-ingesting
    a paper into a second session used to silently steal it from the
    first, and this is the other half - deleting either session must not
    destroy data the other one still legitimately owns."""
    session_a = client.post("/sessions", json={"name": "Session A"}).json()["id"]
    session_b = client.post("/sessions", json={"name": "Session B"}).json()["id"]

    app_state.paper_store.save("shared-paper", title="Shared Paper", status="ready", session_id=session_a)
    app_state.paper_store.save("shared-paper", title="Shared Paper", status="ready", session_id=session_b)
    app_state.chunks.upsert_paper("shared-paper", ["Some shared paper text."])
    shared_node = Node(id="shared-node", type=NodeType.CONCEPT, name="Shared Concept")
    app_state.graph.add_node(shared_node)
    app_state.graph.add_node(Node(**{**shared_node.model_dump(), "session_id": session_a}))
    app_state.graph.add_node(Node(**{**shared_node.model_dump(), "session_id": session_b}))

    response = client.delete(f"/sessions/{session_a}")

    assert response.status_code == 204
    # Session A is gone, but the shared paper/node survive via session B.
    assert any(p["id"] == "shared-paper" for p in app_state.paper_store.list())
    assert app_state.chunks.paper_chunks("shared-paper") != []
    assert "shared-node" in app_state.graph.graph
    listing = client.get("/papers", params={"session_id": session_b}).json()
    assert {p["id"] for p in listing} == {"shared-paper"}

    response = client.delete(f"/sessions/{session_b}")

    assert response.status_code == 204
    # Now the last session sharing it is gone too - it's actually removed.
    assert not any(p["id"] == "shared-paper" for p in app_state.paper_store.list())
    assert app_state.chunks.paper_chunks("shared-paper") == []
    assert "shared-node" not in app_state.graph.graph


def test_query_feedback_accepts_and_returns_no_content(client):
    response = client.post(
        "/query/feedback", json={"node_id": "some-node", "helpful": True}
    )
    assert response.status_code == 204


def test_gap_feedback_accepts_and_returns_no_content(client):
    response = client.post(
        "/gaps/feedback",
        json={"node_a_id": "a", "node_b_id": "b", "interesting": False},
    )
    assert response.status_code == 204


def test_answer_clarification_end_to_end(client, app_state):
    """Real path: register a question directly on the shared orchestrator
    (as apply_extraction_result would), list it via GET, answer it via
    POST, and confirm the graph mutation actually happened - the same
    round trip a frontend would do."""
    graph = app_state.graph
    existing = Node(id="existing", type=NodeType.CONCEPT, name="Existing")
    provisional = Node(id="provisional", type=NodeType.CONCEPT, name="Provisional")
    graph.add_node(existing)
    graph.add_node(provisional)

    question = app_state.clarification.register_entity_merge_question(
        provisional_node_id="provisional",
        entity_name="Provisional",
        candidate_node_id="existing",
        candidate_name="Existing",
    )

    listed = client.get("/clarifications").json()
    assert len(listed) == 1
    assert listed[0]["id"] == question.id
    assert listed[0]["kind"] == "entity_merge"

    response = client.post(
        f"/clarifications/{question.id}/answer", json={"option_id": "existing"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    assert client.get("/clarifications").json() == []

    edges = list(graph.graph.get_edge_data("provisional", "existing").values())
    assert len(edges) == 1
    assert edges[0]["type"] == "SAME_AS"


def test_answer_clarification_invalid_option_returns_400(client, app_state):
    app_state.clarification.register_entity_merge_question(
        provisional_node_id="p", entity_name="X", candidate_node_id="c", candidate_name="Y"
    )
    question_id = client.get("/clarifications").json()[0]["id"]

    response = client.post(
        f"/clarifications/{question_id}/answer", json={"option_id": "not-real"}
    )
    assert response.status_code == 400


def test_papers_upload_returns_the_real_graph_writes_for_the_build_animation(
    client, app_state, monkeypatch
):
    """new_nodes/new_edges must reflect GraphManager's actual post-
    canonicalization writes (GraphIngestionReport.node_writes/edge_writes),
    not a synthesized count - the frontend's live graph-building animation
    renders exactly this data."""
    from agent.document_ingestion import DocumentIngestionResult
    from agent.extraction_agent import ExtractionOutcome
    from agent.schema import ExtractedEntity, ExtractionResult

    def fake_extract_one(paper_id, path, fail_closed=False):
        return ExtractionOutcome(
            paper_id=paper_id,
            document=DocumentIngestionResult(
                paper_id=paper_id, pdf_path=path, pages=[], raw_text="text", chunks=["text"]
            ),
            result=ExtractionResult(
                paper_id=paper_id,
                entities=[
                    ExtractedEntity(name="GraphRAG", type=NodeType.METHOD, description="A method")
                ],
                relations=[],
            ),
        )

    monkeypatch.setattr(app_state.extraction_agent, "extract_one", fake_extract_one)

    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"paper_id": "viz-test-paper"},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["new_nodes"]) == 1
    assert data["new_nodes"][0]["name"] == "GraphRAG"
    assert data["new_nodes"][0]["type"] == "METHOD"
    assert data["new_edges"] == []


def test_papers_upload_backfills_implicit_relation_endpoint_nodes(
    client, app_state, monkeypatch
):
    """Regression: GraphManager._resolve_relation_endpoint can create an
    'implicit relation endpoint' node directly via add_node() when a
    relation names an entity outside the extracted entities list - that
    node is real and used as a real edge endpoint, but never gets a
    NodeWriteResult, so node_writes alone misses it. Reproduced live:
    the frontend's force-graph threw 'node not found' repeatedly because
    an edge referenced a node id that was never in new_nodes. Every
    edge's source/target must have a corresponding entry in new_nodes."""
    from agent.document_ingestion import DocumentIngestionResult
    from agent.extraction_agent import ExtractionOutcome
    from agent.schema import EdgeType, ExtractedEntity, ExtractedRelation, ExtractionResult

    def fake_extract_one(paper_id, path, fail_closed=False):
        return ExtractionOutcome(
            paper_id=paper_id,
            document=DocumentIngestionResult(
                paper_id=paper_id, pdf_path=path, pages=[], raw_text="text", chunks=["text"]
            ),
            result=ExtractionResult(
                paper_id=paper_id,
                entities=[
                    ExtractedEntity(name="MyMethod", type=NodeType.METHOD, description="A method")
                ],
                # "SomeImplicitModel" is deliberately absent from entities
                # above - this is exactly what triggers the implicit-node
                # creation path in _resolve_relation_endpoint.
                relations=[
                    ExtractedRelation(
                        source_entity="MyMethod",
                        source_type=NodeType.METHOD,
                        relation=EdgeType.USES,
                        target_entity="SomeImplicitModel",
                        target_type=NodeType.MODEL,
                        source_quote="MyMethod uses SomeImplicitModel",
                    )
                ],
            ),
        )

    monkeypatch.setattr(app_state.extraction_agent, "extract_one", fake_extract_one)

    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"paper_id": "implicit-node-test-paper"},
    )

    assert response.status_code == 200
    data = response.json()
    node_ids = {node["node_id"] for node in data["new_nodes"]}
    for edge in data["new_edges"]:
        assert edge["source_id"] in node_ids
        assert edge["target_id"] in node_ids
    names = {node["name"] for node in data["new_nodes"]}
    assert "SomeImplicitModel" in names


def test_papers_upload_auto_merges_a_bare_abbreviation_without_asking(
    client, app_state, monkeypatch
):
    """Regression guard for GraphManager.canonicalize's abbreviation-in-
    parens auto-merge (Part A2): a bare "moral ODD" extracted against an
    already-ingested "moral operational design domain (moral ODD)" node
    must land as a silent auto_merge through the full HTTP ingest path,
    not raise a clarification question - this is exactly the class of
    false-positive question that made a real ingest dump 9 of them at
    once."""
    from agent.document_ingestion import DocumentIngestionResult
    from agent.extraction_agent import ExtractionOutcome
    from agent.schema import ExtractedEntity, ExtractionResult

    existing = Node(
        id="existing-moral-odd",
        type=NodeType.CONCEPT,
        name="moral operational design domain (moral ODD)",
    )
    app_state.graph.add_node(existing)

    def fake_extract_one(paper_id, path, fail_closed=False):
        return ExtractionOutcome(
            paper_id=paper_id,
            document=DocumentIngestionResult(
                paper_id=paper_id, pdf_path=path, pages=[], raw_text="text", chunks=["text"]
            ),
            result=ExtractionResult(
                paper_id=paper_id,
                entities=[
                    ExtractedEntity(
                        name="moral ODD", type=NodeType.CONCEPT, description="An abbreviation"
                    )
                ],
                relations=[],
            ),
        )

    monkeypatch.setattr(app_state.extraction_agent, "extract_one", fake_extract_one)

    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"paper_id": "abbreviation-test-paper"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["pending_clarification_count"] == 0
    assert len(data["new_nodes"]) == 1
    assert data["new_nodes"][0]["node_id"] == existing.id
    assert data["new_nodes"][0]["reused_existing_node"] is True


def test_papers_upload_and_ingest(client):
    pdf_bytes = b"%PDF-1.4 fake"
    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
        data={"paper_id": "test-paper-1"},
    )
    # ChunkOnlyStructuredExtractor makes no real extraction calls, and
    # PdfTextExtractor's pdftotext output on a fake PDF is whatever the
    # real pdftotext binary produces (or a DocumentExtractionError) - only
    # assert the endpoint round-trips without crashing and returns one of
    # the two documented outcomes, not a specific text result.
    assert response.status_code in (200, 422)


def test_papers_upload_persists_the_given_session_id(client, app_state, monkeypatch):
    from agent.document_ingestion import DocumentIngestionResult
    from agent.extraction_agent import ExtractionOutcome
    from agent.schema import ExtractionResult

    def fake_extract_one(paper_id, path, fail_closed=False):
        return ExtractionOutcome(
            paper_id=paper_id,
            document=DocumentIngestionResult(
                paper_id=paper_id, pdf_path=path, pages=[], raw_text="text", chunks=["text"]
            ),
            result=ExtractionResult(paper_id=paper_id, entities=[], relations=[]),
        )

    monkeypatch.setattr(app_state.extraction_agent, "extract_one", fake_extract_one)

    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"paper_id": "session-tagged-paper", "session_id": "session-xyz"},
    )

    assert response.status_code == 200
    saved = next(p for p in app_state.paper_store.list() if p["id"] == "session-tagged-paper")
    assert saved["session_ids"] == ["session-xyz"]


def test_arxiv_ingest_persists_the_given_session_id(client, app_state, monkeypatch):
    from agent.document_ingestion import DocumentIngestionResult
    from agent.extraction_agent import ExtractionOutcome
    from agent.schema import ExtractionResult

    def fake_extract_one(paper_id, path, fail_closed=False):
        return ExtractionOutcome(
            paper_id=paper_id,
            document=DocumentIngestionResult(
                paper_id=paper_id, pdf_path=path, pages=[], raw_text="text", chunks=["text"]
            ),
            result=ExtractionResult(paper_id=paper_id, entities=[], relations=[]),
        )

    monkeypatch.setattr(app_state.extraction_agent, "extract_one", fake_extract_one)
    monkeypatch.setattr(
        "service.routers.papers.requests.get",
        lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            iter_content=lambda chunk_size=None: [b"%PDF-1.4 fake"],
        ),
    )

    response = client.post(
        "/papers/arxiv",
        json={
            "arxiv_id": "2101.00001",
            "title": "A Session-Tagged Paper",
            "session_id": "session-xyz",
        },
    )

    assert response.status_code == 200
    saved = next(
        p for p in app_state.paper_store.list() if p["id"] == "arxiv-2101.00001"
    )
    assert saved["session_ids"] == ["session-xyz"]


def _fake_extract_one(paper_id, path, fail_closed=False):
    from agent.document_ingestion import DocumentIngestionResult
    from agent.extraction_agent import ExtractionOutcome
    from agent.schema import ExtractionResult

    return ExtractionOutcome(
        paper_id=paper_id,
        document=DocumentIngestionResult(
            paper_id=paper_id, pdf_path=path, pages=[], raw_text="text", chunks=["text"]
        ),
        result=ExtractionResult(
            paper_id=paper_id, entities=[], relations=[], chunks=["text"]
        ),
    )


def _fake_parse_document(paper_id, path):
    from agent.document_ingestion import DocumentIngestionResult

    return DocumentIngestionResult(
        paper_id=paper_id, pdf_path=path, pages=[], raw_text="text", chunks=["text"]
    )


def test_papers_upload_pregenerates_and_caches_the_guide(client, app_state, monkeypatch):
    """The guide worker in _ingest runs concurrently with extraction - by
    the time /papers returns, a successful guide should already be
    cached on the paper record, with no separate /guide call needed."""
    monkeypatch.setattr(app_state.extraction_agent, "extract_one", _fake_extract_one)
    monkeypatch.setattr(app_state.extraction_agent, "parse_document", _fake_parse_document)

    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"paper_id": "pregen-paper"},
    )

    assert response.status_code == 200
    saved = next(p for p in app_state.paper_store.list() if p["id"] == "pregen-paper")
    assert isinstance(saved["guide"], dict)
    assert saved["guide"]["sections"]


def test_papers_upload_survives_a_failing_guide_worker(client, app_state, monkeypatch):
    """A guide-generation failure must never affect the paper ingest
    itself - the paper still ends up ready, just without a cached guide,
    and a later on-demand /guide call still works normally."""
    monkeypatch.setattr(app_state.extraction_agent, "extract_one", _fake_extract_one)
    monkeypatch.setattr(app_state.extraction_agent, "parse_document", _fake_parse_document)

    def failing_generate(title, chunks):
        raise RuntimeError("Gemini exploded")

    monkeypatch.setattr(app_state.paper_guide, "generate", failing_generate)

    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"paper_id": "guide-fails-paper"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    saved = next(p for p in app_state.paper_store.list() if p["id"] == "guide-fails-paper")
    assert saved["guide"] is None

    # Undo the failing patch - the real on-demand path (no cache hit,
    # since guide is None) should still work normally afterward.
    monkeypatch.undo()
    guide_response = client.post("/papers/guide-fails-paper/guide")
    assert guide_response.status_code == 200


def test_papers_upload_discards_a_pregenerated_guide_on_extraction_failure(
    client, app_state, monkeypatch
):
    """Extraction failing must still 422 and mark the paper failed even
    when the concurrently-run guide worker succeeds - that guide result
    must be discarded, not leak onto a paper that never really ingested."""
    from agent.extraction_agent import ExtractionIssue, ExtractionOutcome

    def failing_extract_one(paper_id, path, fail_closed=False):
        return ExtractionOutcome(
            paper_id=paper_id,
            issue=ExtractionIssue(
                paper_id=paper_id, stage="document_ingestion", message="bad pdf"
            ),
        )

    monkeypatch.setattr(app_state.extraction_agent, "extract_one", failing_extract_one)
    monkeypatch.setattr(app_state.extraction_agent, "parse_document", _fake_parse_document)

    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"paper_id": "extraction-fails-paper"},
    )

    assert response.status_code == 422
    saved = next(
        p for p in app_state.paper_store.list() if p["id"] == "extraction-fails-paper"
    )
    assert saved["status"] == "failed"
    # fake_db's set(merge=True) replaces the whole document rather than
    # truly merging (documented simplification elsewhere in this test
    # suite), so the failed-status save - which doesn't itself pass
    # guide= - leaves the key absent here rather than explicitly None.
    # Real Firestore's actual merge would keep the earlier guide=None
    # from the initial "processing" save; either way, nothing readable
    # as a real guide payload survives.
    assert saved.get("guide") is None


def test_papers_list_filters_by_session_id(client, app_state):
    app_state.paper_store.save(
        "paper-a", title="Paper A", status="ready", session_id="session-a"
    )
    app_state.paper_store.save(
        "paper-b", title="Paper B", status="ready", session_id="session-b"
    )

    response = client.get("/papers", params={"session_id": "session-a"})

    assert response.status_code == 200
    ids = {p["id"] for p in response.json()}
    assert ids == {"paper-a"}


def test_detach_paper_removes_only_the_given_sessions_membership(client, app_state):
    app_state.paper_store.save(
        "paper-a", title="Paper A", authors="A. Uthor", status="ready", session_id="session-a"
    )

    response = client.post("/papers/paper-a/detach", params={"session_id": "session-a"})

    assert response.status_code == 200
    assert response.json()["session_ids"] == []

    saved = next(p for p in app_state.paper_store.list() if p["id"] == "paper-a")
    assert saved["session_ids"] == []
    assert saved["title"] == "Paper A"
    assert saved["authors"] == "A. Uthor"

    # A subsequent session-scoped listing must no longer include it.
    listing = client.get("/papers", params={"session_id": "session-a"}).json()
    assert listing == []


def test_detach_paper_leaves_a_paper_shared_with_another_session_intact(client, app_state):
    """Adding the same paper to a second session, then detaching it from
    the first, must not remove it from the second - confirmed live this
    was previously the reverse bug (re-ingest silently stole it from the
    first session entirely)."""
    app_state.paper_store.save("paper-a", title="Paper A", status="ready", session_id="session-a")
    app_state.paper_store.save("paper-a", title="Paper A", status="ready", session_id="session-b")

    response = client.post("/papers/paper-a/detach", params={"session_id": "session-a"})

    assert response.status_code == 200
    assert response.json()["session_ids"] == ["session-b"]
    listing = client.get("/papers", params={"session_id": "session-b"}).json()
    assert {p["id"] for p in listing} == {"paper-a"}


def test_detach_paper_404_for_unknown_paper(client):
    response = client.post("/papers/does-not-exist/detach", params={"session_id": "session-a"})
    assert response.status_code == 404


def test_detach_paper_requires_session_id(client, app_state):
    app_state.paper_store.save("paper-a", title="Paper A", status="ready", session_id="session-a")
    response = client.post("/papers/paper-a/detach")
    assert response.status_code == 422


def test_papers_upload_rejects_path_traversal_in_paper_id(client, app_state, tmp_path):
    """A client-supplied paper_id becomes a filename - without
    sanitization, paper_id="../../../../tmp/evil" would let an upload
    write outside upload_root before PdfTextExtractor's allowed_root
    check ever runs (that check only guards the extraction read, not this
    write). Confirms both that the traversal write never happens and that
    nothing lands outside the upload root at all."""
    pdf_bytes = b"%PDF-1.4 fake"
    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
        data={"paper_id": "../../../../tmp/evil"},
    )
    assert response.status_code in (200, 422)

    escaped_path = tmp_path.parent / "evil.pdf"
    assert not escaped_path.exists()
    # Whatever got written landed inside the real, sanitized upload root.
    written = list(Path(app_state.upload_root).glob("*.pdf"))
    assert written
    for path in written:
        assert path.resolve().is_relative_to(Path(app_state.upload_root).resolve())


def test_papers_upload_sanitizes_unsafe_paper_id_characters(client, app_state):
    pdf_bytes = b"%PDF-1.4 fake"
    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
        data={"paper_id": "weird/name with spaces!*.pdf"},
    )
    assert response.status_code in (200, 422)
    if response.status_code == 200:
        # The sanitized id shouldn't contain the raw unsafe characters.
        assert "/" not in response.json()["paper_id"]


def test_paper_status_endpoint_requires_api_key_when_secret_is_set(client, app_state, monkeypatch):
    """Every other /papers route already required this - status was the
    one route missing it, letting anyone poll ingest status for any
    paper_id with no key once API_SHARED_SECRET is set."""
    app_state.paper_store.save("paper-a", title="Paper A", status="ready")
    monkeypatch.setenv("API_SHARED_SECRET", "correct-secret")

    unauthenticated = client.get("/papers/paper-a/status")
    assert unauthenticated.status_code == 401

    authenticated = client.get("/papers/paper-a/status", headers={"X-API-Key": "correct-secret"})
    assert authenticated.status_code == 200
    assert authenticated.json()["status"] == "ready"


def test_require_api_key_rejects_wrong_key(client, monkeypatch):
    monkeypatch.setenv("API_SHARED_SECRET", "correct-secret")
    response = client.post(
        "/query", json={"query": "test"}, headers={"X-API-Key": "wrong"}
    )
    assert response.status_code == 401


def test_require_api_key_accepts_correct_key(client, monkeypatch):
    monkeypatch.setenv("API_SHARED_SECRET", "correct-secret")
    response = client.post(
        "/query", json={"query": "test"}, headers={"X-API-Key": "correct-secret"}
    )
    assert response.status_code == 200


def test_no_api_key_required_when_secret_unset(client, monkeypatch):
    monkeypatch.delenv("API_SHARED_SECRET", raising=False)
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 200


def test_general_chat_uses_shared_fastapi_surface(client, app_state, monkeypatch):
    monkeypatch.setattr(
        app_state.general_chat,
        "answer",
        lambda message, history: f"Gemini says: {message} ({len(history)} prior turns)",
    )
    response = client.post(
        "/chat",
        json={
            "message": "hello",
            "history": [{"role": "user", "text": "earlier"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["retrieval_mode"] == "general"
    assert response.json()["answer"] == "Gemini says: hello (1 prior turns)"


def test_paper_chat_is_scoped_to_requested_paper(client, app_state):
    app_state.chunks.upsert_paper("paper-a", ["Alpha paper studies apples."])
    app_state.chunks.upsert_paper("paper-b", ["Beta paper studies bananas."])

    response = client.post(
        "/chat", json={"message": "What does the paper study?", "paper_ids": ["paper-b"]}
    )

    assert response.status_code == 200
    assert {item["paper_id"] for item in response.json()["citations"]} == {"paper-b"}


def test_chat_drops_paper_ids_not_actually_in_the_given_session(client, app_state):
    """If the client's paper_ids array has drifted from the server's real
    session membership (stale fetch, missed refetch after switching
    sessions), a paper_id that isn't actually in the given session_id
    must be dropped server-side rather than silently answered against -
    a session_id with no matching real membership must not leak evidence
    from a paper the caller doesn't actually have in that session."""
    app_state.chunks.upsert_paper("paper-a", ["Alpha paper studies apples."])
    app_state.chunks.upsert_paper("paper-b", ["Beta paper studies bananas."])
    app_state.paper_store.save("paper-a", title="Alpha", status="ready", session_id="session-a")
    # paper-b deliberately NOT tagged to session-a.

    response = client.post(
        "/chat",
        json={
            "message": "What do the papers study?",
            "paper_ids": ["paper-a", "paper-b"],
            "session_id": "session-a",
        },
    )

    assert response.status_code == 200
    cited_papers = {item["paper_id"] for item in response.json()["citations"]}
    assert cited_papers <= {"paper-a"}
    assert "paper-b" not in cited_papers


def test_chat_without_session_id_keeps_trusting_the_client_paper_ids(client, app_state):
    """session_id is optional - omitting it (unscoped/general chat, or
    any caller with no session concept) must behave exactly as before,
    with no server-side filtering applied."""
    app_state.chunks.upsert_paper("paper-a", ["Alpha paper studies apples."])

    response = client.post(
        "/chat", json={"message": "What does the paper study?", "paper_ids": ["paper-a"]}
    )

    assert response.status_code == 200
    assert {item["paper_id"] for item in response.json()["citations"]} == {"paper-a"}


def test_paper_chat_scopes_to_a_multi_paper_working_set(client, app_state):
    """paper_ids is a session's working set, not a single paper - a query
    should draw evidence from every paper in the set and nothing outside
    it."""
    app_state.chunks.upsert_paper("paper-a", ["Alpha paper studies apples."])
    app_state.chunks.upsert_paper("paper-b", ["Beta paper studies bananas."])
    app_state.chunks.upsert_paper("paper-c", ["Gamma paper studies cherries."])

    response = client.post(
        "/chat",
        json={
            "message": "What do the papers study?",
            "paper_ids": ["paper-a", "paper-b"],
        },
    )

    assert response.status_code == 200
    cited_papers = {item["paper_id"] for item in response.json()["citations"]}
    assert cited_papers <= {"paper-a", "paper-b"}
    assert "paper-c" not in cited_papers


def test_unscoped_chat_searches_the_graph_before_falling_back_to_general_chat(
    client, app_state, monkeypatch
):
    """Regression: /chat with no paper_id used to skip the graph entirely
    and go straight to ungrounded general_chat, even when the message was
    about real content already in the graph (e.g. a gap-suggestion click).
    Confirmed live: this produced a confident but wrong hallucinated answer
    for a real graph entity. general_chat must only be reached when the
    graph genuinely has nothing relevant."""
    app_state.chunks.upsert_paper("paper-a", ["Alpha paper studies apples."])

    def unexpected_general_chat(message, history):
        raise AssertionError(
            "general_chat must not be called when the graph has a real match"
        )

    monkeypatch.setattr(app_state.general_chat, "answer", unexpected_general_chat)

    response = client.post("/chat", json={"message": "What does the paper study?"})

    assert response.status_code == 200
    assert response.json()["retrieval_mode"] != "general"
    assert {item["paper_id"] for item in response.json()["citations"]} == {"paper-a"}


def test_upload_token_is_one_use(client, monkeypatch):
    monkeypatch.setenv("API_SHARED_SECRET", "correct-secret")
    issued = client.post(
        "/papers/upload-token", headers={"X-API-Key": "correct-secret"}
    )
    assert issued.status_code == 200
    token = issued.json()["token"]

    first = client.post(
        "/papers",
        headers={"X-Upload-Token": token},
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert first.status_code in (200, 422)
    second = client.post(
        "/papers",
        headers={"X-Upload-Token": token},
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert second.status_code == 401


def test_upload_rejects_non_pdf_content(client):
    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", b"not a pdf", "application/pdf")},
    )
    assert response.status_code == 415


def test_upload_rejects_oversized_pdf(client, monkeypatch):
    monkeypatch.setattr("service.routers.papers.MAX_PDF_BYTES", 8)
    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", b"%PDF-1234", "application/pdf")},
    )
    assert response.status_code == 413


def test_arxiv_ingest_rejects_invalid_identifier_before_download(client, monkeypatch):
    def unexpected_request(*args, **kwargs):
        raise AssertionError("invalid identifiers must not trigger a download")

    monkeypatch.setattr("service.routers.papers.requests.get", unexpected_request)
    response = client.post(
        "/papers/arxiv", json={"arxiv_id": "../../secret", "title": "Bad"}
    )
    assert response.status_code == 400


def test_arxiv_ingest_has_the_same_path_traversal_guard_as_upload(client, monkeypatch):
    """Regression: ingest_arxiv built its destination path with no
    is_relative_to check, unlike upload_paper's identical write pattern -
    _ARXIV_ID's regex makes this unreachable via a real arXiv id today,
    so exercise it directly by making _sanitize_paper_id (the layer the
    regex is supposed to make redundant) return something unsafe, proving
    the second, independent guard is actually there and would catch it."""
    monkeypatch.setattr(
        "service.routers.papers._sanitize_paper_id", lambda raw: "../../../../tmp/evil"
    )
    response = client.post(
        "/papers/arxiv",
        json={"arxiv_id": "2306.14753", "title": "Test"},
    )
    assert response.status_code == 400


def test_local_frontend_origin_is_allowed_by_cors(client):
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_paper_guide_covers_indexed_paper_and_is_cached(client, app_state, monkeypatch):
    app_state.chunks.upsert_paper(
        "guided-paper",
        ["Introduction and motivation.", "Method and experimental results."],
    )
    app_state.paper_store.save("guided-paper", title="A Guided Paper", status="ready")
    calls = []

    def generate(title, chunks):
        calls.append((title, len(chunks)))
        return PaperGuide(
            title=title,
            big_picture="The paper tests a method.",
            sections=[
                GuideSection(
                    title="Method",
                    plain_language="The method is explained simply.",
                    key_points=["One key point"],
                    why_it_matters="It supports the result.",
                )
            ],
        )

    monkeypatch.setattr(app_state.paper_guide, "generate", generate)
    first = client.post("/papers/guided-paper/guide")
    second = client.post("/papers/guided-paper/guide")

    assert first.status_code == 200
    assert first.json()["big_picture"] == "The paper tests a method."
    assert second.status_code == 200
    assert calls == [("A Guided Paper", 2)]


def test_paper_guide_returns_404_when_paper_has_no_chunks(client):
    response = client.post("/papers/missing/guide")
    assert response.status_code == 404
