from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.clarification_orchestrator import ClarificationOrchestrator
from agent.document_ingestion import PdfTextExtractor
from agent.extraction_agent import ChunkOnlyStructuredExtractor, ExtractionAgent
from agent.gap_finder import GapFinder
from agent.general_chat import GeneralChatAgent
from agent.graph_manager import GraphManager
from agent.query_agent import QueryAgent
from agent.research_store import ResearchStore
from agent.retrieval import ChunkIndex
from agent.schema import Node, NodeType
from service.app import app
from service.deps import get_state
from service.state import AppState
from service.storage import PaperStore, UploadTokenStore


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
        gap_finder=GapFinder(graph, db_client=fake_db),
        extraction_agent=ExtractionAgent(
            document_extractor=PdfTextExtractor(allowed_root=str(tmp_path)),
            structured_extractor=ChunkOnlyStructuredExtractor(),
        ),
        research_store=ResearchStore(chunks, graph),
        upload_root=str(tmp_path),
        paper_store=PaperStore(fake_db),
        upload_tokens=UploadTokenStore(fake_db),
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


def test_healthz(client):
    response = client.get("/healthz")
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


def test_gaps_empty_graph(client):
    assert client.get("/gaps").json() == []


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
