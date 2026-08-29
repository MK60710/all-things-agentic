from __future__ import annotations

from service.state import build_state


def test_build_state_injects_one_gemini_client_everywhere(
    fake_db, monkeypatch, tmp_path
) -> None:
    shared_client = object()
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setenv("UPLOAD_ROOT", str(tmp_path))
    monkeypatch.delenv("PAPER_STORAGE_BUCKET", raising=False)
    monkeypatch.setattr("service.state.firestore.Client", lambda **_: fake_db)
    monkeypatch.setattr("service.state.genai.Client", lambda **_: shared_client)

    state = build_state()

    assert state.gemini_client is shared_client
    assert state.query_agent._client is shared_client
    assert state.general_chat._client is shared_client
    assert state.paper_guide._client is shared_client
    assert state.gap_finder._explain_fn._client is shared_client
    assert state.contradiction_finder._judge._client is shared_client
    assert state.feynman_checker._judge._client is shared_client
    assert state.extraction_agent._structured_extractor._client is shared_client
    assert state.session_summarizer._client is shared_client
