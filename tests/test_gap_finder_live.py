"""Live integration tests against real Vertex AI.

These exist because two real bugs (a 404 on the model name, and Gemini
2.5 Flash's "thinking" tokens silently eating the output budget) were only
ever caught by manually running GeminiExplainer against the real API - the
unit tests in test_gap_finder.py use a fake client and cannot reproduce
either failure mode, since they depend on actual Vertex AI model
availability and actual token-budget consumption behavior.

Skipped by default (RUN_LIVE_TESTS is unset) so `pytest tests/` stays
credential-free, matching the rest of this project's convention. Run with:

    RUN_LIVE_TESTS=1 uv run pytest tests/test_gap_finder_live.py -v

Requires `gcloud auth application-default login` to have been run first.
"""

from __future__ import annotations

import os

import pytest

from agent.gap_finder import GeminiExplainer

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to run tests against real Vertex AI",
)


@pytest.mark.live
def test_live_gemini_explainer_returns_complete_untruncated_text():
    """Would have caught both real bugs found this session: a 404 if the
    configured model name isn't actually available in this project/region,
    and silent truncation if thinking tokens ever start consuming the
    output budget again (e.g. a future model swap re-enables thinking)."""
    explainer = GeminiExplainer()

    result = explainer(
        "Chain of Thought Prompting",
        "Retrieval Augmented Generation",
        ["Large Language Models", "Reasoning"],
    )

    assert result
    assert len(result) > 20  # not a truncated fragment
    # A truncated response reads as an obviously cut-off sentence fragment;
    # the deterministic template's phrasing is the other failure signal
    # (would mean the call silently fell back instead of really answering).
    assert "share context" not in result
