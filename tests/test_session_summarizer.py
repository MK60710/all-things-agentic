from __future__ import annotations

from agent.session_summarizer import GeminiSessionSummarizer, build_transcript


def test_build_transcript_skips_empty_text_messages():
    messages = [
        {"role": "user", "text": "What is attention?"},
        {"role": "assistant", "text": "  "},
        {"role": "assistant", "guideLoading": True},
    ]

    transcript = build_transcript(messages)

    assert transcript == "user: What is attention?"


def test_build_transcript_formats_role_and_text():
    messages = [
        {"role": "user", "text": "Summarize this paper."},
        {"role": "assistant", "text": "It proposes a new attention mechanism."},
    ]

    transcript = build_transcript(messages)

    assert transcript == (
        "user: Summarize this paper.\n"
        "assistant: It proposes a new attention mechanism."
    )


def test_build_transcript_returns_empty_string_for_no_real_messages():
    assert build_transcript([{"role": "assistant", "text": ""}]) == ""


def test_build_transcript_skips_a_non_string_text_field_instead_of_crashing():
    """messages is a raw list[dict] with no schema enforcement (see
    SessionMessagesRequest's own docstring) - a non-string text field
    (e.g. an int) used to raise AttributeError on .strip(), uncaught,
    before call_structured_judge's own try/except ever got a chance to
    run. Found live via code review, not hypothetical."""
    messages = [
        {"role": "user", "text": 123},
        {"role": "assistant", "text": "A real message."},
    ]

    transcript = build_transcript(messages)

    assert transcript == "assistant: A real message."


def test_gemini_session_summarizer_returns_none_on_empty_transcript():
    # No client/project -> stays client-less, matching GeminiFeynmanJudge's
    # own client-less test pattern. build_transcript returns "" for this
    # input, so __call__ short-circuits before ever attempting a live call.
    summarizer = GeminiSessionSummarizer()

    result = summarizer([{"role": "assistant", "text": ""}])

    assert result is None
