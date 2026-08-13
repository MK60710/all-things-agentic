from types import SimpleNamespace

import pytest

from agent.document_ingestion import DocumentIngestionResult
from agent.gemini_extractor import (
    GeminiStructuredExtractor,
    QuoteVerificationError,
    SemanticExtraction,
    has_valid_signature,
    normalize_for_quote_match,
    verify_relations,
)
from agent.schema import (
    EdgeType,
    ExtractedEntity,
    ExtractedRelation,
    NodeType,
)


def _relation(quote: str) -> ExtractedRelation:
    return ExtractedRelation(
        source_entity="GraphRAG",
        source_type=NodeType.METHOD,
        relation=EdgeType.EVALUATES_ON,
        target_entity="HotpotQA",
        target_type=NodeType.BENCHMARK_DATASET,
        source_quote=quote,
    )


def test_quote_verification_handles_pdf_line_hyphenation():
    source = "We evaluate Graph-\nRAG on HotpotQA."
    relation = _relation("We evaluate GraphRAG on HotpotQA.")

    assert verify_relations([relation], source) == [relation]
    assert normalize_for_quote_match(source).startswith("we evaluate graphrag")


def test_quote_verification_rejects_unsupported_relation():
    with pytest.raises(QuoteVerificationError):
        verify_relations([_relation("A fabricated quote.")], "Real source text.")


def test_relation_signature_rejects_wrong_endpoint_types():
    invalid = _relation("We evaluate GraphRAG on HotpotQA.").model_copy(
        update={"target_type": NodeType.CONCEPT}
    )

    assert has_valid_signature(invalid) is False
    assert has_valid_signature(_relation("We evaluate GraphRAG on HotpotQA.")) is True


def test_extractor_caches_calls_and_preserves_chunks():
    semantic = SemanticExtraction(
        entities=[
            ExtractedEntity(
                name="GraphRAG", type=NodeType.METHOD, description="Method"
            ),
            ExtractedEntity(
                name="HotpotQA",
                type=NodeType.BENCHMARK_DATASET,
                description="Benchmark",
            ),
        ],
        relations=[_relation("We evaluate GraphRAG on HotpotQA.")],
    )

    class Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(parsed=semantic, text=semantic.model_dump_json())

    models = Models()
    client = SimpleNamespace(models=models)
    extractor = GeminiStructuredExtractor(project="test", client=client)
    document = DocumentIngestionResult(
        paper_id="paper-1",
        pdf_path="paper.pdf",
        pages=[],
        raw_text="We evaluate GraphRAG on HotpotQA.",
        chunks=["We evaluate GraphRAG on HotpotQA."],
    )

    first = extractor.extract(document)
    second = extractor.extract(document)

    assert len(first.entities) == 2
    assert len(first.relations) == 1
    assert second == first
    assert models.calls == 1


def test_one_truncated_window_does_not_discard_other_windows():
    """A window whose JSON response got cut off by max_output_tokens must
    not take the whole paper's extraction down with it - reproduces a real
    failure found by running structured extraction against an actual
    corpus PDF: one dense window's response was truncated mid-string
    ("EOF while parsing a string"), which previously raised uncaught and
    discarded every other window's already-successfully-extracted
    entities/relations too."""
    good_semantic = SemanticExtraction(
        entities=[
            ExtractedEntity(
                name="GraphRAG", type=NodeType.METHOD, description="Method"
            ),
        ],
        relations=[],
    )

    class Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                # Truncated JSON, exactly the shape a cut-off response
                # produces - unterminated string, no closing braces.
                return SimpleNamespace(
                    parsed=None,
                    text='{\n  "entities": [\n    {"name": "cut off mid',
                )
            return SimpleNamespace(parsed=good_semantic, text=good_semantic.model_dump_json())

    models = Models()
    client = SimpleNamespace(models=models)
    extractor = GeminiStructuredExtractor(
        project="test", client=client, max_characters_per_call=10
    )
    document = DocumentIngestionResult(
        paper_id="paper-1",
        pdf_path="paper.pdf",
        pages=[],
        raw_text="first window text here. second window text here.",
        chunks=["first window text here.", "second window text here."],
    )

    result = extractor.extract(document)

    assert models.calls == 2
    assert len(result.entities) == 1
    assert result.entities[0].name == "GraphRAG"
    # A silently partial result must not look identical to a clean one -
    # ExtractionAgent relies on this to avoid reporting ok=True when a
    # window was actually dropped.
    assert result.skipped_windows == 1


def test_non_validation_error_on_one_window_also_does_not_discard_others():
    """A transient API error (rate limit, timeout, safety block) on one
    window must be handled the same way as a truncated-JSON ValidationError
    - not just JSON-parse failures. A narrower except clause here would
    reproduce the exact whole-paper data loss this handling exists to
    prevent, just triggered by a different exception type."""
    good_semantic = SemanticExtraction(
        entities=[
            ExtractedEntity(
                name="GraphRAG", type=NodeType.METHOD, description="Method"
            ),
        ],
        relations=[],
    )

    class Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("503 RESOURCE_EXHAUSTED")
            return SimpleNamespace(parsed=good_semantic, text=good_semantic.model_dump_json())

    models = Models()
    client = SimpleNamespace(models=models)
    extractor = GeminiStructuredExtractor(
        project="test", client=client, max_characters_per_call=10
    )
    document = DocumentIngestionResult(
        paper_id="paper-1",
        pdf_path="paper.pdf",
        pages=[],
        raw_text="first window text here. second window text here.",
        chunks=["first window text here.", "second window text here."],
    )

    result = extractor.extract(document)

    assert models.calls == 2
    assert len(result.entities) == 1
    assert result.entities[0].name == "GraphRAG"
    assert result.skipped_windows == 1
