from __future__ import annotations

from types import SimpleNamespace

from agent.document_ingestion import DocumentExtractionError, DocumentIngestionResult
from agent.extraction_agent import ExtractionAgent
from agent.schema import (
    EdgeType,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
    NodeType,
)


class _FakeDocumentExtractor:
    def __init__(self, failures: set[str] | None = None):
        self._failures = failures or set()

    def extract(self, paper_id: str, pdf_path: str) -> DocumentIngestionResult:
        if pdf_path in self._failures:
            raise DocumentExtractionError("boom")
        return DocumentIngestionResult(
            paper_id=paper_id,
            pdf_path=pdf_path,
            pages=[],
            raw_text="clean text",
            chunks=["chunk one", "chunk two"],
        )


class _FakeStructuredExtractor:
    def extract(self, document: DocumentIngestionResult) -> ExtractionResult:
        return ExtractionResult(
            paper_id=document.paper_id,
            entities=[
                ExtractedEntity(
                    name="Chain of Thought",
                    type=NodeType.CONCEPT,
                    description="Reasoning style",
                ),
                ExtractedEntity(
                    name="Reasoning",
                    type=NodeType.CONCEPT,
                    description="Inference process",
                ),
            ],
            relations=[
                ExtractedRelation(
                    source_entity="Chain of Thought",
                    relation=EdgeType.SUPPORTS,
                    target_entity="Reasoning",
                    source_quote="Chain of thought prompts can support reasoning.",
                    source_section="Introduction",
                )
            ],
            chunks=document.chunks,
        )


class _FailingStructuredExtractor:
    def extract(self, document: DocumentIngestionResult) -> ExtractionResult:
        raise RuntimeError("structured extraction failed")


def test_extract_one_uses_structured_extractor():
    agent = ExtractionAgent(
        document_extractor=_FakeDocumentExtractor(),
        structured_extractor=_FakeStructuredExtractor(),
    )

    outcome = agent.extract_one("paper-1", "paper.pdf")

    assert outcome.ok is True
    assert outcome.result is not None
    assert outcome.result.paper_id == "paper-1"
    assert len(outcome.result.entities) == 2
    assert outcome.result.chunks == ["chunk one", "chunk two"]


class _FailingStructuredExtractor:
    def extract(self, document: DocumentIngestionResult) -> ExtractionResult:
        raise RuntimeError("structured extraction blew up")


def test_extract_one_fail_closed_true_returns_no_result():
    agent = ExtractionAgent(
        document_extractor=_FakeDocumentExtractor(),
        structured_extractor=_FailingStructuredExtractor(),
    )

    outcome = agent.extract_one("paper-1", "paper.pdf", fail_closed=True)

    assert outcome.ok is False
    assert outcome.result is None
    assert outcome.document is not None
    assert outcome.issue is not None
    assert outcome.issue.stage == "structured_extraction"


def test_extract_one_fail_closed_false_falls_back_to_chunks_only():
    """fail_closed=False must actually behave differently from True -
    degrade to a chunks-only result instead of dropping the paper, while
    still reporting the issue so callers can tell it wasn't a clean
    success."""
    agent = ExtractionAgent(
        document_extractor=_FakeDocumentExtractor(),
        structured_extractor=_FailingStructuredExtractor(),
    )

    outcome = agent.extract_one("paper-1", "paper.pdf", fail_closed=False)

    assert outcome.ok is False  # issue is still set
    assert outcome.result is not None
    assert outcome.result.paper_id == "paper-1"
    assert outcome.result.entities == []
    assert outcome.result.relations == []
    assert outcome.result.chunks == ["chunk one", "chunk two"]
    assert outcome.issue is not None
    assert outcome.issue.stage == "structured_extraction"


def test_extract_batch_isolates_failure_per_paper():
    agent = ExtractionAgent(
        document_extractor=_FakeDocumentExtractor(failures={"bad.pdf"}),
        structured_extractor=_FakeStructuredExtractor(),
    )

    batch = agent.extract_batch(
        [("paper-1", "good.pdf"), ("paper-2", "bad.pdf")]
    )

    assert len(batch.outcomes) == 2
    assert len(batch.results) == 1
    assert len(batch.failures) == 1
    assert batch.failures[0].paper_id == "paper-2"
    assert batch.failures[0].stage == "document_ingestion"


def test_extract_one_can_degrade_to_chunks_when_structuring_fails():
    agent = ExtractionAgent(
        document_extractor=_FakeDocumentExtractor(),
        structured_extractor=_FailingStructuredExtractor(),
    )

    outcome = agent.extract_one("paper-1", "paper.pdf", fail_closed=False)

    assert outcome.result is not None
    assert outcome.result.chunks == ["chunk one", "chunk two"]
    assert outcome.issue is not None
    assert outcome.issue.stage == "structured_extraction"


def test_extract_one_fails_closed_when_structuring_fails():
    agent = ExtractionAgent(
        document_extractor=_FakeDocumentExtractor(),
        structured_extractor=_FailingStructuredExtractor(),
    )

    outcome = agent.extract_one("paper-1", "paper.pdf", fail_closed=True)

    assert outcome.result is None
    assert outcome.issue is not None
