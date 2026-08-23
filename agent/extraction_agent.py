"""High-level paper extraction orchestration.

This module keeps raw document handling separate from semantic
structuring. The structured extractor is injected so the pipeline can use
Gemini in production, while tests can supply a deterministic stub.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from pydantic import BaseModel, Field

from agent.document_ingestion import (
    DocumentExtractionError,
    DocumentIngestionResult,
    PdfTextExtractor,
)
from agent.schema import ExtractionResult


class StructuredExtractor(Protocol):
    def extract(self, document: DocumentIngestionResult) -> ExtractionResult: ...


class ExtractionIssue(BaseModel):
    paper_id: str
    stage: str
    message: str


class ExtractionOutcome(BaseModel):
    paper_id: str
    document: DocumentIngestionResult | None = None
    result: ExtractionResult | None = None
    issue: ExtractionIssue | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.issue is None


class ExtractionBatchResult(BaseModel):
    outcomes: list[ExtractionOutcome] = Field(default_factory=list)

    @property
    def results(self) -> list[ExtractionResult]:
        return [outcome.result for outcome in self.outcomes if outcome.result is not None]

    @property
    def failures(self) -> list[ExtractionIssue]:
        return [outcome.issue for outcome in self.outcomes if outcome.issue is not None]


class ChunkOnlyStructuredExtractor:
    """Fallback structured extractor that preserves chunks without hallucinating.

    This is useful for wiring and tests. Production should inject a
    semantic extractor that maps the cleaned text into entities and
    relations.
    """

    def extract(self, document: DocumentIngestionResult) -> ExtractionResult:
        return ExtractionResult(
            paper_id=document.paper_id,
            entities=[],
            relations=[],
            chunks=document.chunks,
            chunk_metadata=document.chunk_metadata,
        )


class ExtractionAgent:
    def __init__(
        self,
        document_extractor: PdfTextExtractor | None = None,
        structured_extractor: StructuredExtractor | None = None,
    ):
        self._document_extractor = document_extractor or PdfTextExtractor()
        self._structured_extractor = structured_extractor or ChunkOnlyStructuredExtractor()

    def parse_document(self, paper_id: str, pdf_path: str) -> DocumentIngestionResult:
        """The fast, local (non-LLM) half of extract_one on its own -
        callers that only need chunked text (e.g. guide pre-generation
        run concurrently with the slow structured-extraction call below)
        don't need to wait on or trigger that call."""
        return self._document_extractor.extract(paper_id, pdf_path)

    def extract_one(
        self, paper_id: str, pdf_path: str, *, fail_closed: bool = True
    ) -> ExtractionOutcome:
        try:
            document = self._document_extractor.extract(paper_id, pdf_path)
        except DocumentExtractionError as exc:
            issue = ExtractionIssue(
                paper_id=paper_id,
                stage="document_ingestion",
                message=str(exc),
            )
            return ExtractionOutcome(paper_id=paper_id, issue=issue)
        except Exception as exc:  # pragma: no cover - safety net
            issue = ExtractionIssue(
                paper_id=paper_id,
                stage="document_ingestion",
                message=str(exc),
            )
            return ExtractionOutcome(paper_id=paper_id, issue=issue)

        try:
            result = self._structured_extractor.extract(document)
            if result.paper_id != paper_id:
                result = result.model_copy(update={"paper_id": paper_id})
            if result.skipped_windows:
                # A structured extractor can return a "successful" result
                # that's actually partial - some of its source windows
                # failed and were skipped rather than raising (see
                # GeminiStructuredExtractor). Without this check, a
                # partial extraction is indistinguishable from a clean
                # one at the ExtractionOutcome.ok level, and a caller
                # relying on fail_closed=False's chunks-only fallback for
                # this scenario would never see it triggered.
                issue = ExtractionIssue(
                    paper_id=paper_id,
                    stage="structured_extraction",
                    message=(
                        f"{result.skipped_windows} of the paper's source "
                        "window(s) failed and were skipped; result is partial"
                    ),
                )
                return ExtractionOutcome(
                    paper_id=paper_id, document=document, result=result, issue=issue
                )
            return ExtractionOutcome(paper_id=paper_id, document=document, result=result)
        except Exception as exc:
            issue = ExtractionIssue(
                paper_id=paper_id,
                stage="structured_extraction",
                message=str(exc),
            )
            if fail_closed:
                return ExtractionOutcome(paper_id=paper_id, document=document, issue=issue)
            fallback = ChunkOnlyStructuredExtractor().extract(document)
            return ExtractionOutcome(
                paper_id=paper_id,
                document=document,
                result=fallback,
                issue=issue,
            )

    def extract_batch(
        self, papers: Iterable[tuple[str, str]]
    ) -> ExtractionBatchResult:
        outcomes = [self.extract_one(paper_id, pdf_path) for paper_id, pdf_path in papers]
        return ExtractionBatchResult(outcomes=outcomes)
