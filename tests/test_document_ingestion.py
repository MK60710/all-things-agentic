from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent.document_ingestion import (
    DocumentExtractionError,
    DocumentPage,
    PdfTextExtractor,
    chunk_pages,
    chunk_text,
)


def test_chunk_text_splits_large_text():
    text = "\n\n".join([f"Paragraph {index}" for index in range(8)])
    chunks = chunk_text(text, max_chars=25, overlap_chars=5)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)


def test_chunk_pages_preserves_page_and_section_metadata():
    records = chunk_pages(
        [
            DocumentPage(
                page_number=2,
                text="2 Methodology\n\nWe train the model.\n\nMore details.",
            ),
            DocumentPage(page_number=3, text="The method continues here."),
        ]
    )

    assert records[0].section == "2 Methodology"
    assert records[0].page_start == 2
    assert records[-1].section == "2 Methodology"
    assert records[-1].page_start == 3
    assert [record.ordinal for record in records] == list(range(len(records)))


def test_chunk_pages_detects_split_numbered_heading():
    records = chunk_pages(
        [DocumentPage(page_number=4, text="3\nExperimental Results\nDetails.")]
    )

    assert records[0].section == "3 Experimental Results"


def test_pdf_text_extractor_uses_pdftotext_output(tmp_path, monkeypatch):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="Title\fBody text", stderr="")

    monkeypatch.setattr(
        "agent.document_ingestion.subprocess.run",
        fake_run,
    )

    extractor = PdfTextExtractor(min_usable_chars=0)
    result = extractor.extract("paper-1", pdf_path)

    assert result.paper_id == "paper-1"
    assert result.raw_text == "Title\fBody text"
    assert [page.page_number for page in result.pages] == [1, 2]
    assert result.chunks
    assert result.used_ocr is False


def test_pdf_text_extractor_falls_back_to_ocr(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=" ", stderr="")

    monkeypatch.setattr(
        "agent.document_ingestion.subprocess.run",
        fake_run,
    )

    extractor = PdfTextExtractor(
        ocr_fallback=lambda path: "OCR recovered text",
        min_usable_chars=20,
    )
    result = extractor.extract("paper-ocr", pdf_path)

    assert result.used_ocr is True
    assert result.raw_text == "OCR recovered text"
    assert "used_ocr_fallback" in result.warnings


def test_pdf_text_extractor_uses_ocr_when_pdftotext_is_missing(
    tmp_path, monkeypatch
):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("pdftotext")

    monkeypatch.setattr(
        "agent.document_ingestion.subprocess.run",
        fake_run,
    )

    extractor = PdfTextExtractor(ocr_fallback=lambda path: "OCR fallback text")
    result = extractor.extract("paper-ocr", pdf_path)

    assert result.used_ocr is True
    assert result.raw_text == "OCR fallback text"
    assert any("pdftotext_unavailable" in warning for warning in result.warnings)


def test_pdf_text_extractor_fails_without_pdf(tmp_path, monkeypatch):
    missing = tmp_path / "missing.pdf"

    extractor = PdfTextExtractor(min_usable_chars=0)

    try:
        extractor.extract("paper-missing", missing)
    except DocumentExtractionError as exc:
        assert "PDF not found" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DocumentExtractionError")
