"""Deterministic PDF text ingestion with an optional OCR fallback.

The goal is to keep raw document handling cheap and predictable:
extract embedded text first, fall back to OCR only when the document has
too little usable text, then chunk the cleaned text for downstream
structuring.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from agent.schema import ExtractionChunk

try:
    import pymupdf
except ImportError:  # pragma: no cover - pdftotext remains available
    pymupdf = None


class DocumentExtractionError(RuntimeError):
    """Raised when a PDF cannot be turned into usable text."""


OcrFallbackFn = Callable[[Path], str]


class DocumentPage(BaseModel):
    page_number: int
    text: str
    source: str = "pdf_text"


class DocumentIngestionResult(BaseModel):
    paper_id: str
    pdf_path: str
    pages: list[DocumentPage]
    raw_text: str
    chunks: list[str] = Field(default_factory=list)
    chunk_metadata: list[ExtractionChunk] = Field(default_factory=list)
    used_ocr: bool = False
    warnings: list[str] = Field(default_factory=list)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(
    text: str, max_chars: int = 3500, overlap_chars: int = 250
) -> list[str]:
    """Chunk text on paragraph boundaries while keeping chunks compact.

    This is intentionally simple: we want stable, low-cost chunking before
    any semantic processing happens.
    """

    cleaned = _normalize_text(text)
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()
            start = 0
            step = max(1, max_chars - overlap_chars)
            while start < len(paragraph):
                chunks.append(paragraph[start : start + max_chars].strip())
                start += step
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        flush()
        current = paragraph

    flush()
    return chunks


_SECTION_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s+)?"
    r"(?:abstract|introduction|background|related work|method(?:ology)?|"
    r"approach|experiments?|experimental results?|evaluation|results?|discussion|limitations?|"
    r"conclusion|references|appendix)\b.*$",
    re.IGNORECASE,
)


def chunk_pages(
    pages: list[DocumentPage], max_chars: int = 3500
) -> list[ExtractionChunk]:
    """Create page- and section-aware chunks without a model call."""

    records: list[ExtractionChunk] = []
    current_section: str | None = None
    for page in pages:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", page.text)
            if paragraph.strip()
        ]
        section_parts: list[str] = []

        def flush() -> None:
            if not section_parts:
                return
            text = "\n\n".join(section_parts)
            for piece in chunk_text(text, max_chars=max_chars):
                records.append(
                    ExtractionChunk(
                        text=piece,
                        ordinal=len(records),
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section=current_section,
                        source=page.source,
                    )
                )
            section_parts.clear()

        for paragraph in paragraphs:
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            first_line = lines[0]
            heading = first_line
            if (
                re.fullmatch(r"\d+(?:\.\d+)*[.)]?", first_line)
                and len(lines) > 1
            ):
                heading = f"{first_line} {lines[1]}"
            if len(heading) <= 100 and _SECTION_HEADING.fullmatch(heading):
                flush()
                current_section = heading
            section_parts.append(paragraph)
        flush()
    return records


class PdfTextExtractor:
    """Extracts text using ``pdftotext`` and optionally OCRs low-text PDFs."""

    def __init__(
        self,
        ocr_fallback: OcrFallbackFn | None = None,
        min_usable_chars: int = 200,
        allowed_root: str | Path | None = None,
    ):
        self._ocr_fallback = ocr_fallback
        self._min_usable_chars = min_usable_chars
        # Off by default so today's hardcoded-manifest caller is unaffected.
        # Set this once pdf_path can come from anything other than the fixed
        # corpus manifest, so extract() can't be pointed at an arbitrary
        # local file (path traversal / arbitrary file read).
        self._allowed_root = Path(allowed_root).resolve() if allowed_root else None

    def extract(self, paper_id: str, pdf_path: str | Path) -> DocumentIngestionResult:
        path = Path(pdf_path)
        if not path.exists():
            raise DocumentExtractionError(f"PDF not found: {path}")
        if self._allowed_root is not None:
            resolved = path.resolve()
            if self._allowed_root not in resolved.parents and resolved != self._allowed_root:
                raise DocumentExtractionError(
                    f"pdf_path {resolved} is outside the allowed root {self._allowed_root}"
                )

        warnings: list[str] = []
        try:
            extracted = self._extract_with_layout(path)
        except DocumentExtractionError as exc:
            if self._ocr_fallback is None:
                raise
            extracted = _normalize_text(self._ocr_fallback(path))
            warnings.append(f"pdftotext_unavailable: {exc}")
            warnings.append("used_ocr_fallback")
            pages = self._split_pages(extracted, source="ocr")
            metadata = chunk_pages(pages)
            return DocumentIngestionResult(
                paper_id=paper_id,
                pdf_path=str(path),
                pages=pages,
                raw_text=extracted,
                chunks=[chunk.text for chunk in metadata],
                chunk_metadata=metadata,
                used_ocr=True,
                warnings=warnings,
            )

        used_ocr = False

        if len(re.sub(r"\s+", "", extracted)) < self._min_usable_chars:
            if self._ocr_fallback is None:
                raise DocumentExtractionError(
                    "PDF text extraction produced too little usable text and no "
                    "OCR fallback was configured"
                )
            extracted = self._ocr_fallback(path)
            extracted = _normalize_text(extracted)
            used_ocr = True
            warnings.append("used_ocr_fallback")

        pages = self._split_pages(extracted, source="ocr" if used_ocr else "pdf_text")
        metadata = chunk_pages(pages)
        return DocumentIngestionResult(
            paper_id=paper_id,
            pdf_path=str(path),
            pages=pages,
            raw_text=extracted,
            chunks=[chunk.text for chunk in metadata],
            chunk_metadata=metadata,
            used_ocr=used_ocr,
            warnings=warnings,
        )

    def _extract_with_pdftotext(self, pdf_path: Path) -> str:
        try:
            completed = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise DocumentExtractionError("pdftotext is not installed") from exc
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise DocumentExtractionError(
                f"pdftotext failed for {pdf_path.name}: {stderr or 'unknown error'}"
            )
        return _normalize_text(completed.stdout or "")

    def _extract_with_layout(self, pdf_path: Path) -> str:
        if pymupdf is None:
            return self._extract_with_pdftotext(pdf_path)
        try:
            with pymupdf.open(pdf_path) as document:
                pages = [self._layout_page_text(page) for page in document]
        except Exception:
            return self._extract_with_pdftotext(pdf_path)
        extracted = _normalize_text("\f".join(pages))
        if extracted:
            return extracted
        return self._extract_with_pdftotext(pdf_path)

    def _layout_page_text(self, page) -> str:
        blocks = [
            block
            for block in page.get_text("blocks", sort=True)
            if len(block) >= 7 and block[6] == 0 and block[4].strip()
        ]
        if not blocks:
            return ""

        midpoint = page.rect.width / 2
        margin = page.rect.width * 0.04
        full_width, left, right = [], [], []
        for block in blocks:
            x0, _, x1, _, text, *_ = block
            item = (*block[:4], text.strip())
            if x0 < midpoint - margin and x1 > midpoint + margin:
                full_width.append(item)
            elif x1 <= midpoint + margin:
                left.append(item)
            elif x0 >= midpoint - margin:
                right.append(item)
            else:
                full_width.append(item)

        column_top = min(
            [block[1] for block in left + right], default=float("inf")
        )
        headers = [block for block in full_width if block[1] < column_top]
        remaining = [block for block in full_width if block[1] >= column_top]
        ordered = (
            sorted(headers, key=lambda block: (block[1], block[0]))
            + sorted(left, key=lambda block: (block[1], block[0]))
            + sorted(right, key=lambda block: (block[1], block[0]))
            + sorted(remaining, key=lambda block: (block[1], block[0]))
        )
        return "\n\n".join(block[4] for block in ordered)

    def _split_pages(self, text: str, source: str = "pdf_text") -> list[DocumentPage]:
        if not text.strip():
            return []

        raw_pages = text.split("\f")
        pages = []
        for index, page_text in enumerate(raw_pages, start=1):
            cleaned = page_text.strip()
            if cleaned:
                pages.append(DocumentPage(page_number=index, text=cleaned, source=source))
        if not pages and text.strip():
            pages.append(DocumentPage(page_number=1, text=text.strip(), source=source))
        return pages
