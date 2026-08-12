from pathlib import Path

import pytest

from agent.document_ingestion import DocumentExtractionError
from agent.local_ocr import TesseractPdfOcr


def test_local_ocr_reports_missing_system_commands(monkeypatch):
    monkeypatch.setattr("agent.local_ocr.shutil.which", lambda command: None)

    with pytest.raises(DocumentExtractionError, match="pdftoppm, tesseract"):
        TesseractPdfOcr()(Path("scan.pdf"))
