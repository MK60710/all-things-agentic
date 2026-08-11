"""Local PDF OCR using Poppler rendering and Tesseract."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from agent.document_ingestion import DocumentExtractionError


class TesseractPdfOcr:
    """Callable OCR fallback with no cloud service or per-page charge."""

    def __init__(self, language: str = "eng", dpi: int = 300):
        self.language = language
        self.dpi = dpi

    def __call__(self, pdf_path: Path) -> str:
        missing = [
            command
            for command in ("pdftoppm", "tesseract")
            if shutil.which(command) is None
        ]
        if missing:
            raise DocumentExtractionError(
                "local OCR requires these commands: " + ", ".join(missing)
            )

        with tempfile.TemporaryDirectory(prefix="research-agent-ocr-") as temp_dir:
            prefix = Path(temp_dir) / "page"
            rendered = subprocess.run(
                [
                    "pdftoppm",
                    "-png",
                    "-r",
                    str(self.dpi),
                    str(pdf_path),
                    str(prefix),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if rendered.returncode != 0:
                raise DocumentExtractionError(
                    "pdftoppm failed during OCR: "
                    + ((rendered.stderr or "unknown error").strip())
                )

            pages = []
            for image_path in sorted(Path(temp_dir).glob("page-*.png")):
                recognized = subprocess.run(
                    [
                        "tesseract",
                        str(image_path),
                        "stdout",
                        "-l",
                        self.language,
                        "--psm",
                        "3",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if recognized.returncode != 0:
                    raise DocumentExtractionError(
                        f"tesseract failed for {image_path.name}: "
                        + ((recognized.stderr or "unknown error").strip())
                    )
                pages.append(recognized.stdout.strip())
            if not pages:
                raise DocumentExtractionError("OCR rendering produced no pages")
            return "\f".join(pages)
