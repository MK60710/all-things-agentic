FROM python:3.11-slim

# poppler-utils provides pdftotext, which PdfTextExtractor shells out to
# for all real extraction. tesseract-ocr is intentionally NOT installed -
# service/state.py constructs PdfTextExtractor with no ocr_fallback, so
# the service never uses OCR in v1. This is a deliberate scope decision,
# not an oversight - add tesseract-ocr here if OCR support is ever wired
# into service/state.py.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY agent/ ./agent/
COPY service/ ./service/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

# Cloud Run injects $PORT at runtime (default 8080) - shell form is
# required for ${PORT:-8080} to actually expand; the exec-form JSON array
# CMD does not perform shell substitution and would otherwise silently
# bind to a literal "${PORT:-8080}".
#
# --workers 1: required, not a tuning choice - see service/state.py's
# module docstring for why more than one worker/instance would split
# in-memory state (ChunkIndex, ClarificationOrchestrator) across
# processes that can't see each other's data.
CMD exec uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
