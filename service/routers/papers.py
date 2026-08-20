from __future__ import annotations

import os
import re
import secrets
import uuid
from collections.abc import Iterable
from pathlib import Path

import requests
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from service.deps import get_state, require_api_key
from service.schemas import (
    ArxivIngestRequest,
    PaperIngestResponse,
    PaperMetadata,
    UploadTokenResponse,
)
from service.state import AppState
from agent.paper_guide import PaperGuide

router = APIRouter(prefix="/papers", tags=["papers"])

MAX_PDF_BYTES = 25 * 1024 * 1024
_UNSAFE_PAPER_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_ARXIV_ID = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9.-]*/\d{7})(?:v\d+)?$"
)


def _sanitize_paper_id(raw: str) -> str:
    cleaned = _UNSAFE_PAPER_ID_CHARS.sub("_", raw).strip("._")
    return cleaned or str(uuid.uuid4())


def _write_pdf(dest: Path, chunks: Iterable[bytes], *, max_bytes: int) -> None:
    size = 0
    signature = b""
    try:
        with dest.open("wb") as output:
            for chunk in chunks:
                if not chunk:
                    continue
                if len(signature) < 5:
                    signature += chunk[: 5 - len(signature)]
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail="PDF exceeds the 25 MiB limit")
                output.write(chunk)
        if signature != b"%PDF-":
            raise HTTPException(status_code=415, detail="file is not a valid PDF")
    except Exception:
        dest.unlink(missing_ok=True)
        raise


def _file_chunks(file: UploadFile, chunk_size: int = 1024 * 1024) -> Iterable[bytes]:
    while chunk := file.file.read(chunk_size):
        yield chunk


def _ingest(
    state: AppState,
    *,
    paper_id: str,
    path: Path,
    title: str,
    authors: str | None = None,
    abstract: str | None = None,
    pdf_url: str | None = None,
) -> PaperIngestResponse:
    metadata = dict(title=title, authors=authors, abstract=abstract, pdf_url=pdf_url)
    state.paper_store.save(paper_id, **metadata, status="processing", guide=None)
    outcome = state.extraction_agent.extract_one(paper_id, str(path), fail_closed=False)
    if outcome.result is None:
        message = outcome.issue.message if outcome.issue else "extraction failed"
        state.paper_store.save(paper_id, **metadata, status="failed", error=message)
        raise HTTPException(status_code=422, detail=message)

    pending_before = len(state.clarification.pending())
    report = state.research_store.ingest(
        outcome.result,
        paper_name=title,
        entity_embedding_fn=state.entity_embedding_fn,
        clarification=state.clarification,
    )
    pending_added = len(state.clarification.pending()) - pending_before
    state.paper_store.save(paper_id, **metadata, status="ready")
    return PaperIngestResponse(
        paper_id=paper_id,
        id=paper_id,
        title=title,
        authors=authors,
        abstract=abstract,
        pdf_url=pdf_url,
        status="ready",
        extraction_ok=outcome.ok,
        issue_message=outcome.issue.message if outcome.issue else None,
        chunk_ids=report.chunk_ids,
        entities_added=len(outcome.result.entities),
        relations_added=len(outcome.result.relations),
        pending_clarification_count=pending_added,
    )


def _authorize_upload(state: AppState, *, x_api_key: str, x_upload_token: str) -> int:
    expected = os.environ.get("API_SHARED_SECRET")
    if expected and x_api_key and secrets.compare_digest(x_api_key, expected):
        return MAX_PDF_BYTES
    if not expected:
        return MAX_PDF_BYTES
    constraints = state.upload_tokens.consume(x_upload_token)
    if constraints is None:
        raise HTTPException(status_code=401, detail="invalid or expired upload token")
    return min(int(constraints.get("max_bytes", MAX_PDF_BYTES)), MAX_PDF_BYTES)


@router.get("", response_model=list[PaperMetadata], dependencies=[Depends(require_api_key)])
def list_papers(state: AppState = Depends(get_state)) -> list[PaperMetadata]:
    return [PaperMetadata.model_validate(item) for item in state.paper_store.list()]


@router.post(
    "/upload-token",
    response_model=UploadTokenResponse,
    dependencies=[Depends(require_api_key)],
)
def create_upload_token(state: AppState = Depends(get_state)) -> UploadTokenResponse:
    token, expires_at = state.upload_tokens.issue(max_bytes=MAX_PDF_BYTES)
    return UploadTokenResponse(token=token, expires_at=expires_at.isoformat(), max_bytes=MAX_PDF_BYTES)


@router.post("", response_model=PaperIngestResponse)
def upload_paper(
    file: UploadFile = File(...),
    paper_id: str | None = Form(default=None),
    title: str | None = Form(default=None),
    x_api_key: str = Header(default=""),
    x_upload_token: str = Header(default=""),
    state: AppState = Depends(get_state),
) -> PaperIngestResponse:
    max_bytes = _authorize_upload(state, x_api_key=x_api_key, x_upload_token=x_upload_token)
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="only PDF uploads are accepted")

    raw_pid = paper_id or (Path(file.filename).stem if file.filename else str(uuid.uuid4()))
    pid = _sanitize_paper_id(raw_pid)
    upload_root = Path(state.upload_root).resolve()
    dest = (upload_root / f"{pid}.pdf").resolve()
    if not dest.is_relative_to(upload_root):
        raise HTTPException(status_code=400, detail="invalid paper_id")
    _write_pdf(dest, _file_chunks(file), max_bytes=max_bytes)
    paper_title = (title or (Path(file.filename).stem if file.filename else pid)).strip()
    return _ingest(state, paper_id=pid, path=dest, title=paper_title or pid)


@router.post(
    "/arxiv",
    response_model=PaperIngestResponse,
    dependencies=[Depends(require_api_key)],
)
def ingest_arxiv(body: ArxivIngestRequest, state: AppState = Depends(get_state)) -> PaperIngestResponse:
    arxiv_id = body.arxiv_id.removeprefix("arxiv:").strip()
    if not _ARXIV_ID.fullmatch(arxiv_id):
        raise HTTPException(status_code=400, detail="invalid arXiv identifier")
    pid = _sanitize_paper_id(f"arxiv-{arxiv_id}")
    upload_root = Path(state.upload_root).resolve()
    dest = (upload_root / f"{pid}.pdf").resolve()
    if not dest.is_relative_to(upload_root):
        raise HTTPException(status_code=400, detail="invalid arXiv identifier")
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "AtlasResearchAssistant/0.1"},
            stream=True,
            timeout=(5, 45),
        )
        response.raise_for_status()
        _write_pdf(dest, response.iter_content(chunk_size=1024 * 1024), max_bytes=MAX_PDF_BYTES)
    except HTTPException:
        raise
    except requests.RequestException as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail="could not download arXiv PDF") from exc
    return _ingest(
        state,
        paper_id=pid,
        path=dest,
        title=body.title,
        authors=body.authors,
        abstract=body.abstract,
        pdf_url=body.pdf_url or url,
    )


@router.post(
    "/{paper_id}/guide",
    response_model=PaperGuide,
    dependencies=[Depends(require_api_key)],
)
def create_paper_guide(
    paper_id: str, state: AppState = Depends(get_state)
) -> PaperGuide:
    metadata = state.paper_store.get(paper_id)
    chunks = state.chunks.paper_chunks(paper_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="paper has no indexed content")
    if metadata and isinstance(metadata.get("guide"), dict):
        return PaperGuide.model_validate(metadata["guide"])
    title = str((metadata or {}).get("title") or paper_id)
    guide = state.paper_guide.generate(title, chunks)
    state.paper_store.save(paper_id, guide=guide.model_dump(mode="json"))
    return guide
