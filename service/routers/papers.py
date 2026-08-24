from __future__ import annotations

import logging
import os
import re
import secrets
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile

from service.deps import get_state, require_api_key
from service.schemas import (
    ArxivIngestRequest,
    GraphVizEdge,
    GraphVizNode,
    PaperIngestResponse,
    PaperMetadata,
    UploadTokenResponse,
)
from service.state import AppState
from service.storage import _paper_session_ids
from agent.paper_guide import PaperGuide
from agent.retrieval import ChunkRecord
from agent.schema import ExtractionChunk

logger = logging.getLogger(__name__)

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


def _pregenerate_guide(
    state: AppState, *, paper_id: str, path: Path, title: str
) -> PaperGuide | None:
    """Runs concurrently with extract_one (see _ingest) - independently
    re-parses the same PDF (fast, local, not an LLM call) rather than
    waiting on extraction's document, so the guide's own slow Gemini call
    overlaps with extraction's instead of following it. Any failure here
    - a bad PDF, a Gemini error - just means no guide yet, exactly like
    today's on-demand /guide endpoint failing; it must never affect
    whether the paper itself ingests successfully.

    Builds its own throwaway ChunkRecords in memory rather than calling
    ChunkIndex.upsert_paper - PaperGuideAgent.generate only reads
    ordinal/text/page metadata off them, never persisted state, and this
    runs before extraction's own pass/fail is known. Persisting chunks
    here would leave them reachable by chat/search even when extraction
    goes on to fail and the paper is never actually ready - real
    research_store.ingest (the durable, session-scoped write) still only
    ever runs after a successful extraction, unchanged.
    """
    try:
        document = state.extraction_agent.parse_document(paper_id, str(path))
        raw_chunks = document.chunk_metadata or [
            ExtractionChunk(text=text, ordinal=index)
            for index, text in enumerate(document.chunks)
        ]
        chunks = [
            ChunkRecord(
                id=f"guide-preview:{paper_id}:{index}",
                paper_id=paper_id,
                ordinal=chunk.ordinal,
                text=chunk.text,
                embedding=[],
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                source=chunk.source,
                content_type=chunk.content_type,
            )
            for index, chunk in enumerate(raw_chunks)
        ]
        if not chunks:
            return None
        return state.paper_guide.generate(title, chunks)
    except Exception:
        logger.warning(
            "Concurrent guide pre-generation failed for %s", paper_id, exc_info=True
        )
        return None


def _ingest(
    state: AppState,
    *,
    paper_id: str,
    path: Path,
    title: str,
    authors: str | None = None,
    abstract: str | None = None,
    pdf_url: str | None = None,
    session_id: str | None = None,
) -> PaperIngestResponse:
    metadata = dict(title=title, authors=authors, abstract=abstract, pdf_url=pdf_url)
    # guide=None invalidates any previously generated walkthrough on a
    # re-ingest of the same paper_id - stale content shouldn't be served
    # from cache once the underlying extraction reruns.
    state.paper_store.save(paper_id, **metadata, status="extracting", session_id=session_id, guide=None)
    # Entity extraction and the guide walkthrough are two independent
    # slow Gemini calls over the same PDF - neither depends on the
    # other's output (see _pregenerate_guide) - so they run concurrently
    # instead of back-to-back. Wall-clock cost becomes whichever is
    # slower, not both summed.
    with ThreadPoolExecutor(max_workers=2) as pool:
        extraction_future = pool.submit(
            state.extraction_agent.extract_one, paper_id, str(path), fail_closed=False
        )
        guide_future = pool.submit(
            _pregenerate_guide, state, paper_id=paper_id, path=path, title=title
        )
        outcome = extraction_future.result()
        guide = guide_future.result()

    if outcome.result is None:
        # Extraction genuinely failed - discard any concurrently-computed
        # guide rather than caching a walkthrough for a paper that never
        # successfully ingested.
        message = outcome.issue.message if outcome.issue else "extraction failed"
        state.paper_store.save(paper_id, **metadata, status="failed", error=message, session_id=session_id)
        raise HTTPException(status_code=422, detail=message)

    pending_before = len(state.clarification.pending())
    report = state.research_store.ingest(
        outcome.result,
        paper_name=title,
        entity_embedding_fn=state.entity_embedding_fn,
        clarification=state.clarification,
        session_id=session_id,
    )
    pending_added = len(state.clarification.pending()) - pending_before
    # Caches the concurrently pre-generated guide (None if that worker
    # failed) so the frontend's separate POST /papers/{id}/guide call -
    # unchanged, still fires after this response - finds it already
    # warm via create_paper_guide's existing cache check, instead of
    # running a second full generation.
    guide_payload = guide.model_dump(mode="json") if guide is not None else None
    state.paper_store.save(
        paper_id, **metadata, status="ready", session_id=session_id, guide=guide_payload
    )

    new_nodes: list[GraphVizNode] = []
    new_edges: list[GraphVizEdge] = []
    if report.graph is not None:
        new_nodes = [
            GraphVizNode(
                node_id=write.node_id,
                name=write.entity_name,
                type=state.graph.graph.nodes.get(write.node_id, {}).get("type"),
                reused_existing_node=write.reused_existing_node,
            )
            for write in report.graph.node_writes
        ]
        new_edges = [
            GraphVizEdge(
                edge_id=write.edge_id,
                source_id=write.source_id,
                target_id=write.target_id,
                relation=write.relation,
            )
            for write in report.graph.edge_writes
        ]
        # _resolve_relation_endpoint (agent/graph_manager.py) can create an
        # "implicit relation endpoint" node directly via add_node() when a
        # relation names an entity outside the extracted entities list -
        # that node is real and used as a real edge endpoint, but never
        # gets a NodeWriteResult, so node_writes alone isn't a complete
        # picture of every node this ingest touched. Backfill any edge
        # endpoint missing from new_nodes, or the frontend's force-graph
        # renders a link to a node it never received.
        covered_ids = {node.node_id for node in new_nodes}
        missing_ids = {
            node_id
            for edge in new_edges
            for node_id in (edge.source_id, edge.target_id)
            if node_id not in covered_ids
        }
        for node_id in missing_ids:
            data = state.graph.graph.nodes.get(node_id)
            if data is None:
                continue
            new_nodes.append(
                GraphVizNode(
                    node_id=node_id,
                    name=data.get("name", node_id),
                    type=data.get("type"),
                    reused_existing_node=True,
                )
            )
            covered_ids.add(node_id)

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
        new_nodes=new_nodes,
        new_edges=new_edges,
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
def list_papers(
    session_id: str | None = Query(default=None),
    state: AppState = Depends(get_state),
) -> list[PaperMetadata]:
    papers = state.paper_store.list()
    if session_id is not None:
        papers = [p for p in papers if session_id in _paper_session_ids(p)]
    return [PaperMetadata.model_validate(item) for item in papers]


@router.get("/{paper_id}/status", dependencies=[Depends(require_api_key)])
def get_paper_status(paper_id: str, state: AppState = Depends(get_state)) -> dict[str, str]:
    """Lightweight poll target for ingest progress - deliberately just the
    status string, not the full PaperMetadata list_papers already returns,
    so the frontend can poll every second or two during a "Reading..."
    card without pulling the whole papers list on each tick."""
    paper = state.paper_store.get(paper_id)
    return {"status": paper.get("status", "unknown") if paper else "unknown"}


@router.post(
    "/{paper_id}/detach",
    response_model=PaperMetadata,
    dependencies=[Depends(require_api_key)],
)
def detach_paper(
    paper_id: str, session_id: str = Query(...), state: AppState = Depends(get_state)
) -> PaperMetadata:
    """Removes this one session's membership - the real, server-side
    version of "remove this from my session," so it stays gone on the
    next listPapersForSession(session_id) fetch instead of reappearing
    on a later switch. A paper genuinely shared with another session
    (added there separately) survives; only detaching from every session
    that has it makes it disappear entirely - see PaperStore.detach_session."""
    existing = next((p for p in state.paper_store.list() if p.get("id") == paper_id), None)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"no paper {paper_id!r}")
    updated = state.paper_store.detach_session(paper_id, session_id)
    return PaperMetadata.model_validate(updated)


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
    session_id: str | None = Form(default=None),
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
    return _ingest(state, paper_id=pid, path=dest, title=paper_title or pid, session_id=session_id)


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
    # Written before the download starts, not just at _ingest's own first
    # write - a concurrent GET /papers/{id}/status poll (see below) needs
    # something to find from the moment the "Reading..." card appears,
    # not several seconds later once the PDF finishes downloading.
    state.paper_store.save(
        pid,
        title=body.title,
        authors=body.authors,
        abstract=body.abstract,
        pdf_url=body.pdf_url or url,
        status="downloading",
        session_id=body.session_id,
    )
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
        session_id=body.session_id,
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
