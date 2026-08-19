from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from service.deps import get_state, require_api_key
from service.schemas import PaperIngestResponse
from service.state import AppState

router = APIRouter(prefix="/papers", tags=["papers"], dependencies=[Depends(require_api_key)])

_UNSAFE_PAPER_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_paper_id(raw: str) -> str:
    """A client-supplied paper_id becomes a filename below - allowing path
    separators or ".." through would let an upload write outside
    upload_root (e.g. paper_id="../../../../tmp/evil"), and
    PdfTextExtractor's allowed_root check runs later, after that write
    already happened. Strip to a safe charset first, defense in depth
    with the is_relative_to check below."""
    cleaned = _UNSAFE_PAPER_ID_CHARS.sub("_", raw).strip("._")
    return cleaned or str(uuid.uuid4())


@router.post("", response_model=PaperIngestResponse)
def upload_paper(
    file: UploadFile = File(...),
    paper_id: str | None = Form(default=None),
    state: AppState = Depends(get_state),
) -> PaperIngestResponse:
    raw_pid = paper_id or (Path(file.filename).stem if file.filename else str(uuid.uuid4()))
    pid = _sanitize_paper_id(raw_pid)

    upload_root = Path(state.upload_root).resolve()
    dest = (upload_root / f"{pid}.pdf").resolve()
    if not dest.is_relative_to(upload_root):
        raise HTTPException(status_code=400, detail="invalid paper_id")
    dest.write_bytes(file.file.read())

    # fail_closed=False: still return whatever chunks-only ingestion is
    # possible when structured extraction fails, rather than a bare error,
    # matching scripts/run_extraction_batch.py's real-run behavior.
    outcome = state.extraction_agent.extract_one(pid, str(dest), fail_closed=False)
    if outcome.result is None:
        raise HTTPException(
            status_code=422,
            detail=outcome.issue.message if outcome.issue else "extraction failed",
        )

    # PendingQuestion carries no paper_id, so pending() can't be scoped by
    # paper directly - counting before/after this ingest call gives the
    # count this upload actually added, not ClarificationOrchestrator's
    # entire cross-paper backlog (which is what len(pending()) alone would
    # report, misleadingly, on any upload after the first).
    pending_before = len(state.clarification.pending())
    report = state.research_store.ingest(
        outcome.result,
        paper_name=pid,
        entity_embedding_fn=state.entity_embedding_fn,
        clarification=state.clarification,
    )
    pending_added = len(state.clarification.pending()) - pending_before

    return PaperIngestResponse(
        paper_id=pid,
        extraction_ok=outcome.ok,
        issue_message=outcome.issue.message if outcome.issue else None,
        chunk_ids=report.chunk_ids,
        entities_added=len(outcome.result.entities),
        relations_added=len(outcome.result.relations),
        pending_clarification_count=pending_added,
    )
