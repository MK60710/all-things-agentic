from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from service.deps import get_state, require_api_key
from service.schemas import PaperIngestResponse
from service.state import AppState

router = APIRouter(prefix="/papers", tags=["papers"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=PaperIngestResponse)
def upload_paper(
    file: UploadFile = File(...),
    paper_id: str | None = Form(default=None),
    state: AppState = Depends(get_state),
) -> PaperIngestResponse:
    pid = paper_id or (Path(file.filename).stem if file.filename else str(uuid.uuid4()))
    dest = Path(state.upload_root) / f"{pid}.pdf"
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

    report = state.research_store.ingest(
        outcome.result,
        paper_name=pid,
        entity_embedding_fn=state.entity_embedding_fn,
        clarification=state.clarification,
    )
    return PaperIngestResponse(
        paper_id=pid,
        extraction_ok=outcome.ok,
        issue_message=outcome.issue.message if outcome.issue else None,
        chunk_ids=report.chunk_ids,
        entities_added=len(outcome.result.entities),
        relations_added=len(outcome.result.relations),
        pending_clarification_count=len(state.clarification.pending()),
    )
