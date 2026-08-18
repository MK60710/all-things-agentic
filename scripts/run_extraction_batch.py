"""Run extraction over a directory of PDFs and ingest the results.

This is the concrete entrypoint the Part 5 plan described: extraction
never blocks on a person mid-batch, but the run ends by reporting how
many clarification questions are open and, unless --no-review is passed,
drops straight into a terminal prompt loop to answer them right there -
the interim way a person finds out about extraction-side ambiguity until
Part 8 (the frontend) exists to show it instead.

Usage:
    GOOGLE_CLOUD_PROJECT=my-project python scripts/run_extraction_batch.py
    python scripts/run_extraction_batch.py --local --corpus-dir corpus --max-papers 3

--local runs entirely off an in-process fake Firestore client and the
chunks-only structured extractor (no Vertex AI calls, no GCP credentials
needed) - useful for exercising the wiring, not for a real extraction run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from agent.clarification_orchestrator import ClarificationOrchestrator
from agent.extraction_agent import ChunkOnlyStructuredExtractor, ExtractionAgent
from agent.document_ingestion import PdfTextExtractor
from agent.gemini_extractor import GeminiStructuredExtractor
from agent.graph_manager import GraphManager
from agent.research_store import ResearchStore
from agent.retrieval import ChunkIndex, LocalHashingEmbedder


class _InProcessFirestore:
    """Minimal Firestore-shaped in-memory client for --local runs.

    Same shape as tests/conftest.py's FakeFirestoreClient (set/stream on a
    per-collection dict) - duplicated rather than imported since tests/
    isn't part of the installed package and a script shouldn't depend on
    the test tree.
    """

    class _DocRef:
        def __init__(self, store: dict, doc_id: str):
            self._store, self._doc_id = store, doc_id

        def set(self, data: dict, merge: bool = True) -> None:
            self._store[self._doc_id] = data

        def delete(self) -> None:
            self._store.pop(self._doc_id, None)

    class _Snapshot:
        def __init__(self, doc_id: str, data: dict):
            self.id, self._data = doc_id, data

        def to_dict(self) -> dict:
            return self._data

    class _Collection:
        def __init__(self, store: dict, filters: list[tuple[str, object]] | None = None):
            self._store = store
            self._filters = filters or []

        def document(self, doc_id: str) -> "_InProcessFirestore._DocRef":
            return _InProcessFirestore._DocRef(self._store, doc_id)

        def where(self, field: str, operator: str, value: object) -> "_InProcessFirestore._Collection":
            if operator != "==":
                raise NotImplementedError("fake client only supports equality filters")
            return _InProcessFirestore._Collection(
                self._store, [*self._filters, (field, value)]
            )

        def stream(self):
            for doc_id, data in self._store.items():
                if any(data.get(field) != value for field, value in self._filters):
                    continue
                yield _InProcessFirestore._Snapshot(doc_id, data)

    def __init__(self) -> None:
        self._collections: dict[str, dict] = {}

    def collection(self, name: str) -> "_InProcessFirestore._Collection":
        return _InProcessFirestore._Collection(self._collections.setdefault(name, {}))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument(
        "--local",
        action="store_true",
        help="No Vertex AI or real Firestore - chunks-only extraction against an in-memory graph, for exercising the wiring.",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip the terminal review loop at the end - just print the pending count.",
    )
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument(
        "--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.local and not args.project:
        raise SystemExit(
            "No project given - set GOOGLE_CLOUD_PROJECT or pass --project, "
            "or use --local to run without live GCP."
        )

    pdf_paths = sorted(Path(args.corpus_dir).glob("*.pdf"))
    if args.max_papers is not None:
        pdf_paths = pdf_paths[: args.max_papers]
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {args.corpus_dir!r}")

    document_extractor = PdfTextExtractor()
    structured_extractor: Any = (
        ChunkOnlyStructuredExtractor()
        if args.local
        else GeminiStructuredExtractor(project=args.project, location=args.location)
    )
    extraction_agent = ExtractionAgent(
        document_extractor=document_extractor,
        structured_extractor=structured_extractor,
    )

    db_client = _InProcessFirestore() if args.local else None
    graph = GraphManager(
        project_id=args.project or "local", db_client=db_client
    )
    chunks = ChunkIndex(db_client=db_client)
    store = ResearchStore(chunks, graph)
    orchestrator = ClarificationOrchestrator(graph_manager=graph)

    # Entity-level embeddings for canonicalization - reuses the same local
    # hashing embedder ChunkIndex already uses for chunk retrieval, applied
    # one level up. Without this, canonicalize() never sees an embedding
    # and needs_clarification can only ever fire on an exact string match,
    # so the Part 5 extraction-side hook would never actually trigger on
    # a real run.
    embedder = LocalHashingEmbedder()

    def entity_embedding_fn(entity) -> list[float]:
        return embedder(f"{entity.name}: {entity.description}")

    papers = [(path.stem, str(path)) for path in pdf_paths]
    print(f"Extracting {len(papers)} paper(s)...")
    batch = extraction_agent.extract_batch(papers)

    ingested = 0
    for outcome in batch.outcomes:
        if outcome.result is None:
            print(f"  [FAILED]  {outcome.paper_id}: {outcome.issue.message if outcome.issue else 'unknown error'}")
            continue
        status = "ok" if outcome.ok else "partial"
        store.ingest(
            outcome.result,
            paper_name=outcome.paper_id,
            entity_embedding_fn=entity_embedding_fn,
            clarification=orchestrator,
        )
        ingested += 1
        print(f"  [{status.upper():7}] {outcome.paper_id}")

    print(
        f"\nIngested {ingested}/{len(papers)} paper(s). "
        f"{len(batch.failures)} failure(s)."
    )

    pending = orchestrator.pending()
    print(f"{len(pending)} clarification question(s) need your input.")
    if pending and not args.no_review:
        orchestrator.run_terminal_review_loop()


if __name__ == "__main__":
    main()
