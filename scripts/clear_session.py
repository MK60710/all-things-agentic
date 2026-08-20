"""Delete every paper/node/edge/chunk (and any now-dangling clarification
question) tagged with a given session id.

Usage:
    GOOGLE_CLOUD_PROJECT=my-project python scripts/clear_session.py <session_id>
    python scripts/clear_session.py <session_id> --project my-project
    python scripts/clear_session.py <session_id> --dry-run

Replaces the ad hoc, hand-written cleanup done for every round of live
testing this project has gone through so far - now parameterized by
session_id instead of manually figuring out which nodes/papers a given
round of testing touched.

Known, accepted limitation: a node this session created can be reused by a
*different* session's edge via canonicalization (see
GraphManager.apply_extraction_result - a reused node is never re-tagged
with the reusing session's id, so it stays owned by whoever created it).
If that node gets deleted here, the other session's edge is left pointing
at a node that no longer exists. This is a dev/test cleanup tool, not a
referential-integrity system - acceptable until multiple people are
concurrently using the same shared graph for real, which isn't true yet.
"""

from __future__ import annotations

import argparse
import os
from typing import Any


def _ids_with_session(collection: Any, session_id: str) -> list[str]:
    return [doc.id for doc in collection.where("session_id", "==", session_id).stream()]


def clear_session(db: Any, session_id: str, *, dry_run: bool = False) -> dict[str, int]:
    papers = db.collection("papers")
    nodes = db.collection("nodes")
    edges = db.collection("edges")
    chunks = db.collection("chunks")
    clarifications = db.collection("clarifications")

    paper_ids = _ids_with_session(papers, session_id)
    node_ids = _ids_with_session(nodes, session_id)
    edge_ids = _ids_with_session(edges, session_id)

    # Chunks carry no session_id of their own (they're already keyed by
    # paper_id, and paper_id -> session_id is 1:1 via the papers
    # collection) - derive membership transitively instead.
    chunk_ids: list[str] = []
    for paper_id in paper_ids:
        chunk_ids.extend(
            doc.id for doc in chunks.where("paper_id", "==", paper_id).stream()
        )

    node_id_set = set(node_ids)
    clarification_ids = [
        doc.id
        for doc in clarifications.stream()
        if doc.to_dict().get("provisional_node_id") in node_id_set
        or doc.to_dict().get("candidate_node_id") in node_id_set
    ]

    counts = {
        "papers": len(paper_ids),
        "nodes": len(node_ids),
        "edges": len(edge_ids),
        "chunks": len(chunk_ids),
        "clarifications": len(clarification_ids),
    }
    if dry_run:
        return counts

    for collection, ids in (
        (papers, paper_ids),
        (nodes, node_ids),
        (edges, edge_ids),
        (chunks, chunk_ids),
        (clarifications, clarification_ids),
    ):
        for doc_id in ids:
            collection.document(doc_id).delete()

    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without deleting anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.project:
        raise SystemExit(
            "No project given - set GOOGLE_CLOUD_PROJECT or pass --project."
        )
    from google.cloud import firestore

    db = firestore.Client(project=args.project)
    counts = clear_session(db, args.session_id, dry_run=args.dry_run)
    verb = "Would delete" if args.dry_run else "Deleted"
    for kind, count in counts.items():
        print(f"{verb} {count} {kind}")


if __name__ == "__main__":
    main()
