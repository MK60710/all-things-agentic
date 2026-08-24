"""Delete a session's papers/nodes/edges/chunks (and any now-dangling
clarification question) - but only what actually becomes ownerless.

Usage:
    GOOGLE_CLOUD_PROJECT=my-project python scripts/clear_session.py <session_id>
    python scripts/clear_session.py <session_id> --project my-project
    python scripts/clear_session.py <session_id> --dry-run

Replaces the ad hoc, hand-written cleanup done for every round of live
testing this project has gone through so far - now parameterized by
session_id instead of manually figuring out which nodes/papers a given
round of testing touched.

Papers/nodes/edges are session-accumulating (see GraphManager.add_node/
PaperStore.save): the same paper re-ingested into a second session
belongs to both, not just whichever session touched it first or last.
This script mirrors that - a doc still genuinely shared with another
session survives, with just this session's membership stripped from it;
only a doc that becomes ownerless is actually deleted. Older docs written
before multi-session membership existed only ever had a single
"session_id" field - agent.session_membership.session_ids (the one
shared implementation of this, also used by GraphManager and PaperStore)
falls back to that, so this script still finds and correctly cleans up
that legacy data too.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from agent.session_membership import session_ids as _session_ids


def _member_of(data: dict, session_id: str) -> bool:
    return session_id in _session_ids(data)


def _remaining_sessions(data: dict, session_id: str) -> list[str]:
    return [s for s in _session_ids(data) if s != session_id]


def _strip_or_delete(collection: Any, session_id: str) -> tuple[list[str], list[str]]:
    """Returns (deleted_ids, survived_ids) for every doc in this
    collection tagged with session_id - deleted if this was its only
    session, updated in place (membership stripped) if another session
    still legitimately has it."""
    deleted: list[str] = []
    survived: list[str] = []
    for doc in collection.stream():
        data = doc.to_dict()
        if not _member_of(data, session_id):
            continue
        remaining = _remaining_sessions(data, session_id)
        if remaining:
            survived.append(doc.id)
            collection.document(doc.id).set({**data, "session_ids": remaining}, merge=True)
        else:
            deleted.append(doc.id)
    return deleted, survived


def clear_session(db: Any, session_id: str, *, dry_run: bool = False) -> dict[str, int]:
    papers = db.collection("papers")
    nodes = db.collection("nodes")
    edges = db.collection("edges")
    chunks = db.collection("chunks")
    clarifications = db.collection("clarifications")

    if dry_run:
        deleted_paper_ids, _ = _strip_or_delete_dry_run(papers, session_id)
        deleted_node_ids, _ = _strip_or_delete_dry_run(nodes, session_id)
        deleted_edge_ids, _ = _strip_or_delete_dry_run(edges, session_id)
    else:
        deleted_paper_ids, _ = _strip_or_delete(papers, session_id)
        deleted_node_ids, _ = _strip_or_delete(nodes, session_id)
        deleted_edge_ids, _ = _strip_or_delete(edges, session_id)

    # Chunks carry no session tag of their own (they're keyed by
    # paper_id) - only follow chunks for papers that were actually fully
    # deleted, not ones that merely had this session's membership
    # stripped and survive with another session still owning them.
    chunk_ids: list[str] = []
    for paper_id in deleted_paper_ids:
        chunk_ids.extend(
            doc.id for doc in chunks.where("paper_id", "==", paper_id).stream()
        )

    node_id_set = set(deleted_node_ids)
    clarification_ids = [
        doc.id
        for doc in clarifications.stream()
        if doc.to_dict().get("provisional_node_id") in node_id_set
        or doc.to_dict().get("candidate_node_id") in node_id_set
    ]

    counts = {
        "papers": len(deleted_paper_ids),
        "nodes": len(deleted_node_ids),
        "edges": len(deleted_edge_ids),
        "chunks": len(chunk_ids),
        "clarifications": len(clarification_ids),
    }
    if dry_run:
        return counts

    for doc_id in chunk_ids:
        chunks.document(doc_id).delete()
    for doc_id in clarification_ids:
        clarifications.document(doc_id).delete()
    for collection, ids in ((papers, deleted_paper_ids), (nodes, deleted_node_ids), (edges, deleted_edge_ids)):
        for doc_id in ids:
            collection.document(doc_id).delete()

    return counts


def _strip_or_delete_dry_run(collection: Any, session_id: str) -> tuple[list[str], list[str]]:
    """Same membership classification as _strip_or_delete, without
    writing anything - dry-run must report accurate counts without
    mutating a single doc."""
    deleted: list[str] = []
    survived: list[str] = []
    for doc in collection.stream():
        data = doc.to_dict()
        if not _member_of(data, session_id):
            continue
        if _remaining_sessions(data, session_id):
            survived.append(doc.id)
        else:
            deleted.append(doc.id)
    return deleted, survived


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
