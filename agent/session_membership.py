"""The one shared definition of "which sessions does this record belong
to" - used by graph nodes/edges (agent/graph_manager.py), paper records
(service/storage.py), and the standalone cleanup script
(scripts/clear_session.py).

Previously reimplemented independently three times, each with its own
copy of the exact same legacy-fallback logic. The commit that changed
membership semantics from a single scalar to a list had to touch all
three copies by hand, and scripts/clear_session.py's own copy needed a
separate follow-up fix or it would have unconditionally destroyed data
another session still legitimately owned - a future editor updating some
but not all copies would silently reintroduce that same class of bug.
Zero dependencies on GraphManager/PaperStore/Firestore on purpose, so
every layer that needs this can import the same one function without a
layering concern.
"""

from __future__ import annotations


def session_ids(data: dict) -> list[str]:
    """A record's real session membership, read from the persisted
    "session_ids" list every writer now uses. Falls back to the old
    single "session_id" scalar field for records written before multi-
    session membership existed - this is what lets pre-existing legacy-
    tagged data keep working exactly as before with no migration script,
    and self-heal into the new list-based model for free the next time
    anything re-touches it."""
    ids = data.get("session_ids")
    if ids:
        return ids
    legacy = data.get("session_id")
    return [legacy] if legacy else []
