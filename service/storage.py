"""Small Firestore-backed stores for browser uploads and paper metadata."""

from __future__ import annotations

import hashlib
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from agent.session_membership import session_ids as _paper_session_ids


class UploadTokenStore:
    def __init__(self, db_client: Any, *, ttl_seconds: int = 120) -> None:
        self._db = db_client
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def issue(self, *, max_bytes: int) -> tuple[str, datetime]:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._ttl_seconds)
        self._db.collection("upload_tokens").document(self._digest(token)).set(
            {
                "expires_at": expires_at.isoformat(),
                "max_bytes": max_bytes,
                "content_type": "application/pdf",
            }
        )
        return token, expires_at

    def consume(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None
        digest = self._digest(token)
        with self._lock:
            snapshot = self._db.collection("upload_tokens").document(digest).get()
            if not snapshot.exists:
                return None
            data = snapshot.to_dict()
            self._db.collection("upload_tokens").document(digest).delete()
        try:
            expires_at = datetime.fromisoformat(data["expires_at"])
        except (KeyError, TypeError, ValueError):
            return None
        if expires_at <= datetime.now(timezone.utc):
            return None
        return data


class PaperStore:
    def __init__(self, db_client: Any) -> None:
        self._collection = db_client.collection("papers")
        # save()'s and detach_session()'s session-membership merges are
        # both a read-then-write over the same document - two concurrent
        # calls (e.g. the same paper re-ingested into two sessions at
        # once) can each read the same "before" state and then each
        # write their own merged result, the second clobbering the
        # first. This is the exact bug class save()'s own docstring says
        # was already found and fixed once for the sequential case;
        # this lock closes the same gap under real concurrency. In-
        # process only, matching GraphManager's own RLock and this
        # service's documented single-instance deploy profile.
        self._lock = threading.Lock()

    def save(self, paper_id: str, **values: Any) -> dict[str, Any]:
        """A `session_id` kwarg is treated specially: it's session-
        accumulating, not overwriting - re-ingesting an already-known
        paper into a new session ADDS that session to the paper's
        membership rather than stealing it from whichever session
        originally added it. Confirmed live as a real bug: re-ingesting
        a paper elsewhere silently made it vanish from its original
        session's paper list with no warning. Every other field keeps
        the exact same partial-field-merge behavior as before."""
        with self._lock:
            if "session_id" in values:
                new_session_id = values.pop("session_id")
                existing = self.get(paper_id) or {}
                merged = set(_paper_session_ids(existing))
                if new_session_id:
                    merged.add(new_session_id)
                values["session_ids"] = sorted(merged)
            data = {
                "id": paper_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **values,
            }
            self._collection.document(paper_id).set(data, merge=True)
            return data

    def detach_session(self, paper_id: str, session_id: str) -> dict[str, Any]:
        """Removes just this one session's membership - the record (and
        its chunks/graph data) survives if another session still has it.
        The caller decides whether to fully delete once ownerless (see
        service/routers/sessions.py's delete_session cascade)."""
        with self._lock:
            existing = self.get(paper_id) or {}
            remaining = [s for s in _paper_session_ids(existing) if s != session_id]
            data = {
                **existing,
                "id": paper_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "session_ids": remaining,
            }
            data.pop("session_id", None)
            self._collection.document(paper_id).set(data, merge=True)
            return data

    def list(self) -> list[dict[str, Any]]:
        papers = [snapshot.to_dict() for snapshot in self._collection.stream()]
        return sorted(papers, key=lambda paper: str(paper.get("updated_at", "")), reverse=True)

    def get(self, paper_id: str) -> dict[str, Any] | None:
        snapshot = self._collection.document(paper_id).get()
        return snapshot.to_dict() if snapshot.exists else None

    def delete(self, paper_id: str) -> None:
        self._collection.document(paper_id).delete()


class SessionStore:
    def __init__(self, db_client: Any) -> None:
        self._collection = db_client.collection("sessions")

    def save(self, session_id: str, **values: Any) -> dict[str, Any]:
        data = {
            "id": session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        self._collection.document(session_id).set(data, merge=True)
        return data

    def list(self) -> list[dict[str, Any]]:
        sessions = [snapshot.to_dict() for snapshot in self._collection.stream()]
        return sorted(sessions, key=lambda session: str(session.get("updated_at", "")), reverse=True)

    def get(self, session_id: str) -> dict[str, Any] | None:
        """Targeted single-document read, same pattern as PaperStore.get -
        every route resolving "does this session_id belong to me" was
        previously doing `next(... for s in self.list() ...)`, a full scan
        of every session ever created by every account just to find one by
        id. Confirmed live as a real, measured contributor to slow page
        loads (every session-scoped route pays this on every request, and
        it gets worse as the collection grows) - use this instead of
        list() wherever only one session_id's data is actually needed."""
        snapshot = self._collection.document(session_id).get()
        return snapshot.to_dict() if snapshot.exists else None

    def delete(self, session_id: str) -> None:
        self._collection.document(session_id).delete()


class SessionMessagesStore:
    """Chat message history, kept in its own collection rather than on the
    session document SessionStore writes - list_sessions() streams every
    session doc in full for the session switcher, and putting messages
    there would mean that read pulls down every session's entire chat
    history just to show session names (SessionMetadata.model_validate
    would silently drop the field before it reaches the client, but the
    Firestore read cost is already paid by then). This collection is never
    enumerated in bulk, only fetched by exact session id."""

    def __init__(self, db_client: Any) -> None:
        self._collection = db_client.collection("session_messages")
        # Latest pre-compaction snapshot only, overwritten on each new
        # compaction - not a full history of every compaction event. Exists
        # so a long conversation's earlier citation-precision detail isn't
        # silently gone forever just because it got summarized out of the
        # live view; nothing reads this yet, it's a safety net.
        self._archive_collection = db_client.collection("session_messages_archive")

    def get(self, session_id: str) -> list[dict[str, Any]]:
        snapshot = self._collection.document(session_id).get()
        if not snapshot.exists:
            return []
        return snapshot.to_dict().get("messages", [])

    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        data = {"messages": messages, "updated_at": datetime.now(timezone.utc).isoformat()}
        self._collection.document(session_id).set(data, merge=True)

    def archive(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        data = {"messages": messages, "archived_at": datetime.now(timezone.utc).isoformat()}
        self._archive_collection.document(session_id).set(data, merge=True)

    def delete(self, session_id: str) -> None:
        self._collection.document(session_id).delete()
        self._archive_collection.document(session_id).delete()
