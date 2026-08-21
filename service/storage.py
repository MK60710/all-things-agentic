"""Small Firestore-backed stores for browser uploads and paper metadata."""

from __future__ import annotations

import hashlib
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any


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

    def save(self, paper_id: str, **values: Any) -> dict[str, Any]:
        data = {
            "id": paper_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        self._collection.document(paper_id).set(data, merge=True)
        return data

    def list(self) -> list[dict[str, Any]]:
        papers = [snapshot.to_dict() for snapshot in self._collection.stream()]
        return sorted(papers, key=lambda paper: str(paper.get("updated_at", "")), reverse=True)

    def get(self, paper_id: str) -> dict[str, Any] | None:
        snapshot = self._collection.document(paper_id).get()
        return snapshot.to_dict() if snapshot.exists else None


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
