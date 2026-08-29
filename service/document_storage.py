"""Private durable storage for original paper PDFs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from google.api_core.exceptions import NotFound
from google.cloud import storage


class DocumentStore(Protocol):
    def persist(self, paper_id: str, local_path: Path) -> str: ...

    def read(self, object_key: str) -> bytes: ...

    def delete(self, object_key: str) -> None: ...


class LocalDocumentStore:
    """Development/test implementation backed by the configured upload root."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    def persist(self, paper_id: str, local_path: Path) -> str:
        path = local_path.resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("paper path is outside the document root")
        return path.name

    def read(self, object_key: str) -> bytes:
        path = (self._root / object_key).resolve()
        if not path.is_relative_to(self._root) or not path.is_file():
            raise FileNotFoundError(object_key)
        return path.read_bytes()

    def delete(self, object_key: str) -> None:
        path = (self._root / object_key).resolve()
        if path.is_relative_to(self._root):
            path.unlink(missing_ok=True)


class CloudStorageDocumentStore:
    """Google Cloud Storage implementation; objects remain private."""

    def __init__(self, bucket_name: str, *, project: str) -> None:
        self._bucket = storage.Client(project=project).bucket(bucket_name)

    @staticmethod
    def _object_key(paper_id: str) -> str:
        digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()
        return f"papers/{digest}.pdf"

    def persist(self, paper_id: str, local_path: Path) -> str:
        object_key = self._object_key(paper_id)
        self._bucket.blob(object_key).upload_from_filename(
            str(local_path), content_type="application/pdf", timeout=60
        )
        return object_key

    def read(self, object_key: str) -> bytes:
        try:
            return self._bucket.blob(object_key).download_as_bytes(timeout=60)
        except NotFound as exc:
            raise FileNotFoundError(object_key) from exc

    def delete(self, object_key: str) -> None:
        try:
            self._bucket.blob(object_key).delete(timeout=30)
        except NotFound:
            return
