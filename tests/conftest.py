"""Fake Firestore client so unit tests don't need live GCP credentials."""

from __future__ import annotations

import pytest


class _FakeDocRef:
    def __init__(self, store: dict, doc_id: str):
        self._store = store
        self._doc_id = doc_id

    def set(self, data: dict, merge: bool = True) -> None:
        self._store[self._doc_id] = data

    def delete(self) -> None:
        self._store.pop(self._doc_id, None)


class _FakeSnapshot:
    def __init__(self, doc_id: str, data: dict):
        self.id = doc_id
        self._data = data

    def to_dict(self) -> dict:
        return self._data


class _FakeCollection:
    def __init__(self, store: dict):
        self._store = store

    def document(self, doc_id: str) -> _FakeDocRef:
        return _FakeDocRef(self._store, doc_id)

    def stream(self):
        for doc_id, data in self._store.items():
            yield _FakeSnapshot(doc_id, data)


class FakeFirestoreClient:
    def __init__(self):
        self._collections: dict[str, dict] = {"nodes": {}, "edges": {}}

    def collection(self, name: str) -> _FakeCollection:
        return _FakeCollection(self._collections.setdefault(name, {}))


@pytest.fixture
def fake_db() -> FakeFirestoreClient:
    return FakeFirestoreClient()
