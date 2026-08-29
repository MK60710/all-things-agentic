from __future__ import annotations

from pathlib import Path

import pytest

from service.document_storage import CloudStorageDocumentStore, LocalDocumentStore


def test_local_document_store_persists_reads_and_deletes(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-test")
    store = LocalDocumentStore(tmp_path)

    key = store.persist("paper", source)
    assert key == "paper.pdf"
    assert store.read(key) == b"%PDF-test"

    store.delete(key)
    with pytest.raises(FileNotFoundError):
        store.read(key)


def test_local_document_store_rejects_paths_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.pdf"
    outside.write_bytes(b"%PDF-test")
    try:
        with pytest.raises(ValueError, match="outside"):
            LocalDocumentStore(tmp_path).persist("paper", outside)
    finally:
        outside.unlink(missing_ok=True)


def test_cloud_object_keys_do_not_expose_user_or_paper_ids() -> None:
    key = CloudStorageDocumentStore._object_key("private-user-id-secret-paper")

    assert key.startswith("papers/")
    assert key.endswith(".pdf")
    assert "private-user-id" not in key
