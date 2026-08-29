from __future__ import annotations

import pytest

from service.config import validate_production_environment


def test_development_environment_remains_permissive(monkeypatch) -> None:
    monkeypatch.delenv("ATLAS_ENV", raising=False)
    monkeypatch.delenv("API_SHARED_SECRET", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    validate_production_environment()


def test_production_requires_a_strong_shared_secret(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_ENV", "production")
    monkeypatch.setenv("API_SHARED_SECRET", "short")
    monkeypatch.setenv("CORS_ORIGINS", "https://atlas.example")

    with pytest.raises(RuntimeError, match="at least 32"):
        validate_production_environment()


@pytest.mark.parametrize(
    "origins",
    ["", "*", "http://atlas.example", "http://localhost:3000"],
)
def test_production_rejects_unsafe_cors_origins(monkeypatch, origins: str) -> None:
    monkeypatch.setenv("ATLAS_ENV", "production")
    monkeypatch.setenv("API_SHARED_SECRET", "s" * 32)
    monkeypatch.setenv("CORS_ORIGINS", origins)

    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        validate_production_environment()


def test_production_accepts_explicit_https_origins(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_ENV", "production")
    monkeypatch.setenv("API_SHARED_SECRET", "s" * 32)
    monkeypatch.setenv(
        "CORS_ORIGINS", "https://atlas.example,https://www.atlas.example"
    )
    monkeypatch.setenv("PAPER_STORAGE_BUCKET", "atlas-private-papers")

    validate_production_environment()


def test_production_requires_durable_paper_storage(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_ENV", "production")
    monkeypatch.setenv("API_SHARED_SECRET", "s" * 32)
    monkeypatch.setenv("CORS_ORIGINS", "https://atlas.example")
    monkeypatch.delenv("PAPER_STORAGE_BUCKET", raising=False)

    with pytest.raises(RuntimeError, match="PAPER_STORAGE_BUCKET"):
        validate_production_environment()
