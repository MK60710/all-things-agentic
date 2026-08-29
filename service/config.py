"""Deployment-time configuration validation.

Local development stays permissive. Production fails during startup when a
security-sensitive setting is absent or still points at a development host.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse


def validate_production_environment() -> None:
    if os.environ.get("ATLAS_ENV", "development").lower() != "production":
        return

    secret = os.environ.get("API_SHARED_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError("API_SHARED_SECRET must contain at least 32 characters in production")

    origins = [
        value.strip()
        for value in os.environ.get("CORS_ORIGINS", "").split(",")
        if value.strip()
    ]
    if not origins:
        raise RuntimeError("CORS_ORIGINS must contain the production frontend origin")
    for origin in origins:
        parsed = urlparse(origin)
        if (
            origin == "*"
            or parsed.scheme != "https"
            or parsed.hostname in {"localhost", "127.0.0.1"}
        ):
            raise RuntimeError(
                "CORS_ORIGINS may contain only explicit HTTPS origins in production"
            )
