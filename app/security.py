"""Small authentication helpers.  Reference material is never authorized out."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Header, HTTPException, status

from .config import AppSettings


def verify_api_key(provided: str | None, expected: str | None) -> bool:
    """Compare configured and presented keys without a timing side channel."""

    if not expected:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def require_api_key(
    settings: AppSettings,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """FastAPI dependency for synthesis (and any future admin endpoint)."""

    if not verify_api_key(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key ausente ou inválida.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
