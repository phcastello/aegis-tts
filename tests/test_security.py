from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.config import AppSettings
from app.security import require_api_key, verify_api_key


def test_api_key_is_optional_when_not_configured() -> None:
    assert verify_api_key(None, None) is True
    assert verify_api_key("any-value", "") is True


def test_api_key_comparison_accepts_only_an_exact_match() -> None:
    assert verify_api_key("correct-key", "correct-key") is True
    assert verify_api_key("wrong-key", "correct-key") is False
    assert verify_api_key(None, "correct-key") is False


def test_fastapi_dependency_returns_401_without_leaking_the_secret() -> None:
    settings = AppSettings(api_key="do-not-log-this")

    with pytest.raises(HTTPException) as captured:
        require_api_key(settings, x_api_key="wrong")

    assert captured.value.status_code == 401
    assert "do-not-log-this" not in str(captured.value.detail)
    require_api_key(settings, x_api_key="do-not-log-this")

