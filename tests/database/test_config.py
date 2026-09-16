"""資料庫設定測試。"""

from __future__ import annotations

import pytest

from ai_quant_trading.database.config import DatabaseSettings


def test_file_backend_is_safe_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_QUANT_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("AI_QUANT_DATABASE_URL", raising=False)

    settings = DatabaseSettings.from_environment(env_path="missing.env")

    assert settings.backend == "file"
    assert not settings.enabled


def test_postgres_requires_database_url() -> None:
    with pytest.raises(ValueError, match="AI_QUANT_DATABASE_URL"):
        DatabaseSettings(backend="postgres")


def test_rejects_non_postgres_trading_database() -> None:
    with pytest.raises(ValueError, match="只接受 PostgreSQL"):
        DatabaseSettings(backend="postgres", database_url="sqlite:///trading.db")


def test_redacts_password() -> None:
    settings = DatabaseSettings(
        backend="postgres",
        database_url="postgresql+psycopg://robot:secret@localhost:5432/ai_quant",
    )

    assert settings.redacted_url == (
        "postgresql+psycopg://robot:***@localhost:5432/ai_quant"
    )
    assert "secret" not in settings.redacted_url
