from pathlib import Path

import pytest

from ai_quant_trading.live_trading.binance_client import BinancePrivateClient
from ai_quant_trading.live_trading.credentials import BinanceCredentials
from ai_quant_trading.live_trading.notification import (
    _validated_webhook_url,
    redact_sensitive_text,
)


def test_private_client_rejects_non_binance_host_without_injected_test_session() -> None:
    with pytest.raises(ValueError, match="官方 HTTPS"):
        BinancePrivateClient(
            BinanceCredentials("key", "secret"),
            "https://attacker.example",
        )


def test_private_client_disables_environment_proxy() -> None:
    client = BinancePrivateClient(
        BinanceCredentials("key", "secret"),
        "https://api.binance.com",
    )
    assert client.session is not None
    assert client.session.trust_env is False


def test_webhook_allowlist_and_log_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_QUANT_NOTIFICATION_ALLOWED_HOSTS", raising=False)
    with pytest.raises(ValueError, match="允許"):
        _validated_webhook_url("https://attacker.example/collect")
    assert _validated_webhook_url("https://hooks.slack.com/services/test")
    text = redact_sensitive_text(
        "api_secret=abc123 postgresql+psycopg://user:password@127.0.0.1/db"
    )
    assert "abc123" not in text
    assert "password" not in text


def test_env_file_is_not_tracked() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / ".gitignore").read_text(encoding="utf-8").splitlines().count(".env") >= 1
