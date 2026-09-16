"""FinBERT 載入與 Windows 憑證設定測試。"""

from __future__ import annotations

import httpx

from ai_quant_trading.sentiment import finbert as finbert_module


def test_configure_huggingface_truststore_on_windows(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(finbert_module.sys, "platform", "win32")
    monkeypatch.setattr(
        "huggingface_hub.set_client_factory",
        lambda factory: captured.update(factory=factory),
    )

    finbert_module._configure_huggingface_truststore()

    factory = captured["factory"]
    client = factory()
    try:
        assert isinstance(client, httpx.Client)
        assert client.follow_redirects is True
    finally:
        client.close()


def test_configure_huggingface_truststore_is_noop_outside_windows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(finbert_module.sys, "platform", "linux")
    finbert_module._configure_huggingface_truststore()
