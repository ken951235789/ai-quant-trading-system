"""啟動資料更新服務測試。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ai_quant_trading.startup_refresh import (
    StartupRefreshConfig,
    _preserve_extra_feature_columns,
    _refresh_lock,
    discover_market_targets,
    load_startup_refresh_config,
    refresh_live_news_sentiment,
    save_startup_refresh_config,
)


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            "exchange": ["binance", "binance"],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "interval": ["1d", "1d"],
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [10.0, 20.0],
        }
    )


def test_discover_market_targets_reads_canonical_files(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "crypto" / "binance" / "ohlcv" / "btc_latest.csv"
    path.parent.mkdir(parents=True)
    _market_frame().to_csv(path, index=False)

    targets = discover_market_targets(tmp_path / "raw")

    assert len(targets) == 1
    assert targets[0].key == ("binance", "BTC/USDT", "1d")
    assert targets[0].latest_timestamp == pd.Timestamp("2026-01-02T00:00:00Z")


def test_refresh_config_round_trip(tmp_path: Path) -> None:
    config = StartupRefreshConfig(yahoo_news_per_symbol=5, score_finbert=False)

    save_startup_refresh_config(tmp_path, config)
    restored = load_startup_refresh_config(tmp_path)

    assert restored == config


def test_preserves_external_features_for_matching_timestamps() -> None:
    previous = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "close": [1.0],
            "transformer_available": [1.0],
            "transformer_return_1": [0.2],
        }
    )
    refreshed = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            "close": [1.1, 1.2],
        }
    )

    result = _preserve_extra_feature_columns(previous, refreshed)

    assert result["transformer_available"].tolist() == [1.0, 0.0]
    assert result.loc[0, "transformer_return_1"] == 0.2
    assert pd.isna(result.loc[1, "transformer_return_1"])


def test_refresh_lock_recovers_when_recorded_process_is_gone(tmp_path: Path) -> None:
    lock_path = tmp_path / "data" / "config" / "startup_refresh.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("99999999", encoding="utf-8")

    with _refresh_lock(tmp_path) as acquired:
        assert acquired is True

    assert not lock_path.exists()


def test_live_news_refresh_uses_cooldown_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "ai_quant_trading.startup_refresh._load_pipeline_config",
        lambda _root: SimpleNamespace(finbert_enabled=False),
    )

    first = refresh_live_news_sentiment(tmp_path, minimum_interval_seconds=900)
    second = refresh_live_news_sentiment(tmp_path, minimum_interval_seconds=900)

    assert first["status"] == "disabled"
    assert first["cached"] is False
    assert second["cached"] is True
