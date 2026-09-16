"""BTC 5m 即時模型趨勢防護測試。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pandas as pd
import numpy as np

from ai_quant_trading.automation import automation_paths
from ai_quant_trading.data_collection.binance_stream import BinanceKlineUpdate
from ai_quant_trading.paper_trading.realtime import (
    RealtimeMarketBuffer,
    RealtimePaperTradingSession,
    RealtimeStatusStore,
    TransformerTrendGateConfig,
    build_transformer_target_adjuster,
    classify_transformer_trend,
    latest_completed_bar_open,
)


class _FakeSpotClient:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls = 0
        self.fail = False

    def fetch_ohlcv(self, **_kwargs: object) -> pd.DataFrame:
        self.calls += 1
        if self.fail:
            raise ConnectionError("temporary outage")
        return self.frame.copy()

    def fetch_market_context(self, *_args: object, **_kwargs: object) -> pd.DataFrame:
        if self.fail:
            raise ConnectionError("temporary context outage")
        return pd.DataFrame(
            {
                "timestamp": [self.frame.iloc[0]["timestamp"]],
                "funding_rate": [0.0001],
                "open_interest": [1_000.0],
                "open_interest_value": [60_000_000.0],
                "open_interest_change": [0.01],
                "spread_bps": [0.2],
            }
        )


def _market_frame(periods: int = 270) -> pd.DataFrame:
    prices = pd.Series(range(60_000, 60_000 + periods), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01T00:00:00Z",
                periods=periods,
                freq="1min",
            ),
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "interval": "1m",
            "open": prices,
            "high": prices + 2,
            "low": prices - 1,
            "close": prices + 1,
            "volume": 10.0,
        }
    )


def _transformer_row(**overrides: float) -> pd.Series:
    values = {
        "transformer_available": 1.0,
        "transformer_bull_probability": 0.60,
        "transformer_bear_probability": 0.20,
        "transformer_uncertainty": 0.40,
        "transformer_return_1": 0.001,
        "transformer_return_5": 0.003,
        "transformer_return_20": 0.008,
    }
    values.update(overrides)
    return pd.Series(values)


def test_classify_transformer_bullish_trend() -> None:
    trend, details = classify_transformer_trend(
        _transformer_row(),
        TransformerTrendGateConfig(),
    )

    assert trend == "偏多"
    assert details["transformer_bull_probability"] == 0.60


def test_transformer_blocks_unconfirmed_short_entry() -> None:
    adjust = build_transformer_target_adjuster(
        TransformerTrendGateConfig(),
        lambda: {"spread_bps": 0.5},
        max_spread_bps=5.0,
    )

    approved, details = adjust(-0.25, 0.0, _transformer_row())

    assert approved == 0.0
    assert "未獲 Transformer 確認" in str(details["decision_guard_reason"])


def test_wide_spread_does_not_block_risk_reduction() -> None:
    adjust = build_transformer_target_adjuster(
        TransformerTrendGateConfig(),
        lambda: {"spread_bps": 20.0},
        max_spread_bps=5.0,
    )

    approved, details = adjust(0.10, 0.25, _transformer_row())

    assert approved == 0.10
    assert "不增加風險" in str(details["decision_guard_reason"])


def test_bearish_transformer_forces_long_toward_flat() -> None:
    adjust = build_transformer_target_adjuster(
        TransformerTrendGateConfig(),
        lambda: {"spread_bps": 0.5},
        max_spread_bps=5.0,
    )
    bearish = _transformer_row(
        transformer_bull_probability=0.15,
        transformer_bear_probability=0.65,
        transformer_return_5=-0.004,
    )

    approved, details = adjust(0.20, 0.25, bearish)

    assert approved == 0.0
    assert details["transformer_trend"] == "偏空"


def test_realtime_market_buffer_uses_memory_and_caps_saved_rows(tmp_path: Path) -> None:
    client = _FakeSpotClient(_market_frame())
    buffer = RealtimeMarketBuffer(
        "BTC/USDT",
        ("1m",),
        max_bars=260,
        raw_dir=tmp_path,
        client=client,
    )

    buffer.sync_from_binance()
    latest_open = int(pd.Timestamp("2025-01-01T04:30:00Z").timestamp() * 1000)
    buffer.append_closed_kline(
        BinanceKlineUpdate(
            symbol="BTC/USDT",
            interval="1m",
            open_time_ms=latest_open,
            close_time_ms=latest_open + 59_999,
            open=61_000.0,
            high=61_002.0,
            low=60_999.0,
            close=61_001.0,
            volume=12.0,
            quote_asset_volume=100.0,
            number_of_trades=10,
            taker_buy_base_volume=6.0,
            taker_buy_quote_volume=50.0,
            closed=True,
            event_time_ms=latest_open + 60_000,
        )
    )
    # 決策前的 REST 補漏也會把最新 Funding/OI 對齊到剛收盤 K 線。
    buffer.sync_from_binance()

    snapshot = buffer.snapshot()["1m"]
    paths = buffer.persist_rolling_snapshots()
    saved = pd.read_csv(paths[0])

    assert client.calls == 2
    assert len(snapshot) == 260
    assert len(saved) == 260
    assert float(snapshot.iloc[-1]["close"]) == 61_001.0
    assert float(snapshot.iloc[-1]["funding_rate"]) == 0.0001


def test_realtime_market_buffer_keeps_memory_during_rest_outage(tmp_path: Path) -> None:
    client = _FakeSpotClient(_market_frame())
    buffer = RealtimeMarketBuffer(
        "BTC/USDT",
        ("1m",),
        max_bars=260,
        raw_dir=tmp_path,
        client=client,
    )
    buffer.sync_from_binance()
    client.fail = True

    buffer.sync_from_binance()

    assert len(buffer.snapshot()["1m"]) == 260


def test_initial_decision_warms_rest_before_starting_websocket(tmp_path: Path) -> None:
    events: list[str] = []

    class _Buffer:
        def sync_from_binance(self) -> None:
            events.append("rest_warmup")

    class _Stream:
        def start(self) -> None:
            events.append("websocket_start")

        def snapshot(self) -> dict[str, object]:
            return {"state": "connected", "connected": True, "spread_bps": 0.1}

    class _Status:
        def update(self, **_values: object) -> None:
            return None

    session = object.__new__(RealtimePaperTradingSession)
    session.config = SimpleNamespace(stream_connect_timeout_seconds=1.0)
    session.market_buffer = _Buffer()
    session.stream = _Stream()
    session.status = _Status()
    session.run_decision = lambda _trigger, *, sync_latest: events.append(
        f"decision_sync={sync_latest}"
    )

    session._run_initial_decision(automation_paths(tmp_path / "automation"))

    assert events == ["rest_warmup", "websocket_start", "decision_sync=False"]


def test_latest_completed_bar_waits_for_settlement_buffer() -> None:
    before_settlement = latest_completed_bar_open(
        pd.Timestamp("2026-08-20T18:15:01Z"),
        "15m",
        settle_seconds=2,
    )
    after_settlement = latest_completed_bar_open(
        pd.Timestamp("2026-08-20T18:15:02Z"),
        "15m",
        settle_seconds=2,
    )

    assert before_settlement == pd.Timestamp("2026-08-20T17:45:00Z")
    assert after_settlement == pd.Timestamp("2026-08-20T18:00:00Z")


def test_realtime_status_serializes_numpy_and_pandas_values(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    status = RealtimeStatusStore(path)

    status.update(
        drifted=np.bool_(False),
        target=np.float64(0.25),
        timestamp=pd.Timestamp("2026-08-20T00:00:00Z"),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["drifted"] is False
    assert payload["target"] == 0.25
    assert payload["timestamp"] == "2026-08-20T00:00:00+00:00"


def test_realtime_status_recovers_from_windows_access_denied(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "status.json"
    status = RealtimeStatusStore(path)
    real_replace = os.replace
    attempts = 0

    def flaky_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "存取被拒", str(destination))
        real_replace(source, destination)

    monkeypatch.setattr(
        "ai_quant_trading.paper_trading.realtime.os.replace",
        flaky_replace,
    )

    status.update(state="running")

    assert attempts == 3
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "running"
    assert not list(tmp_path.glob(".status.json.*.tmp"))


def test_wait_for_decision_uses_rest_when_kline_stream_is_silent(tmp_path: Path) -> None:
    calls: list[tuple[str, bool]] = []

    class _Stream:
        def snapshot(self) -> dict[str, object]:
            return {"state": "connected", "connected": True, "spread_bps": 0.1}

    session = object.__new__(RealtimePaperTradingSession)
    session.config = SimpleNamespace(decision_interval="15m", settle_seconds=0.0)
    session.stream = _Stream()
    session._decisions = Queue()
    session._last_processed_timestamp = pd.Timestamp("2026-08-20T00:00:00Z")
    session.run_decision = lambda trigger, *, sync_latest: (
        calls.append((trigger, sync_latest)) or "fallback-result"
    )

    result = session._wait_for_decision(automation_paths(tmp_path / "automation"))

    assert result == "fallback-result"
    assert calls == [("rest_close_fallback", True)]
