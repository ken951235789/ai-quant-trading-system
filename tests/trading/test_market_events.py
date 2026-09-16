"""歷史與即時市場事件共用合約測試。"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ai_quant_trading.data_collection.binance_stream import BinanceKlineUpdate
from ai_quant_trading.research.replay import HistoricalMarketEventSource
from ai_quant_trading.trading.market_events import (
    ConsumerDeliveryError,
    MarketEventBus,
    MarketEventGap,
    OutOfOrderMarketEvent,
)


def _frame(rows: int = 6) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="15min")
    opened = timestamps.view("int64") // 1_000_000
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "exchange": "binance_futures",
            "symbol": "BTC/USDT",
            "interval": "15m",
            "open": [60_000.0 + index for index in range(rows)],
            "high": [60_010.0 + index for index in range(rows)],
            "low": [59_990.0 + index for index in range(rows)],
            "close": [60_005.0 + index for index in range(rows)],
            "volume": 10.0,
            "open_time_ms": opened,
            "close_time_ms": opened + 899_999,
            "collected_at": timestamps + pd.Timedelta(minutes=15),
        }
    )


def test_historical_and_websocket_kline_share_event_id() -> None:
    frame = _frame(2)
    historical = HistoricalMarketEventSource(frame).events[0]
    row = frame.iloc[0]
    websocket = BinanceKlineUpdate(
        symbol="BTC/USDT",
        interval="15m",
        open_time_ms=int(row["open_time_ms"]),
        close_time_ms=int(row["close_time_ms"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        quote_asset_volume=0.0,
        number_of_trades=0,
        taker_buy_base_volume=0.0,
        taker_buy_quote_volume=0.0,
        closed=True,
        event_time_ms=int(row["close_time_ms"]),
    ).to_market_event()

    assert historical.event_id == websocket.event_id
    assert historical.to_ohlcv_row()["close"] == websocket.to_ohlcv_row()["close"]


def test_bus_blocks_duplicate_and_detects_ordering_and_gap() -> None:
    events = HistoricalMarketEventSource(_frame()).events
    delivered: list[str] = []
    bus = MarketEventBus([lambda event: delivered.append(event.event_id)])

    assert bus.publish(events[0]).status == "delivered"
    assert bus.publish(events[0]).status == "duplicate"
    bus.publish(events[2])
    with pytest.raises(OutOfOrderMarketEvent):
        bus.publish(events[1])

    assert len(delivered) == 2
    assert bus.metrics.duplicates == 1
    assert bus.metrics.gaps == 1
    assert bus.metrics.out_of_order == 1

    strict_gap_bus = MarketEventBus(strict_continuity=True)
    strict_gap_bus.publish(events[0])
    with pytest.raises(MarketEventGap):
        strict_gap_bus.publish(events[2])


def test_consumer_failure_does_not_mark_event_as_delivered() -> None:
    event = HistoricalMarketEventSource(_frame(2)).events[0]
    attempts = 0

    def fail_once(_event) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")

    bus = MarketEventBus([fail_once])

    with pytest.raises(ConsumerDeliveryError):
        bus.publish(event)
    receipt = bus.publish(event)

    assert receipt.status == "delivered"
    assert bus.metrics.delivered == 1
    assert bus.metrics.consumer_failures == 1
    assert bus.metrics.duplicates == 0


def test_event_normalizes_naive_observed_time_to_utc() -> None:
    event = HistoricalMarketEventSource(_frame(2)).events[0]
    replaced = event.book_ticker(
        source="rest",
        exchange="binance_futures",
        symbol="BTC/USDT",
        update_id=1,
        observed_at=datetime(2026, 1, 1),
        payload={"bid_price": 60_000, "ask_price": 60_001},
    )

    assert replaced.observed_at.tzinfo == timezone.utc
    with pytest.raises(TypeError):
        replaced.payload["bid_price"] = 1  # type: ignore[index]
