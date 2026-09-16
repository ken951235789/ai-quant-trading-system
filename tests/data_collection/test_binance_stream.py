"""Binance WebSocket 訊息解析與 URL 契約測試。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ai_quant_trading.data_collection.binance_stream import (
    BinanceRealtimeStream,
    BinanceStreamConfig,
    parse_binance_stream_message,
)


def test_parse_closed_kline_to_canonical_frame() -> None:
    message = json.dumps(
        {
            "stream": "btcusdt@kline_5m",
            "data": {
                "e": "kline",
                "E": 1_700_000_300_100,
                "k": {
                    "t": 1_700_000_000_000,
                    "T": 1_700_000_299_999,
                    "s": "BTCUSDT",
                    "i": "5m",
                    "o": "60000.0",
                    "h": "60100.0",
                    "l": "59900.0",
                    "c": "60050.0",
                    "v": "12.5",
                    "n": 321,
                    "x": True,
                    "q": "750000.0",
                    "V": "7.5",
                    "Q": "450000.0",
                },
            },
        }
    )

    parsed = parse_binance_stream_message(message)

    assert parsed is not None
    assert parsed.kline is not None
    assert parsed.kline.closed is True
    frame = parsed.kline.to_frame()
    assert frame.iloc[0]["symbol"] == "BTC/USDT"
    assert frame.iloc[0]["interval"] == "5m"
    assert frame.iloc[0]["close"] == 60050.0
    assert frame.iloc[0]["number_of_trades"] == 321


def test_parse_book_ticker_and_spread() -> None:
    message = json.dumps(
        {
            "stream": "btcusdt@bookTicker",
            "data": {
                "u": 400900217,
                "s": "BTCUSDT",
                "b": "60000.0",
                "B": "1.2",
                "a": "60006.0",
                "A": "0.8",
            },
        }
    )

    parsed = parse_binance_stream_message(message)

    assert parsed is not None
    assert parsed.book_ticker is not None
    assert round(parsed.book_ticker.spread_bps, 3) == 1.0


def test_stream_urls_separate_market_klines_and_public_book_ticker() -> None:
    config = BinanceStreamConfig()

    assert config.kline_stream_url.startswith(
        "wss://fstream.binance.com/market/stream?streams="
    )
    assert "btcusdt@kline_5m" in config.kline_stream_url
    assert "btcusdt@kline_15m" in config.kline_stream_url
    assert "btcusdt@kline_4h" in config.kline_stream_url
    assert "btcusdt@kline_1d" in config.kline_stream_url
    assert "bookTicker" not in config.kline_stream_url
    assert config.book_ticker_stream_url == (
        "wss://fstream.binance.com/public/ws/btcusdt@bookTicker"
    )
    assert len(config.stream_endpoints) == 2


def test_book_ticker_is_sampled_before_json_parsing() -> None:
    message = json.dumps(
        {
            "e": "bookTicker",
            "u": 400900218,
            "s": "BTCUSDT",
            "b": "60000.0",
            "B": "1.2",
            "a": "60003.0",
            "A": "0.8",
        }
    )
    stream = BinanceRealtimeStream(
        BinanceStreamConfig(
            intervals=("5m",),
            book_ticker_sample_seconds=0.1,
        ),
        on_closed_kline=lambda _event: None,
    )

    with (
        patch(
            "ai_quant_trading.data_collection.binance_stream.monotonic",
            side_effect=[10.0, 10.01, 10.11],
        ),
        patch(
            "ai_quant_trading.data_collection.binance_stream.parse_binance_stream_message",
            wraps=parse_binance_stream_message,
        ) as parser,
    ):
        stream._on_message(None, message, "book_ticker")
        stream._on_message(None, message, "book_ticker")
        stream._on_message(None, message, "book_ticker")

    status = stream.snapshot()
    assert parser.call_count == 2
    assert status["messages_received"] == 3
    assert status["book_ticker_messages_processed"] == 2


def test_parse_raw_public_book_ticker_after_binance_stream_migration() -> None:
    message = json.dumps(
        {
            "e": "bookTicker",
            "u": 400900218,
            "s": "BTCUSDT",
            "b": "60000.0",
            "B": "1.2",
            "a": "60003.0",
            "A": "0.8",
            "st": 1,
        }
    )

    parsed = parse_binance_stream_message(message)

    assert parsed is not None
    assert parsed.book_ticker is not None
    assert round(parsed.book_ticker.spread_bps, 3) == 0.5


def test_stream_requires_both_market_channels_before_reporting_connected() -> None:
    stream = BinanceRealtimeStream(
        BinanceStreamConfig(intervals=("5m",)),
        on_closed_kline=lambda _event: None,
    )

    stream._on_open(None, "kline")
    assert stream.snapshot()["kline_connected"] is True
    assert stream.snapshot()["connected"] is False

    stream._on_open(None, "book_ticker")
    assert stream.snapshot()["book_ticker_connected"] is True
    assert stream.snapshot()["connected"] is True


def test_stream_snapshot_exposes_current_unclosed_kline() -> None:
    message = json.dumps(
        {
            "stream": "btcusdt@kline_5m",
            "data": {
                "e": "kline",
                "E": 1_700_000_200_000,
                "k": {
                    "t": 1_700_000_000_000,
                    "T": 1_700_000_299_999,
                    "s": "BTCUSDT",
                    "i": "5m",
                    "o": "60000.0",
                    "h": "60200.0",
                    "l": "59900.0",
                    "c": "60150.0",
                    "v": "8.5",
                    "n": 210,
                    "x": False,
                    "q": "510000.0",
                    "V": "4.0",
                    "Q": "240000.0",
                },
            },
        }
    )
    stream = BinanceRealtimeStream(
        BinanceStreamConfig(intervals=("5m",), include_book_ticker=False),
        on_closed_kline=lambda _event: None,
    )

    stream._on_message(None, message)
    current = stream.snapshot()["latest_klines_by_interval"]["5m"]

    assert current["closed"] is False
    assert current["close"] == 60150.0
    assert current["number_of_trades"] == 210


def test_stream_uses_preconfigured_secure_ssl_context() -> None:
    stream = BinanceRealtimeStream(
        BinanceStreamConfig(intervals=("5m",), include_book_ticker=False),
        on_closed_kline=lambda _event: None,
    )
    context = object()
    app = MagicMock()
    app.run_forever.side_effect = lambda **_kwargs: stream._stop.set()

    with (
        patch(
            "ai_quant_trading.data_collection.binance_stream.create_secure_ssl_context",
            return_value=context,
        ),
        patch("websocket.WebSocketApp", return_value=app),
    ):
        stream._run()

    assert app.run_forever.call_args.kwargs["sslopt"]["context"] is context
