"""Binance Futures 公開市場脈絡測試。"""

from __future__ import annotations

import pandas as pd

from ai_quant_trading.data_collection.binance_futures import (
    BinanceFuturesPublicClient,
    attach_derivatives_context,
)
from tests.data_collection.test_binance_client import FakeSession


def test_fetch_market_context_combines_funding_open_interest_and_spread() -> None:
    session = FakeSession(
        [
            [
                {
                    "symbol": "BTCUSDT",
                    "fundingRate": "0.0001",
                    "fundingTime": 1_704_067_200_000,
                    "markPrice": "42000",
                }
            ],
            [
                {
                    "symbol": "BTCUSDT",
                    "sumOpenInterest": "1000",
                    "sumOpenInterestValue": "42000000",
                    "timestamp": 1_704_067_200_000,
                }
            ],
            {"bidPrice": "41999", "askPrice": "42001"},
        ]
    )
    client = BinanceFuturesPublicClient(session=session)

    frame = client.fetch_market_context("BTC/USDT", "4h")

    assert frame.iloc[0]["funding_rate"] == 0.0001
    assert frame.iloc[0]["open_interest"] == 1000
    assert frame.iloc[-1]["spread_bps"] > 0
    assert session.calls[1]["params"]["period"] == "4h"


def test_attach_context_never_uses_future_row() -> None:
    ohlcv = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            "close": [100, 101],
        }
    )
    context = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T12:00:00Z"],
            "funding_rate": [0.001],
            "mark_price": [100],
            "open_interest": [500],
            "open_interest_value": [50_000],
            "open_interest_change": [0.02],
            "spread_bps": [1.0],
        }
    )

    result = attach_derivatives_context(ohlcv, context)

    assert pd.isna(result.iloc[0]["funding_rate"])
    assert result.iloc[1]["funding_rate"] == 0.001
    assert result["derivatives_context_available"].tolist() == [0, 1]


def test_advanced_context_combines_ratios_and_basis() -> None:
    timestamp = 1_704_067_200_000
    session = FakeSession(
        [
            [],
            [],
            [
                {
                    "timestamp": timestamp,
                    "longShortRatio": "1.5",
                    "longAccount": "0.6",
                    "shortAccount": "0.4",
                }
            ],
            [
                {
                    "timestamp": timestamp,
                    "buySellRatio": "1.2",
                    "buyVol": "120",
                    "sellVol": "100",
                }
            ],
            [
                {
                    "timestamp": timestamp,
                    "basisRate": "0.001",
                    "basis": "10",
                    "indexPrice": "42000",
                }
            ],
            {"bidPrice": "41999", "askPrice": "42001"},
        ]
    )
    client = BinanceFuturesPublicClient(session=session)

    frame = client.fetch_market_context(
        "BTC/USDT",
        "4h",
        include_advanced=True,
    )

    assert frame.iloc[0]["global_long_short_ratio"] == 1.5
    assert frame.iloc[0]["taker_buy_sell_ratio"] == 1.2
    assert frame.iloc[0]["basis_rate"] == 0.001
    basis_call = session.calls[4]["params"]
    assert basis_call["pair"] == "BTCUSDT"
    assert "startTime" not in basis_call


def test_fetch_futures_order_book_computes_depth_imbalance() -> None:
    session = FakeSession(
        [
            {
                "lastUpdateId": 123,
                "bids": [["100.0", "2.0"], ["99.0", "1.0"]],
                "asks": [["101.0", "1.0"], ["102.0", "1.0"]],
            }
        ]
    )
    client = BinanceFuturesPublicClient(session=session)

    frame = client.fetch_order_book_snapshot("BTC/USDT", limit=20)

    assert frame.iloc[0]["exchange"] == "binance_futures"
    assert frame.iloc[0]["best_bid"] == 100.0
    assert frame.iloc[0]["best_ask"] == 101.0
    assert frame.iloc[0]["imbalance_20"] > 0
    assert session.calls[0]["url"].endswith("/fapi/v1/depth")


def test_fetch_futures_24h_market_data_uses_standard_schema() -> None:
    session = FakeSession(
        [
            {
                "symbol": "BTCUSDT",
                "priceChange": "1200.0",
                "priceChangePercent": "2.0",
                "weightedAvgPrice": "60500.0",
                "prevClosePrice": "60000.0",
                "lastPrice": "61200.0",
                "lastQty": "0.1",
                "openPrice": "60000.0",
                "highPrice": "61500.0",
                "lowPrice": "59800.0",
                "volume": "10000.0",
                "quoteVolume": "605000000.0",
                "openTime": 1_704_067_200_000,
                "closeTime": 1_704_153_600_000,
                "firstId": 10,
                "lastId": 20,
                "count": 11,
            }
        ]
    )
    client = BinanceFuturesPublicClient(session=session)

    frame = client.fetch_24h_market_data("BTC/USDT")

    assert frame.iloc[0]["exchange"] == "binance_futures"
    assert frame.iloc[0]["last_price"] == 61200.0
    assert frame.iloc[0]["trade_count"] == 11
    assert session.calls[0]["url"].endswith("/fapi/v1/ticker/24hr")


def test_fetch_futures_ohlcv_excludes_unclosed_bar() -> None:
    old_open = 1_704_067_200_000
    future_open = 4_102_444_800_000
    session = FakeSession(
        [
            [
                [
                    old_open,
                    "42000",
                    "42100",
                    "41900",
                    "42050",
                    "10",
                    old_open + 899_999,
                    "420500",
                    100,
                    "6",
                    "252300",
                    "0",
                ],
                [
                    future_open,
                    "50000",
                    "50100",
                    "49900",
                    "50050",
                    "10",
                    future_open + 899_999,
                    "500500",
                    100,
                    "6",
                    "300300",
                    "0",
                ],
            ]
        ]
    )
    client = BinanceFuturesPublicClient(session=session)

    frame = client.fetch_ohlcv("BTC/USDT", interval="15m")

    assert len(frame) == 1
    assert frame.iloc[0]["open_time_ms"] == old_open
