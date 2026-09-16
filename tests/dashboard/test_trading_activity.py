"""機器人即時交易圖資料整理測試。"""

from __future__ import annotations

import pandas as pd

from ai_quant_trading.dashboard.trading_activity import (
    merge_live_kline,
    realtime_stream_is_fresh,
)


def test_live_kline_replaces_same_timestamp_and_is_marked_live() -> None:
    stored = pd.DataFrame(
        {
            "timestamp": ["2026-08-15T09:25:00Z", "2026-08-15T09:30:00Z"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 12.0],
        }
    )
    status = {
        "stream": {
            "connected": True,
            "last_message_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "latest_klines_by_interval": {
                "5m": {
                    "timestamp": "2026-08-15T09:30:00Z",
                    "open": 101.0,
                    "high": 104.0,
                    "low": 100.0,
                    "close": 103.5,
                    "volume": 14.0,
                    "closed": False,
                }
            },
        }
    }

    result = merge_live_kline(stored, status, "5m")

    assert len(result) == 2
    assert result.iloc[-1]["close"] == 103.5
    assert bool(result.iloc[-1]["_is_live"]) is True


def test_closed_stream_kline_is_not_marked_live() -> None:
    status = {
        "stream": {
            "latest_klines_by_interval": {
                "5m": {
                    "timestamp": "2026-08-15T09:30:00Z",
                    "open": 101.0,
                    "high": 104.0,
                    "low": 100.0,
                    "close": 103.5,
                    "volume": 14.0,
                    "closed": True,
                }
            }
        }
    }

    result = merge_live_kline(pd.DataFrame(), status, "5m")

    assert len(result) == 1
    assert bool(result.iloc[-1]["_is_live"]) is False


def test_disconnected_unclosed_kline_is_not_shown_as_realtime() -> None:
    status = {
        "stream": {
            "connected": False,
            "latest_klines_by_interval": {
                "5m": {
                    "timestamp": "2026-08-15T09:30:00Z",
                    "open": 101.0,
                    "high": 104.0,
                    "low": 100.0,
                    "close": 103.5,
                    "volume": 14.0,
                    "closed": False,
                }
            },
        }
    }

    result = merge_live_kline(pd.DataFrame(), status, "5m")

    assert result.empty


def test_stale_connected_flag_is_not_treated_as_live() -> None:
    now = pd.Timestamp("2026-08-15T10:00:00Z")
    status = {
        "stream": {
            "connected": True,
            "last_message_at": "2026-08-15T09:59:00Z",
        }
    }

    assert realtime_stream_is_fresh(status, now=now) is False


def test_merge_live_kline_can_keep_all_available_rows() -> None:
    stored = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-01", periods=600, freq="15min", tz="UTC"),
            "open": range(600),
            "high": range(1, 601),
            "low": range(600),
            "close": range(1, 601),
            "volume": [1.0] * 600,
        }
    )

    result = merge_live_kline(stored, None, "15m", max_bars=None)

    assert len(result) == 600
