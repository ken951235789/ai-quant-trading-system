"""多週期資料守門測試。"""

from __future__ import annotations

import pandas as pd
import pytest

from ai_quant_trading.trading import (
    MarketDataGuard,
    MarketDataGuardConfig,
    MarketDataRejected,
)


def _bars(interval: str, frequency: str, *, periods: int = 20) -> pd.DataFrame:
    timestamp = pd.date_range("2026-08-20T00:00:00Z", periods=periods, freq=frequency)
    close = pd.Series(range(100, 100 + periods), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "symbol": "BTC/USDT",
            "exchange": "binance_futures",
            "interval": interval,
            "open": close,
            "high": close + 2,
            "low": close - 2,
            "close": close + 1,
            "volume": 10.0,
        }
    )


def test_twenty_closed_bars_pass_data_guard() -> None:
    frame = _bars("15m", "15min")
    guard = MarketDataGuard(
        MarketDataGuardConfig(
            required_intervals=("15m",),
            minimum_contiguous_bars=20,
            freshness_multiple=2.5,
            settle_seconds=0,
            require_spread=True,
        )
    )
    now = pd.Timestamp(frame.iloc[-1]["timestamp"]) + pd.Timedelta(minutes=15)

    health = guard.ensure_healthy({"15m": frame}, now=now, spread_bps=0.8)

    assert health.ready
    assert health.row_counts == {"15m": 20}
    assert not health.reasons


def test_gap_is_rejected_fail_closed() -> None:
    frame = _bars("15m", "15min", periods=21).drop(index=10).reset_index(drop=True)
    guard = MarketDataGuard(
        MarketDataGuardConfig(required_intervals=("15m",), settle_seconds=0)
    )
    now = pd.Timestamp(frame.iloc[-1]["timestamp"]) + pd.Timedelta(minutes=15)

    with pytest.raises(MarketDataRejected, match="缺口"):
        guard.ensure_healthy({"15m": frame}, now=now, spread_bps=0.5)


def test_unclosed_latest_bar_is_rejected() -> None:
    frame = _bars("15m", "15min")
    guard = MarketDataGuard(
        MarketDataGuardConfig(required_intervals=("15m",), settle_seconds=2)
    )
    now = pd.Timestamp(frame.iloc[-1]["timestamp"]) + pd.Timedelta(minutes=14)

    health = guard.assess({"15m": frame}, now=now, spread_bps=0.5)

    assert not health.ready
    assert any("尚未完成收盤" in reason for reason in health.reasons)


def test_wide_spread_is_rejected_but_missing_optional_spread_is_warning() -> None:
    frame = _bars("15m", "15min")
    now = pd.Timestamp(frame.iloc[-1]["timestamp"]) + pd.Timedelta(minutes=15)
    guard = MarketDataGuard(
        MarketDataGuardConfig(required_intervals=("15m",), settle_seconds=0)
    )

    missing = guard.assess({"15m": frame}, now=now)
    wide = guard.assess({"15m": frame}, now=now, spread_bps=8.0)

    assert missing.ready
    assert missing.warnings
    assert not wide.ready
    assert any("Spread" in reason for reason in wide.reasons)
