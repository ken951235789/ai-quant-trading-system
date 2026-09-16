"""多週期市場資料收集流程測試。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_quant_trading.data_collection import collectors
from ai_quant_trading.data_collection.collectors import (
    DataCollectionResult,
    collect_binance_timeframe_bundle,
)
from ai_quant_trading.market_clock import BTC_MULTITIMEFRAME_INTERVALS


def test_collect_binance_timeframe_bundle_downloads_every_interval(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_collect_binance_data(**kwargs) -> DataCollectionResult:
        calls.append(kwargs)
        interval = str(kwargs["interval"])
        return DataCollectionResult(ohlcv_files=[Path(f"{interval}.csv")])

    monkeypatch.setattr(collectors, "collect_binance_data", fake_collect_binance_data)
    progress: list[tuple[str, int, int]] = []

    result = collect_binance_timeframe_bundle(
        ["BTC/USDT"],
        "data/raw",
        start="2025-06-01",
        end="2025-06-30",
        progress_callback=lambda interval, completed, total: progress.append(
            (interval, completed, total)
        ),
    )

    assert tuple(str(call["interval"]) for call in calls) == BTC_MULTITIMEFRAME_INTERVALS
    assert len(result.ohlcv_files) == len(BTC_MULTITIMEFRAME_INTERVALS)
    expected_count = len(BTC_MULTITIMEFRAME_INTERVALS)
    assert progress[-1] == ("1d", expected_count, expected_count)
    assert calls[0]["include_market_data"] is True
    assert all(call["include_market_data"] is False for call in calls[1:])
    assert pd.Timestamp(str(calls[-1]["start"])) < pd.Timestamp("2025-01-01", tz="UTC")
