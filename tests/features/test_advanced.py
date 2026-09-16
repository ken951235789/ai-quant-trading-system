"""進階特徵的因果對齊與市場廣度測試。"""

from __future__ import annotations

import pandas as pd

from ai_quant_trading.features.advanced import (
    attach_advanced_context,
    build_market_breadth,
)


def test_macro_context_never_uses_future_release() -> None:
    market = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-03T00:00:00Z"],
            "close": [100, 101],
        }
    )
    macro = pd.DataFrame(
        {
            "timestamp": ["2024-01-02T00:00:00Z"],
            "series_id": ["VIXCLS"],
            "value": [20],
            "change": [1],
            "pct_change": [0.05],
            "zscore_60": [1.2],
        }
    )

    result = attach_advanced_context(market, macro_frames=[macro])

    assert pd.isna(result.iloc[0]["macro_vixcls_value"])
    assert result.iloc[1]["macro_vixcls_value"] == 20
    assert result["macro_available"].tolist() == [0, 1]


def test_market_breadth_uses_each_symbols_rolling_history() -> None:
    timestamps = pd.date_range("2024-01-01", periods=25, freq="D", tz="UTC")
    first = pd.DataFrame({"timestamp": timestamps, "close": range(100, 125)})
    second = pd.DataFrame({"timestamp": timestamps, "close": range(200, 175, -1)})

    result = build_market_breadth({"UP": first, "DOWN": second})

    assert result.iloc[-1]["breadth_market_count"] == 2
    assert result.iloc[-1]["breadth_advance_ratio"] == 0.5
    assert result.iloc[-1]["breadth_pct_above_sma_20"] == 0.5
