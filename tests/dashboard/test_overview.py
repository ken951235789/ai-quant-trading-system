"""交易總覽 Watchlist 測試。"""

from __future__ import annotations

import pandas as pd

from ai_quant_trading.dashboard.overview_page import build_watchlist


def test_watchlist_keeps_latest_file_per_market(tmp_path) -> None:
    older = tmp_path / "older.csv"
    latest = tmp_path / "latest.csv"
    base = {
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC"),
        "symbol": ["AAPL", "AAPL"],
        "exchange": ["yahoo_finance", "yahoo_finance"],
        "interval": ["1d", "1d"],
        "open": [100, 101],
        "high": [102, 103],
        "low": [99, 100],
        "close": [101, 102],
        "volume": [1_000, 1_100],
    }
    pd.DataFrame(base).to_csv(older, index=False)
    updated = pd.DataFrame(base)
    updated["close"] = [102, 104]
    updated.to_csv(latest, index=False)

    result = build_watchlist(
        ((str(latest), latest.stat().st_mtime), (str(older), older.stat().st_mtime))
    )

    assert len(result) == 1
    assert result.iloc[0]["最新價"] == 104
    assert result.iloc[0]["檔案"] == str(latest)
