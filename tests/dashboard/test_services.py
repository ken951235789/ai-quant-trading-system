"""Dashboard 資料服務測試。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.dashboard.services import (
    ensure_chart_features,
    list_ohlcv_files,
    list_processed_files,
    load_market_frame,
    parse_symbols,
    relative_file_label,
    summarize_market,
)
from ai_quant_trading.features.indicators import FEATURE_COLUMNS


def make_market_frame(rows: int = 230) -> pd.DataFrame:
    """建立 Dashboard 測試使用的日線資料。"""
    index = np.arange(rows, dtype=float)
    close = 100 + index * 0.1 + np.sin(index / 4)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "symbol": "AAPL",
            "exchange": "yahoo_finance",
            "interval": "1d",
            "open": close - 0.2,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000 + index,
        }
    )


class DashboardServicesTest(unittest.TestCase):
    def test_parse_symbols_normalizes_and_removes_duplicates(self):
        symbols = parse_symbols(" btc/usdt, ETH/USDT\nbtc/usdt, AAPL ")

        self.assertEqual(symbols, ["BTC/USDT", "ETH/USDT", "AAPL"])

    def test_lists_raw_and_processed_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_file = root / "raw" / "binance" / "ohlcv" / "btc.csv"
            processed_file = root / "processed" / "features.csv"
            raw_file.parent.mkdir(parents=True)
            processed_file.parent.mkdir(parents=True)
            raw_file.write_text("timestamp,close\n", encoding="utf-8")
            processed_file.write_text("timestamp,close\n", encoding="utf-8")

            self.assertEqual(list_ohlcv_files(root / "raw"), [raw_file])
            self.assertEqual(list_processed_files(root / "processed"), [processed_file])
            self.assertEqual(
                relative_file_label(processed_file, root),
                str(Path("processed") / "features.csv"),
            )

    def test_lists_ohlcv_with_market_category_layer(self):
        """市場分類層加入後仍應找到虛擬貨幣與美股 OHLCV。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "raw"
            crypto_file = root / "crypto" / "binance" / "ohlcv" / "btc.csv"
            equity_file = root / "us_equity" / "yahoo_finance" / "ohlcv" / "aapl.csv"
            crypto_file.parent.mkdir(parents=True)
            equity_file.parent.mkdir(parents=True)
            crypto_file.touch()
            equity_file.touch()

            self.assertEqual(set(list_ohlcv_files(root)), {crypto_file, equity_file})

    def test_load_summary_and_feature_enrichment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "market.csv"
            make_market_frame().to_csv(path, index=False)

            loaded = load_market_frame(path)
            summary = summarize_market(loaded.tail(20))
            enriched = ensure_chart_features(loaded)

            self.assertEqual(summary.bars, 20)
            self.assertEqual(summary.latest_close, float(loaded.iloc[-1]["close"]))
            self.assertTrue(set(FEATURE_COLUMNS).issubset(enriched.columns))
            self.assertIn("future_return", enriched.columns)


if __name__ == "__main__":
    unittest.main()
