from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ai_quant_trading.data_collection.csv_storage import (
    save_market_data_csv,
    save_ohlcv_csv,
    symbol_to_file_slug,
)


class CsvStorageTest(unittest.TestCase):
    def test_symbol_to_file_slug(self):
        self.assertEqual(symbol_to_file_slug("BTC/USDT"), "BTCUSDT")

    def test_save_ohlcv_csv_writes_expected_path(self):
        frame = pd.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00.000Z", "2024-01-02T00:00:00.000Z"],
                "open": [100.0, 110.0],
                "high": [120.0, 130.0],
                "low": [90.0, 100.0],
                "close": [110.0, 120.0],
                "volume": [10.0, 12.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_ohlcv_csv(frame, Path(temp_dir), "binance", "BTC/USDT", "1d")

            self.assertTrue(path.exists())
            self.assertEqual(path.parent.name, "ohlcv")
            self.assertEqual(path.parent.parent.parent.name, "crypto")
            self.assertEqual(path.name, "ohlcv_crypto_binance_BTC-USDT_1d_latest.csv")
            saved = pd.read_csv(path)
            self.assertEqual(len(saved), 2)

    def test_save_ohlcv_merges_new_rows_and_removes_legacy_files(self):
        first = pd.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
                "open": [100.0, 110.0],
                "high": [120.0, 130.0],
                "low": [90.0, 100.0],
                "close": [110.0, 120.0],
                "volume": [10.0, 12.0],
            }
        )
        update = first.iloc[[1]].copy()
        update.loc[:, "close"] = 125.0
        update.loc[:, "timestamp"] = "2024-01-02T00:00:00Z"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = save_ohlcv_csv(first, root, "binance", "BTC/USDT", "1d")
            legacy = path.with_name(
                "ohlcv_crypto_binance_BTC-USDT_1d_20230101_20230101.csv"
            )
            legacy_frame = first.iloc[[0]].copy()
            legacy_frame.loc[:, "timestamp"] = "2023-01-01T00:00:00Z"
            legacy_frame.to_csv(legacy, index=False)

            updated_path = save_ohlcv_csv(update, root, "binance", "BTC/USDT", "1d")
            saved = pd.read_csv(updated_path)

            self.assertEqual(updated_path, path)
            self.assertFalse(legacy.exists())
            self.assertEqual(len(saved), 3)
            self.assertEqual(float(saved.iloc[-1]["close"]), 125.0)

    def test_save_ohlcv_keeps_mixed_iso_timestamp_formats(self):
        existing = pd.DataFrame(
            {
                "timestamp": ["2026-07-23T13:30:00+00:00"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [10.0],
            }
        )
        update = existing.copy()
        update.loc[:, "timestamp"] = "2026-07-31T20:00:00.000Z"
        update.loc[:, "close"] = 105.0

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = save_ohlcv_csv(existing, root, "yahoo_finance", "AAPL", "1h")
            save_ohlcv_csv(update, root, "yahoo_finance", "AAPL", "1h")
            saved = pd.read_csv(path)

            self.assertEqual(len(saved), 2)
            self.assertEqual(
                pd.to_datetime(saved["timestamp"], utc=True, format="mixed").max(),
                pd.Timestamp("2026-07-31T20:00:00Z"),
            )

    def test_save_ohlcv_can_keep_a_fixed_rolling_window(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01T00:00:00Z",
                    periods=8,
                    freq="1min",
                ),
                "open": range(8),
                "high": range(1, 9),
                "low": range(8),
                "close": range(1, 9),
                "volume": range(10, 18),
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_ohlcv_csv(
                frame,
                Path(temp_dir),
                "binance",
                "BTC/USDT",
                "1m",
                max_rows=3,
            )
            saved = pd.read_csv(path)

        self.assertEqual(len(saved), 3)
        self.assertEqual(float(saved.iloc[-1]["close"]), 8.0)

    def test_save_market_data_csv_writes_expected_path(self):
        frame = pd.DataFrame(
            {
                "timestamp": ["2024-01-02T00:00:00.000Z"],
                "symbol": ["BTC/USDT"],
                "price_change": [100.0],
                "price_change_percent": [1.5],
                "volume": [500.0],
                "quote_volume": [21_000_000.0],
                "trade_count": [12345],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_market_data_csv(frame, Path(temp_dir), "binance", "BTC/USDT")

            self.assertTrue(path.exists())
            self.assertEqual(path.parent.name, "market_data")
            self.assertEqual(path.parent.parent.parent.name, "crypto")
            self.assertEqual(path.name, "market_crypto_binance_BTC-USDT_latest.csv")
            saved = pd.read_csv(path)
            self.assertEqual(len(saved), 1)


if __name__ == "__main__":
    unittest.main()
