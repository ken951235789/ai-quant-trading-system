"""特徵 CSV 建置流程測試。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.features.builder import (
    build_features_from_csv,
    build_features_from_csv_batch,
    infer_annualization_periods,
)


def make_source_frame(rows: int = 230) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100 + index * 0.1 + np.sin(index / 3)
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
            "volume": 10_000 + index,
        }
    )


class FeatureBuilderTest(unittest.TestCase):
    def test_infer_annualization_periods_by_exchange(self):
        equity = make_source_frame(2)
        crypto = equity.copy()
        crypto["exchange"] = "binance"

        self.assertEqual(infer_annualization_periods(equity), 252)
        self.assertEqual(infer_annualization_periods(crypto), 365)

        equity["interval"] = "1h"
        crypto["interval"] = "1h"
        self.assertEqual(infer_annualization_periods(equity), 252 * 7)
        self.assertEqual(infer_annualization_periods(crypto), 365 * 24)

    def test_build_features_from_csv_writes_processed_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "ohlcv.csv"
            output_dir = Path(temp_dir) / "processed"
            make_source_frame().to_csv(source_path, index=False)
            legacy = (
                output_dir
                / "features_us_equity_yahoo_finance_AAPL_1d_20240101_20240701_h3.csv"
            )
            output_dir.mkdir()
            legacy.write_text("timestamp,close\n", encoding="utf-8")

            frame, output_path = build_features_from_csv(
                source_path,
                output_dir=output_dir,
                target_horizon=3,
            )

            self.assertTrue(output_path.exists())
            self.assertIn("features_us_equity_yahoo_finance_AAPL_1d", output_path.name)
            self.assertTrue(output_path.name.endswith("_h3.csv"))
            self.assertFalse(legacy.exists())
            self.assertFalse(frame.empty)
            saved = pd.read_csv(output_path)
            self.assertEqual(len(saved), len(frame))
            self.assertIn("target", saved.columns)
            self.assertIn("target_timestamp", saved.columns)
            self.assertTrue(saved["target"].tail(3).isna().all())

    def test_rejects_short_dataset_with_clear_message(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "short.csv"
            make_source_frame(20).to_csv(source_path, index=False)

            with self.assertRaisesRegex(ValueError, "資料不足"):
                build_features_from_csv(source_path, output_dir=temp_dir)

    def test_excludes_latest_unfinished_daily_bar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "ohlcv.csv"
            source = make_source_frame(230)
            source.loc[source.index[-1], "timestamp"] = pd.Timestamp.now(tz="UTC").floor("D")
            source.to_csv(source_path, index=False)

            frame, _ = build_features_from_csv(source_path, output_dir=temp_dir)

            self.assertLess(
                pd.Timestamp(frame["target_timestamp"].max()),
                pd.Timestamp.now(tz="UTC").floor("D"),
            )

    def test_rejects_missing_input_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "找不到"):
            build_features_from_csv("missing.csv")

    def test_batch_builds_multiple_independent_feature_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "aapl.csv"
            second = root / "msft.csv"
            make_source_frame().to_csv(first, index=False)
            second_frame = make_source_frame()
            second_frame["symbol"] = "MSFT"
            second_frame.to_csv(second, index=False)

            artifacts = build_features_from_csv_batch(
                [first, second],
                output_dir=root / "processed",
            )

            self.assertEqual(len(artifacts), 2)
            self.assertTrue(all(artifact.rows > 0 for artifact in artifacts))
            self.assertEqual(len({artifact.output_path for artifact in artifacts}), 2)
            self.assertTrue(all(artifact.output_path.exists() for artifact in artifacts))

    def test_batch_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "至少需要一份"):
            build_features_from_csv_batch([])


if __name__ == "__main__":
    unittest.main()
