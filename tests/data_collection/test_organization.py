"""既有 CSV 資料整理測試。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from ai_quant_trading.data_collection.organization import (
    apply_organization,
    plan_market_data_organization,
)


class MarketDataOrganizationTests(unittest.TestCase):
    def test_legacy_crypto_and_equity_files_are_planned_and_moved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            processed = root / "processed"
            crypto_source = raw / "binance" / "ohlcv" / "old_btc.csv"
            equity_source = raw / "yahoo_finance" / "ohlcv" / "old_aapl.csv"
            crypto_source.parent.mkdir(parents=True)
            equity_source.parent.mkdir(parents=True)
            processed.mkdir()
            common = {
                "timestamp": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
                "interval": ["1d", "1d"],
                "open": [1, 2],
                "high": [2, 3],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
                "volume": [10, 20],
            }
            pd.DataFrame(
                common | {"symbol": ["BTC/USDT"] * 2, "exchange": ["binance"] * 2}
            ).to_csv(crypto_source, index=False)
            pd.DataFrame(
                common
                | {"symbol": ["AAPL"] * 2, "exchange": ["yahoo_finance"] * 2}
            ).to_csv(equity_source, index=False)

            actions = plan_market_data_organization(raw, processed)

            self.assertEqual(len(actions), 2)
            targets = {action.target.name for action in actions}
            self.assertIn("ohlcv_crypto_binance_BTC-USDT_1d_latest.csv", targets)
            self.assertIn(
                "ohlcv_us_equity_yahoo_finance_AAPL_1d_latest.csv",
                targets,
            )
            apply_organization(actions, [raw, processed])
            self.assertTrue(all(action.target.exists() for action in actions))
            self.assertFalse(crypto_source.exists())
            self.assertFalse(equity_source.exists())


if __name__ == "__main__":
    unittest.main()
