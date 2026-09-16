"""模擬帳戶路徑安全性測試。"""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import pandas as pd

from ai_quant_trading.paper_trading import PaperAccountState, PaperTradingConfig
from ai_quant_trading.paper_trading.storage import append_csv_row, validate_account_id
from ai_quant_trading.risk import RiskConfig


class PaperStorageTests(unittest.TestCase):
    def test_account_id_accepts_simple_name(self) -> None:
        self.assertEqual(validate_account_id("demo_AAPL-01"), "demo_AAPL-01")

    def test_account_id_rejects_parent_path(self) -> None:
        with self.assertRaises(ValueError):
            validate_account_id("../outside")

    def test_old_account_payload_defaults_to_rl_model(self) -> None:
        state = PaperAccountState.create(
            account_id="legacy",
            model_dir="old_model",
            exchange="binance",
            symbol="BTC/USDT",
            interval="1d",
            config=PaperTradingConfig(),
            risk_config=RiskConfig(),
        )
        payload = state.to_dict()
        payload.pop("model_kind")
        payload.pop("rl_entry_threshold")
        payload.pop("rl_exit_threshold")
        payload.pop("short_carry_paid")
        payload.pop("position_carry_cost")
        payload["config"].pop("allow_short")
        payload["config"].pop("max_short_fraction")
        payload["config"].pop("short_borrow_rate_annual")

        restored = PaperAccountState.from_dict(payload)

        self.assertEqual(restored.model_kind, "rl")
        self.assertEqual(restored.rl_entry_threshold, 0.10)
        self.assertEqual(restored.rl_exit_threshold, 0.02)
        self.assertFalse(restored.config.allow_short)
        self.assertEqual(restored.short_carry_paid, 0.0)

    def test_append_csv_row_migrates_new_columns_without_losing_old_rows(self) -> None:
        """新增欄位後第一次追加，應先補齊舊 CSV 結構。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "performance.csv"
            path.write_text("timestamp,equity\nold,1000\n", encoding="utf-8")

            append_csv_row(
                path,
                ["timestamp", "equity", "short_carry_cost"],
                {"timestamp": "new", "equity": 1001, "short_carry_cost": 0.5},
            )

            frame = pd.read_csv(path)
            self.assertEqual(
                list(frame.columns),
                ["timestamp", "equity", "short_carry_cost"],
            )
            self.assertEqual(list(frame["timestamp"]), ["old", "new"])
            self.assertTrue(pd.isna(frame.iloc[0]["short_carry_cost"]))
            self.assertEqual(frame.iloc[1]["short_carry_cost"], 0.5)

    def test_removed_model_account_is_marked_as_legacy(self) -> None:
        """舊 XGBoost 帳戶保留紀錄，但不可被誤當成 PPO 帳戶。"""
        state = PaperAccountState.create(
            account_id="archived",
            model_dir="data/processed/models/run_xgboost",
            exchange="yahoo_finance",
            symbol="AAPL",
            interval="1d",
            config=PaperTradingConfig(),
            risk_config=RiskConfig(),
        )
        payload = state.to_dict()
        payload.pop("model_kind")

        restored = PaperAccountState.from_dict(payload)

        self.assertEqual(restored.model_kind, "legacy")


if __name__ == "__main__":
    unittest.main()
