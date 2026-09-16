"""技術指標與 target 測試。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ai_quant_trading.features.indicators import (
    FEATURE_COLUMNS,
    SHORT_TERM_FEATURE_COLUMNS,
    add_market_structure_features,
    add_target,
    build_feature_dataset,
)


def make_ohlcv_frame(rows: int = 260) -> pd.DataFrame:
    """建立具趨勢與週期波動的測試 OHLCV。"""
    index = np.arange(rows, dtype=float)
    close = 100 + index * 0.2 + np.sin(index / 4) * 2
    open_price = close - np.cos(index / 5) * 0.5
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "symbol": "TEST",
            "exchange": "yahoo_finance",
            "interval": "1d",
            "open": open_price,
            "high": np.maximum(open_price, close) + 1,
            "low": np.minimum(open_price, close) - 1,
            "close": close,
            "volume": 1_000 + (index % 20) * 10,
        }
    )


class IndicatorsTest(unittest.TestCase):
    def test_build_feature_dataset_contains_all_features(self):
        result = build_feature_dataset(make_ohlcv_frame(), drop_na=True)

        self.assertFalse(result.empty)
        self.assertTrue(set(FEATURE_COLUMNS).issubset(result.columns))
        self.assertIn("future_return", result.columns)
        self.assertIn("target", result.columns)
        self.assertFalse(result[[*FEATURE_COLUMNS, "future_return", "target"]].isna().any().any())
        self.assertTrue(result["rsi_14"].between(0, 100).all())
        self.assertTrue(result["short_rsi_2"].between(0, 100).all())
        self.assertTrue(
            result[["short_reversal_entry", "short_reversal_exit", "short_reversal_position"]]
            .isin([0, 1])
            .all()
            .all()
        )
        self.assertTrue(set(SHORT_TERM_FEATURE_COLUMNS).issubset(result.columns))

    def test_target_is_aligned_to_future_period(self):
        frame = pd.DataFrame({"close": [100.0, 110.0, 105.0]})
        result = add_target(frame, horizon=1)

        self.assertAlmostEqual(result.loc[0, "future_return"], 0.10)
        self.assertEqual(result.loc[0, "target"], 1)
        self.assertEqual(result.loc[1, "target"], 0)
        self.assertTrue(pd.isna(result.loc[2, "target"]))

    def test_breakout_uses_previous_window_high(self):
        frame = pd.DataFrame(
            {
                "high": [10.0] * 20 + [12.0],
                "low": [8.0] * 21,
                "close": [9.0] * 20 + [11.0],
            }
        )
        result = add_market_structure_features(frame)

        self.assertEqual(result.loc[19, "breakout_20"], 0)
        self.assertEqual(result.loc[20, "breakout_20"], 1)

    def test_future_price_change_does_not_change_past_features(self):
        original = make_ohlcv_frame()
        changed = original.copy()
        changed.loc[220:, ["open", "high", "low", "close"]] *= 5

        original_features = build_feature_dataset(original, drop_na=False)
        changed_features = build_feature_dataset(changed, drop_na=False)

        assert_frame_equal(
            original_features.loc[:219, FEATURE_COLUMNS],
            changed_features.loc[:219, FEATURE_COLUMNS],
            check_dtype=False,
        )

    def test_rejects_invalid_target_horizon(self):
        with self.assertRaisesRegex(ValueError, "horizon"):
            add_target(pd.DataFrame({"close": [100.0]}), horizon=0)

    def test_optional_intraday_sources_use_neutral_values_and_flags(self):
        result = build_feature_dataset(make_ohlcv_frame(), drop_na=True)

        self.assertTrue(result["microstructure_available"].eq(0).all())
        self.assertTrue(result["derivatives_available"].eq(0).all())
        self.assertTrue(result["taker_buy_ratio"].eq(0.5).all())
        self.assertTrue(result["funding_percentile_200"].eq(0.5).all())

    def test_supertrend_supports_pandas_copy_on_write(self):
        """Pandas 3 預設唯讀 NumPy view 時仍可逐根計算 Supertrend。"""
        with pd.option_context("mode.copy_on_write", True):
            result = build_feature_dataset(make_ohlcv_frame(), drop_na=True)

        self.assertFalse(result.empty)
        self.assertTrue(result["supertrend_direction_10_3"].isin([-1.0, 1.0]).all())


if __name__ == "__main__":
    unittest.main()
