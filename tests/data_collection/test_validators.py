from __future__ import annotations

import unittest

import pandas as pd

from ai_quant_trading.data_collection.validators import (
    validate_market_data_dataframe,
    validate_ohlcv_dataframe,
)


def make_valid_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00.000Z", "2024-01-02T00:00:00.000Z"],
            "open": [100.0, 110.0],
            "high": [120.0, 130.0],
            "low": [90.0, 100.0],
            "close": [110.0, 120.0],
            "volume": [10.0, 12.0],
        }
    )


def make_valid_market_data_frame() -> pd.DataFrame:
    return pd.DataFrame(
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


class ValidatorsTest(unittest.TestCase):
    def test_validate_ohlcv_accepts_valid_frame(self):
        validate_ohlcv_dataframe(make_valid_ohlcv_frame())

    def test_validate_ohlcv_rejects_unsorted_timestamp(self):
        frame = make_valid_ohlcv_frame().iloc[[1, 0]].reset_index(drop=True)

        with self.assertRaisesRegex(ValueError, "由舊到新"):
            validate_ohlcv_dataframe(frame)

    def test_validate_ohlcv_rejects_duplicate_timestamp(self):
        frame = make_valid_ohlcv_frame()
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]

        with self.assertRaisesRegex(ValueError, "不可重複"):
            validate_ohlcv_dataframe(frame)

    def test_validate_ohlcv_rejects_invalid_high_low(self):
        frame = make_valid_ohlcv_frame()
        frame.loc[0, "high"] = 80.0

        with self.assertRaisesRegex(ValueError, "high"):
            validate_ohlcv_dataframe(frame)

    def test_validate_market_data_accepts_valid_frame(self):
        validate_market_data_dataframe(make_valid_market_data_frame())

    def test_validate_market_data_rejects_negative_trade_count(self):
        frame = make_valid_market_data_frame()
        frame.loc[0, "trade_count"] = -1

        with self.assertRaisesRegex(ValueError, "trade_count"):
            validate_market_data_dataframe(frame)


if __name__ == "__main__":
    unittest.main()

