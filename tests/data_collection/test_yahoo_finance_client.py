from __future__ import annotations

import unittest

from ai_quant_trading.data_collection.collectors import collect_yahoo_finance_data
from ai_quant_trading.data_collection.yahoo_finance import (
    YahooFinanceClient,
    normalize_us_symbol,
)


SAMPLE_PAYLOAD = {
    "chart": {
        "error": None,
        "result": [
            {
                "timestamp": [1704153600, 1704240000, 1704326400],
                "indicators": {
                    "quote": [
                        {
                            "open": [180.0, 181.0, 182.0],
                            "high": [185.0, 186.0, 187.0],
                            "low": [178.0, 179.0, 180.0],
                            "close": [184.0, 185.0, 186.0],
                            "volume": [10_000_000, 11_000_000, 12_000_000],
                        }
                    ],
                    "adjclose": [{"adjclose": [183.5, 184.5, 185.5]}],
                },
                "events": {
                    "dividends": {
                        "1704240000": {"date": 1704240000, "amount": 0.24}
                    }
                },
            }
        ],
    }
}


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return SAMPLE_PAYLOAD


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse()


class YahooFinanceClientTest(unittest.TestCase):
    def test_normalize_us_symbol(self):
        self.assertEqual(normalize_us_symbol(" aapl "), "AAPL")

    def test_fetch_ohlcv_parses_chart_response_and_keeps_last_limit_rows(self):
        session = FakeSession()
        client = YahooFinanceClient(session=session)

        frame = client.fetch_ohlcv(
            symbol="aapl",
            interval="1d",
            start="2024-01-01",
            end="2024-01-05",
            limit=2,
        )

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.loc[0, "symbol"], "AAPL")
        self.assertEqual(frame.loc[0, "exchange"], "yahoo_finance")
        self.assertEqual(frame.loc[0, "open"], 181.0)
        self.assertEqual(frame.loc[0, "adj_close"], 184.5)
        self.assertEqual(frame.loc[0, "volume"], 11_000_000)
        self.assertEqual(frame.loc[0, "dividends"], 0.24)
        self.assertIn("AAPL", session.calls[0][0])
        self.assertEqual(session.calls[0][1]["params"]["interval"], "1d")
        self.assertIn("period1", session.calls[0][1]["params"])

    def test_collect_yahoo_finance_data_writes_csv(self):
        import tempfile
        from pathlib import Path

        client = YahooFinanceClient(session=FakeSession())
        with tempfile.TemporaryDirectory() as temp_dir:
            result = collect_yahoo_finance_data(
                symbols=["AAPL"],
                interval="1d",
                raw_dir=Path(temp_dir),
                start="2024-01-01",
                end="2024-01-05",
                limit=2,
                client=client,
            )

            self.assertEqual(len(result.ohlcv_files), 1)
            output_path = result.ohlcv_files[0]
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.parent.name, "ohlcv")
            self.assertEqual(output_path.parent.parent.name, "yahoo_finance")

            import pandas as pd

            saved = pd.read_csv(output_path)
            self.assertEqual(list(saved["symbol"].unique()), ["AAPL"])
            self.assertIn("adj_close", saved.columns)

    def test_intraday_defaults_keep_enough_history_for_feature_warmup(self):
        hourly_session = FakeSession()
        YahooFinanceClient(session=hourly_session).fetch_ohlcv(
            symbol="AAPL",
            interval="1h",
            limit=2,
        )
        minute_session = FakeSession()
        YahooFinanceClient(session=minute_session).fetch_ohlcv(
            symbol="AAPL",
            interval="30m",
            limit=2,
        )

        hourly_params = hourly_session.calls[0][1]["params"]
        minute_params = minute_session.calls[0][1]["params"]
        self.assertEqual(hourly_params["interval"], "60m")
        self.assertEqual(hourly_params["range"], "6mo")
        self.assertEqual(minute_params["range"], "1mo")


if __name__ == "__main__":
    unittest.main()
