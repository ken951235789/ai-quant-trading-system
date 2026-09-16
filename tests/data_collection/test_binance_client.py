from __future__ import annotations

import unittest

from ai_quant_trading.data_collection.binance import BinanceSpotClient, to_binance_symbol


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.payloads.pop(0))


class BinanceSpotClientTest(unittest.TestCase):
    def test_to_binance_symbol_removes_separator(self):
        self.assertEqual(to_binance_symbol("BTC/USDT"), "BTCUSDT")
        self.assertEqual(to_binance_symbol("eth-usdt"), "ETHUSDT")

    def test_fetch_ohlcv_parses_binance_kline_payload(self):
        payload = [
            [
                1_704_067_200_000,
                "42000.00",
                "43000.00",
                "41000.00",
                "42500.00",
                "123.45",
                1_704_153_599_999,
                "5246625.00",
                3210,
                "60.00",
                "2550000.00",
                "0",
            ]
        ]
        session = FakeSession([payload])
        client = BinanceSpotClient(session=session)

        frame = client.fetch_ohlcv(
            symbol="BTC/USDT",
            interval="1d",
            start="2024-01-01",
            end="2024-01-02",
        )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "symbol"], "BTC/USDT")
        self.assertEqual(frame.loc[0, "exchange"], "binance")
        self.assertEqual(frame.loc[0, "open"], 42000.0)
        self.assertEqual(frame.loc[0, "high"], 43000.0)
        self.assertEqual(frame.loc[0, "low"], 41000.0)
        self.assertEqual(frame.loc[0, "close"], 42500.0)
        self.assertEqual(frame.loc[0, "volume"], 123.45)
        self.assertEqual(frame.loc[0, "number_of_trades"], 3210)
        self.assertEqual(session.calls[0]["params"]["symbol"], "BTCUSDT")
        self.assertEqual(session.calls[0]["params"]["interval"], "1d")

    def test_fetch_24h_market_data_parses_ticker_payload(self):
        payload = {
            "symbol": "BTCUSDT",
            "priceChange": "100.5",
            "priceChangePercent": "0.25",
            "weightedAvgPrice": "42500.5",
            "prevClosePrice": "42400.0",
            "lastPrice": "42500.0",
            "lastQty": "0.01",
            "bidPrice": "42499.9",
            "bidQty": "1.2",
            "askPrice": "42500.1",
            "askQty": "1.1",
            "openPrice": "42400.0",
            "highPrice": "43000.0",
            "lowPrice": "41000.0",
            "volume": "5000.0",
            "quoteVolume": "212500000.0",
            "openTime": 1_704_067_200_000,
            "closeTime": 1_704_153_599_999,
            "firstId": 100,
            "lastId": 200,
            "count": 101,
        }
        session = FakeSession([payload])
        client = BinanceSpotClient(session=session)

        frame = client.fetch_24h_market_data("BTC/USDT")

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "symbol"], "BTC/USDT")
        self.assertEqual(frame.loc[0, "last_price"], 42500.0)
        self.assertEqual(frame.loc[0, "price_change"], 100.5)
        self.assertEqual(frame.loc[0, "trade_count"], 101)
        self.assertEqual(session.calls[0]["params"]["symbol"], "BTCUSDT")


if __name__ == "__main__":
    unittest.main()

