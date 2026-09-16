"""Binance 私有 API 簽章與未知成交狀態測試。"""

from __future__ import annotations

import hashlib
import hmac
from decimal import Decimal
from urllib.parse import parse_qs
import unittest

import requests

from ai_quant_trading.live_trading.binance_client import (
    BinanceExecutionUnknown,
    BinancePrivateClient,
)
from ai_quant_trading.live_trading.credentials import BinanceCredentials


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, params, headers, timeout):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class BinancePrivateClientTests(unittest.TestCase):
    def make_client(self, responses):
        session = FakeSession(responses)
        client = BinancePrivateClient(
            BinanceCredentials("key", "secret"),
            "https://example.test",
            session=session,
            time_provider=lambda: 1_700_000_000_000,
        )
        return client, session

    def test_signed_account_request_contains_api_key_and_hmac(self) -> None:
        client, session = self.make_client([FakeResponse({"canTrade": True, "balances": []})])

        client.fetch_account()

        call = session.calls[0]
        values = parse_qs(call["params"])
        unsigned = "recvWindow=5000&timestamp=1700000000000"
        expected = hmac.new(b"secret", unsigned.encode(), hashlib.sha256).hexdigest()
        self.assertEqual(call["headers"]["X-MBX-APIKEY"], "key")
        self.assertEqual(values["signature"], [expected])

    def test_order_timeout_is_unknown_and_is_not_retried(self) -> None:
        client, session = self.make_client([requests.Timeout("late")])

        with self.assertRaises(BinanceExecutionUnknown) as context:
            client.place_market_order("BTC/USDT", "BUY", quantity=1, client_order_id="aq1")

        self.assertEqual(context.exception.client_order_id, "aq1")
        self.assertEqual(len(session.calls), 1)

    def test_server_error_during_order_is_unknown(self) -> None:
        client, _ = self.make_client([FakeResponse({"msg": "busy"}, status_code=503)])

        with self.assertRaises(BinanceExecutionUnknown):
            client.place_market_order("BTC/USDT", "BUY", quantity=1, client_order_id="aq2")

    def test_oco_uses_current_order_list_endpoint_and_two_exit_legs(self) -> None:
        client, session = self.make_client(
            [FakeResponse({"orderListId": 7, "listOrderStatus": "EXECUTING"})]
        )

        client.place_oco_sell_order(
            "BTC/USDT",
            Decimal("0.01"),
            Decimal("55000"),
            Decimal("48000"),
            "aq-list",
            "aq-take",
            "aq-stop",
        )

        call = session.calls[0]
        values = parse_qs(call["params"])
        self.assertTrue(call["url"].endswith("/api/v3/orderList/oco"))
        self.assertEqual(values["aboveType"], ["LIMIT_MAKER"])
        self.assertEqual(values["abovePrice"], ["55000"])
        self.assertEqual(values["belowType"], ["STOP_LOSS"])
        self.assertEqual(values["belowStopPrice"], ["48000"])


if __name__ == "__main__":
    unittest.main()
