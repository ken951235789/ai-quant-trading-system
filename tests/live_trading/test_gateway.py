"""實盤安全閘門、餘額限制與送單開關測試。"""

from __future__ import annotations

from decimal import Decimal
import unittest

from ai_quant_trading.live_trading.binance_client import (
    BinanceExecutionUnknown,
    BinancePrivateApiError,
)
from ai_quant_trading.live_trading.config import LiveTradingConfig
from ai_quant_trading.live_trading.gateway import LiveTradingGateway, make_client_order_id


class FakePrivateClient:
    def __init__(self):
        self.placed = []
        self.tested = []
        self.raise_unknown = False
        self.order_exists = False

    def sync_server_time(self):
        return 0

    def fetch_account(self):
        return {
            "canTrade": True,
            "balances": [
                {"asset": "BTC", "free": "0.01", "locked": "0"},
                {"asset": "USDT", "free": "1000", "locked": "0"},
            ],
        }

    def fetch_exchange_info(self, symbol):
        return {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "filters": [
                {
                    "filterType": "LOT_SIZE",
                    "minQty": "0.00001",
                    "maxQty": "10",
                    "stepSize": "0.00001",
                },
                {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
            ],
        }

    def fetch_price(self, symbol):
        return Decimal("50000")

    def test_market_order(self, symbol, side, quantity, client_order_id):
        self.tested.append((symbol, side, quantity, client_order_id))
        return {}

    def place_market_order(self, symbol, side, quantity, client_order_id):
        self.placed.append((symbol, side, quantity, client_order_id))
        if self.raise_unknown:
            raise BinanceExecutionUnknown("unknown", client_order_id)
        return {"status": "FILLED", "clientOrderId": client_order_id}

    def query_order(self, symbol, client_order_id):
        if (self.raise_unknown and self.placed) or self.order_exists:
            return {"status": "FILLED", "clientOrderId": client_order_id}
        raise BinancePrivateApiError("order not found", code=-2013)


class LiveTradingGatewayTests(unittest.TestCase):
    def test_seeded_client_order_id_is_deterministic(self) -> None:
        first = make_client_order_id("BTC/USDT", "BUY", "same-cycle")
        second = make_client_order_id("BTC/USDT", "BUY", "same-cycle")
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 36)

    def test_buy_preview_respects_quote_and_balance_limits(self) -> None:
        gateway = LiveTradingGateway(
            FakePrivateClient(),
            LiveTradingConfig(max_order_quote=100, max_balance_fraction=0.25),
        )

        preview = gateway.preview_market_order("BTCUSDT", "BUY", quote_amount=500)

        self.assertEqual(preview.quantity, Decimal("0.002"))
        self.assertEqual(preview.estimated_notional, Decimal("100.000"))

    def test_symbol_outside_allowlist_is_rejected_before_api_use(self) -> None:
        gateway = LiveTradingGateway(FakePrivateClient(), LiveTradingConfig())
        with self.assertRaisesRegex(ValueError, "白名單"):
            gateway.preview_market_order("SOL/USDT", "BUY", quote_amount=20)

    def test_validation_does_not_require_trading_enabled(self) -> None:
        client = FakePrivateClient()
        gateway = LiveTradingGateway(client, LiveTradingConfig(trading_enabled=False))
        preview = gateway.preview_market_order("BTC/USDT", "BUY", quote_amount=20)

        result = gateway.validate_order(preview, "aq-test")

        self.assertTrue(result.validated_only)
        self.assertEqual(len(client.tested), 1)
        self.assertEqual(len(client.placed), 0)

    def test_actual_order_requires_switch_and_confirmation(self) -> None:
        client = FakePrivateClient()
        disabled = LiveTradingGateway(client, LiveTradingConfig(trading_enabled=False))
        preview = disabled.preview_market_order("BTC/USDT", "BUY", quote_amount=20)
        with self.assertRaises(PermissionError):
            disabled.submit_order(preview, "EXECUTE TESTNET")

        enabled = LiveTradingGateway(client, LiveTradingConfig(trading_enabled=True))
        with self.assertRaises(PermissionError):
            enabled.submit_order(preview, "wrong")
        result = enabled.submit_order(preview, "EXECUTE TESTNET", "aq-live")
        self.assertEqual(result.response["status"], "FILLED")
        self.assertEqual(len(client.placed), 1)

    def test_existing_client_order_id_is_reconciled_without_resubmission(self) -> None:
        client = FakePrivateClient()
        client.order_exists = True
        gateway = LiveTradingGateway(client, LiveTradingConfig(trading_enabled=True))
        preview = gateway.preview_market_order("BTC/USDT", "BUY", quote_amount=20)

        result = gateway.submit_order(preview, "EXECUTE TESTNET", "aq-existing")

        self.assertTrue(result.recovered_after_unknown)
        self.assertEqual(len(client.placed), 0)

    def test_unknown_order_is_queried_without_resubmission(self) -> None:
        client = FakePrivateClient()
        client.raise_unknown = True
        gateway = LiveTradingGateway(client, LiveTradingConfig(trading_enabled=True))
        preview = gateway.preview_market_order("BTC/USDT", "BUY", quote_amount=20)

        result = gateway.submit_order(preview, "EXECUTE TESTNET", "aq-unknown")

        self.assertTrue(result.recovered_after_unknown)
        self.assertEqual(len(client.placed), 1)


if __name__ == "__main__":
    unittest.main()
