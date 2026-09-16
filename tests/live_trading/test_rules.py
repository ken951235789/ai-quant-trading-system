"""Binance 下單數量與名目金額規則測試。"""

from __future__ import annotations

from decimal import Decimal
import unittest

from ai_quant_trading.live_trading.rules import SymbolRules


def make_rules() -> SymbolRules:
    return SymbolRules.from_exchange_info(
        {
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
    )


class SymbolRulesTests(unittest.TestCase):
    def test_quantity_is_rounded_down_to_step(self) -> None:
        rules = make_rules()
        self.assertEqual(rules.normalize_quantity(Decimal("0.001239")), Decimal("0.00123"))

    def test_minimum_notional_is_enforced(self) -> None:
        rules = make_rules()
        with self.assertRaisesRegex(ValueError, "最小金額"):
            rules.validate(Decimal("0.00001"), Decimal("50000"))


if __name__ == "__main__":
    unittest.main()
