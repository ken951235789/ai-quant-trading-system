"""資產名稱與標準代碼測試。"""

from __future__ import annotations

import unittest

from ai_quant_trading.data_collection.assets import (
    CRYPTO_PRESETS,
    US_EQUITY_PRESETS,
    asset_display_name,
    infer_asset_class,
    market_file_symbol_slug,
    normalize_crypto_symbol,
)


class AssetCatalogTests(unittest.TestCase):
    def test_crypto_symbol_is_normalized(self) -> None:
        self.assertEqual(normalize_crypto_symbol(" btcusdt "), "BTC/USDT")
        self.assertEqual(normalize_crypto_symbol("eth-usdt"), "ETH/USDT")

    def test_asset_class_uses_exchange(self) -> None:
        self.assertEqual(infer_asset_class("binance", "BTCUSDT"), "crypto")
        self.assertEqual(infer_asset_class("yahoo_finance", "AAPL"), "us_equity")

    def test_display_name_contains_code_and_chinese_name(self) -> None:
        self.assertIn("比特幣", asset_display_name("BTC/USDT", "crypto"))
        self.assertIn("蘋果", asset_display_name("AAPL", "us_equity"))

    def test_file_slug_keeps_crypto_pair_readable(self) -> None:
        self.assertEqual(market_file_symbol_slug("BTC/USDT", "crypto"), "BTC-USDT")

    def test_expert_presets_contain_recommended_markets(self) -> None:
        self.assertEqual(
            set(CRYPTO_PRESETS["長期專家"]),
            {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "LINK/USDT"},
        )
        self.assertIn("DOGE/USDT", CRYPTO_PRESETS["短期專家"])
        self.assertIn("BRK-B", US_EQUITY_PRESETS["長期專家"])
        self.assertIn("COIN", US_EQUITY_PRESETS["短期專家"])

    def test_new_assets_have_chinese_display_names(self) -> None:
        self.assertIn("埃克森美孚", asset_display_name("XOM", "us_equity"))
        self.assertIn("波克夏", asset_display_name("BRK-B", "us_equity"))


if __name__ == "__main__":
    unittest.main()
