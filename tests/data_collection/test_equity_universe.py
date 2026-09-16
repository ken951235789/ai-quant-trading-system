"""美股快速清單測試。"""

from __future__ import annotations

import unittest

from ai_quant_trading.data_collection.equity_universe import (
    US_EQUITY_PRESETS,
    preset_symbols_text,
)


class EquityUniverseTests(unittest.TestCase):
    def test_presets_are_uppercase_and_unique_within_each_group(self) -> None:
        self.assertGreaterEqual(len(US_EQUITY_PRESETS), 6)
        for symbols in US_EQUITY_PRESETS.values():
            self.assertEqual(len(symbols), len(set(symbols)))
            self.assertTrue(all(symbol == symbol.upper() for symbol in symbols))

    def test_preset_can_be_used_as_input_text(self) -> None:
        self.assertIn("AAPL", preset_symbols_text("大型科技"))
        self.assertIn("SPY", preset_symbols_text("長期專家"))
        self.assertIn("NVDA", preset_symbols_text("短期專家"))


if __name__ == "__main__":
    unittest.main()
