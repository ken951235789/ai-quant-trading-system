"""資料收集命令列介面測試。"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from ai_quant_trading.data_collection.cli import main
from ai_quant_trading.data_collection.yahoo_finance import YahooFinanceDataError


class DataCollectionCliTest(unittest.TestCase):
    @patch(
        "ai_quant_trading.data_collection.cli.collect_yahoo_finance_data",
        side_effect=YahooFinanceDataError("Yahoo 暫時限流"),
    )
    def test_yahoo_error_is_reported_without_traceback(self, _mock_collect):
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            exit_code = main(["--exchange", "yahoo", "--symbols", "AAPL"])

        self.assertEqual(exit_code, 1)
        self.assertIn("資料下載失敗：Yahoo 暫時限流", error_output.getvalue())
        self.assertNotIn("Traceback", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
