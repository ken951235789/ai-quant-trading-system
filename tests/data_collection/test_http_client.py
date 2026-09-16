"""市場資料 HTTP Session 的 SSL 隔離測試。"""

from __future__ import annotations

import unittest
import sys

from ai_quant_trading.data_collection.http_client import (
    NativeTruststoreAdapter,
    create_secure_session,
)


class HttpClientTests(unittest.TestCase):
    def test_windows_session_uses_local_truststore_adapter(self) -> None:
        session = create_secure_session()

        self.assertEqual(session.adapters["https://"].max_retries.total, 3)

        if sys.platform == "win32":
            self.assertIsInstance(session.adapters["https://"], NativeTruststoreAdapter)
        else:
            self.assertNotIsInstance(session.adapters["https://"], NativeTruststoreAdapter)


if __name__ == "__main__":
    unittest.main()
