"""測試共用設定，避免測試程序誤寫正式交易資料庫。"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_live_storage_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """每個測試預設使用檔案儲存；個別資料庫測試可自行覆寫。"""
    monkeypatch.setenv("AI_QUANT_STORAGE_BACKEND", "file")
    monkeypatch.delenv("AI_QUANT_DATABASE_URL", raising=False)
