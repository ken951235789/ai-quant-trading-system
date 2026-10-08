"""故障點覆蓋：提交、CSV 部分寫入、帳戶狀態及日誌投影。"""

import json

import pandas as pd
import pytest

from ai_quant_trading.paper_trading import transactions as tx
from ai_quant_trading.paper_trading.storage import account_paths, append_csv_row
from ai_quant_trading.trading.journal import TradeJournal


def test_recover_committed_partial_csv_once(tmp_path, monkeypatch):
    paths = account_paths(tmp_path, "crash")
    paths.account_dir.mkdir(parents=True)
    original = tx._apply_file
    def interrupt(root, plan):
        import base64
        (root / plan["name"]).write_bytes(base64.b64decode(plan["data"])[:5])
        raise OSError("injected")
    monkeypatch.setattr(tx, "_apply_file", interrupt)
    with pytest.raises(OSError), tx.bar_transaction(paths, {"cash": 100}):
        append_csv_row(paths.orders_csv, ["a", "b"], {"a": 1, "b": 2})
    monkeypatch.setattr(tx, "_apply_file", original)
    assert tx.recover(paths) == 1
    assert tx.recover(paths) == 0
    assert len(pd.read_csv(paths.orders_csv)) == 1
    assert json.loads(paths.account_json.read_text())["cash"] == 100


def test_journal_failure_preserves_queue_and_does_not_block_account(tmp_path, monkeypatch):
    paths = account_paths(tmp_path, "journal")
    paths.account_dir.mkdir(parents=True)
    original = TradeJournal.market
    monkeypatch.setattr(TradeJournal, "market", lambda *a: (_ for _ in ()).throw(OSError("injected")))
    for cash in (100, 101):
        with tx.bar_transaction(paths, {"cash": cash}):
            tx.capture_journal("market", ({"timestamp": "2026-01-01T00:00:00+00:00", "open": 100, "high": 101, "low": 99, "close": 100},), {})
    assert json.loads(paths.account_json.read_text())["cash"] == 101
    with tx.database(paths.account_dir) as db:
        assert db.execute("SELECT applied FROM commits").fetchall() == [(1,), (1,)]
    monkeypatch.setattr(TradeJournal, "market", original)
    assert tx.recover(paths) == 2
    assert tx.recover(paths) == 0


def test_external_csv_change_fails_closed(tmp_path, monkeypatch):
    paths = account_paths(tmp_path, "conflict")
    paths.account_dir.mkdir(parents=True)
    original = tx.recover
    monkeypatch.setattr(tx, "recover", lambda p: None)
    with tx.bar_transaction(paths, {"cash": 100}):
        append_csv_row(paths.orders_csv, ["a"], {"a": 1})
    paths.orders_csv.write_bytes(b"x\r\n")
    monkeypatch.setattr(tx, "recover", original)
    with pytest.raises(ValueError, match="尾端"):
        tx.recover(paths)
    assert paths.orders_csv.read_bytes() == b"x\r\n"
