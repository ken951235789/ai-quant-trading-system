"""交易所事件投影只接受確認成交，重播與乱序不捏造持倉。"""

from types import SimpleNamespace
import json

import pytest

from ai_quant_trading.live_trading.journal import project_confirmed_fill, sync_confirmed_fills
from ai_quant_trading.trading.journal import TradeJournal


def event(trade=1):
    return {"e": "ORDER_TRADE_UPDATE", "T": 1767225600000 + trade*1000,
        "o": {"x": "TRADE", "s": "BTCUSDT", "t": trade, "l": ".01", "L": "90000",
              "S": "BUY", "ps": "BOTH", "n": ".45", "N": "USDT", "rp": "0", "R": False}}


def test_confirmed_fill_duplicate_order_status_and_no_fake_entry(tmp_path):
    journal = TradeJournal(tmp_path)
    update = event()
    assert project_confirmed_fill(journal, "testnet", update)
    assert not project_confirmed_fill(journal, "testnet", update)
    update["o"]["x"] = "NEW"
    assert not project_confirmed_fill(journal, "testnet", update)
    record = journal.records()[0]
    assert record["kind"] == "confirmed_fill" and record["record"]["outcome"] is None
    assert record["record"]["entry"]["initial_stop"] is None
    assert journal.render_pending() == 1


def test_changed_duplicate_rejected_and_environment_separated(tmp_path):
    journal = TradeJournal(tmp_path)
    project_confirmed_fill(journal, "testnet", event())
    assert project_confirmed_fill(journal, "live", event())
    changed = event()
    changed["o"]["l"] = ".02"
    with pytest.raises(ValueError, match="矛盾"):
        project_confirmed_fill(journal, "live", changed)


def test_sync_revisits_earlier_ids_and_redacts_error(tmp_path, monkeypatch):
    class Repository:
        rows = [{"id": 2, "payload": event(2)}]
        def journal_exchange_events(self, env, *, after_id, limit):
            return [r for r in self.rows if r["id"] > after_id][:limit]
    monkeypatch.setattr(TradeJournal, "render_pending", lambda *a, **k: 0)
    repo, paths = Repository(), SimpleNamespace(environment_dir=tmp_path)
    assert sync_confirmed_fills(repo, paths, "testnet", limit=1) == 1
    repo.rows.insert(0, {"id": 1, "payload": event(1)})
    assert sync_confirmed_fills(repo, paths, "testnet", limit=1) == 0
    assert sync_confirmed_fills(repo, paths, "testnet", limit=1) == 1
    assert len(TradeJournal(tmp_path / "trade_journal").records()) == 2
    repo.rows[1]["payload"]["o"]["l"] = "invalid"
    sync_confirmed_fills(repo, paths, "testnet", limit=1)
    status = json.loads((tmp_path / "trade_journal/sync_status.json").read_text())
    assert status == {"ok": False, "error": "ValueError"}


def test_repository_journal_read_only_projection_allowlist(tmp_path):
    from sqlalchemy import create_engine
    from ai_quant_trading.database.models import Base
    from ai_quant_trading.database.repository import PostgresTradingRepository
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    repo = PostgresTradingRepository(engine)
    payload = event()
    payload["o"]["c"] = "private-client-identifier"
    payload["unrelated_account_field"] = "not-for-journal"
    repo.append_exchange_event("testnet", "usd_m_user_data", payload)
    repo.append_exchange_event("live", "usd_m_user_data", event(2))
    rows = repo.journal_exchange_events("testnet")
    assert len(rows) == 1 and rows[0]["payload"]["o"]["t"] == 1
    assert "private-client" not in json.dumps(rows)
    assert "unrelated_account" not in json.dumps(rows)
    assert repo.journal_exchange_events("testnet", after_id=rows[0]["id"]) == []
