"""交易圖文紀錄必須保留因果、帳務、冪等與故障隔離。"""

from types import SimpleNamespace
import json

import pandas as pd
import pytest

from ai_quant_trading.trading.journal import TradeJournal, journal_call


def setup_journal(tmp_path, side=1):
    journal = TradeJournal(tmp_path / "trade_journal")
    state = SimpleNamespace(symbol="BTC/USDT", interval="15m", exchange="binance_usdm", model_dir="model",
        pending_time="2026-01-01T00:00:00+00:00", pending_target_fraction=.1, pending_reason="rl_policy",
        risk_config=SimpleNamespace(stop_loss_mode="atr", max_risk_per_trade=.01),
        config=SimpleNamespace(fee_rate=.0005, slippage_rate=.0002),
        entry_time="2026-01-01T00:15:00+00:00", entry_price=100., quantity=side * 2,
        stop_loss=100 - side * 2, take_profit=100 + side * 4, position_leverage=1)
    bar = {"timestamp": pd.Timestamp(state.pending_time), "open": 100., "high": 101., "low": 99., "close": 100., "volume": 10.}
    journal.market(bar)
    journal.decision(state, bar, {"model_target_fraction": .1,
        "journal_input_snapshot": {"observation": [.1, .2], "feature_columns": ["return_1"], "oos_verified": False}})
    entry_bar = dict(bar, timestamp=pd.Timestamp(state.entry_time), high=10000.)
    journal.market(entry_bar)
    journal.opened(state, entry_bar)
    return journal, state, entry_bar


@pytest.mark.parametrize("side", [1, -1])
def test_entry_no_future_and_close_cost_r(tmp_path, side):
    journal, state, row = setup_journal(tmp_path, side)
    entry = journal.records()[0]["record"]
    assert max(c["high"] for c in entry["candles"]) == 101
    assert entry["entry"]["decision"]["known_at"] == state.entry_time
    kwargs = dict(quantity=2., entry_fee=.1, exit_fee=.1, carry=.05, gross=8., net=7.75,
                  reason="take_profit", fill_price=100+side*4)
    journal.closed(state, row, **kwargs)
    journal.closed(state, row, **kwargs)
    rows = journal.records()
    assert len(rows) == 2
    trade = next(r["record"] for r in rows if r["kind"] == "close")
    assert trade["outcome"]["planned_reward_risk"] == pytest.approx(2)
    assert trade["outcome"]["realized_net_r"] == pytest.approx(1.9375)
    assert trade["outcome"]["intrabar_time_unknown"]
    result = journal.export_training_review(tmp_path / "export")
    assert result == {"rows": 1, "excluded": 0}
    text = (tmp_path / "export/inputs.jsonl").read_text(encoding="utf-8")
    assert "net_pnl" not in text and "exit_price" not in text and "outcome" not in text
    labels = json.loads((tmp_path / "export/labels.jsonl").read_text(encoding="utf-8"))
    assert labels["position_group"] == json.loads(text)["position_group"]


def test_scaled_position_not_fake_initial_r(tmp_path):
    journal, state, row = setup_journal(tmp_path)
    journal.opened(state, row, added=True)
    journal.closed(state, row, quantity=1., entry_fee=.1, exit_fee=.1, carry=0., gross=1., net=.8,
                   reason="reduce", fill_price=101., partial=True)
    r = next(r for r in journal.records() if r["kind"] == "partial_close")
    assert r["record"]["outcome"]["realized_net_r"] is None
    assert journal.export_training_review(tmp_path / "export")["excluded"] == 1


def test_render_retry_and_original_json(tmp_path, monkeypatch):
    journal, _, _ = setup_journal(tmp_path)
    def fail(*args):
        raise OSError("disk failure")
    monkeypatch.setattr("ai_quant_trading.trading.journal.render_journal_png", fail)
    assert journal.render_pending() == 0
    row = journal.records()[0]
    assert row["status"] == "pending" and row["error"] == "OSError"
    assert (journal.root / (row["id"] + ".json")).is_file()
    monkeypatch.undo()
    assert journal.render_pending() == 1
    assert journal.render_pending() == 0
    assert (journal.root / (row["id"] + ".png")).read_bytes().startswith(b"\x89PNG")


def test_journal_failure_never_blocks_execution(tmp_path, monkeypatch):
    def fail(*args):
        raise PermissionError("sensitive content must not be logged")
    monkeypatch.setattr(TradeJournal, "market", fail)
    paths = SimpleNamespace(account_dir=tmp_path)
    assert journal_call(paths, "market", {}) is None
    text = (tmp_path / "trade_journal/last_error.json").read_text()
    assert "PermissionError" in text and "sensitive content" not in text


def test_future_snapshot_is_excluded(tmp_path):
    journal, state, row = setup_journal(tmp_path)
    with journal.connect() as db:
        db.execute("DELETE FROM entries")
        db.execute("DELETE FROM records")
    state.pending_time = "2026-01-01T00:15:00+00:00"
    journal.decision(state, row, {"journal_input_snapshot": {"observation": [1.], "feature_columns": ["return_1"]}})
    journal.opened(state, row)
    assert journal.records()[0]["record"]["entry"]["decision"] is None


def test_missing_holding_bars_are_explicit(tmp_path):
    journal, state, row = setup_journal(tmp_path)
    row["timestamp"] += pd.Timedelta(minutes=45)
    journal.market(row)
    journal.closed(state, row, quantity=2., entry_fee=0., exit_fee=0., carry=0., gross=0., net=0.,
                   reason="time", fill_price=100.)
    r = next(r["record"] for r in journal.records() if r["kind"] == "close")
    assert r["chart_missing_holding_bars"] == 2
    assert not r["chart_coverage_complete"]


def test_scale_in_is_a_separate_idempotent_record(tmp_path):
    journal, state, row = setup_journal(tmp_path)
    state.quantity = 3.
    for _ in range(2):
        journal.opened(state, row, added=True, fill_price=102., filled_quantity=1.)
    additions = [r for r in journal.records() if r["kind"] == "scale_in"]
    assert len(additions) == 1
    assert additions[0]["record"]["execution"]["fill_price"] == 102.


def test_concurrent_render_claim_is_not_duplicated(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    journal, _, _ = setup_journal(tmp_path)
    calls, guard = [], Lock()
    def renderer(record, target):
        with guard:
            calls.append(str(target))
        target.write_bytes(b"PNG-test-only")
    monkeypatch.setattr("ai_quant_trading.trading.journal.render_journal_png", renderer)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: journal.render_pending(), range(2)))
    assert sum(results) == 1 and len(calls) == 1
    assert journal.records()[0]["status"] == "complete"


def test_partial_closes_share_group_and_export_does_not_overwrite(tmp_path):
    journal, state, row = setup_journal(tmp_path)
    journal.closed(state, row, quantity=1., entry_fee=.05, exit_fee=.05, carry=0.,
                   gross=1., net=.9, reason="reduce", fill_price=101., partial=True)
    state.quantity = 1.
    row["timestamp"] += pd.Timedelta(minutes=15)
    journal.closed(state, row, quantity=1., entry_fee=.05, exit_fee=.05, carry=0.,
                   gross=2., net=1.9, reason="close", fill_price=102.)
    assert journal.export_training_review(tmp_path / "export")["rows"] == 2
    rows = [json.loads(r) for r in (tmp_path / "export/inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({r["position_group"] for r in rows}) == 1
    with pytest.raises(FileExistsError):
        journal.export_training_review(tmp_path / "export")
