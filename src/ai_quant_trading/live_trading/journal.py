"""只投影已确认的交易所成交；不下單，也不以委託狀態冒充成交。"""

from datetime import datetime, timezone
import json
import logging

import pandas as pd

from ai_quant_trading.market_clock import interval_duration
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.trading.journal import TradeJournal, VERSION, _digest, _json, _number, _stamp


def project_confirmed_fill(journal, environment, payload, *, interval="15m"):
    if payload.get("e") != "ORDER_TRADE_UPDATE":
        return False
    order = payload.get("o", {})
    if order.get("x") != "TRADE":
        return False
    quantity, price = _number(order.get("l")), _number(order.get("L"))
    trade_id = str(order.get("t", ""))
    if not quantity or quantity <= 0 or not price or price <= 0 or trade_id in {"", "0", "-1"}:
        raise ValueError("確認成交缺少有效價格、數量或成交識別")
    if order.get("S") not in {"BUY", "SELL"} or order.get("s") != "BTCUSDT":
        return False
    stamp = pd.to_datetime(order.get("T", payload.get("T")), unit="ms", utc=True)
    if pd.isna(stamp):
        raise ValueError("成交缺少交易所時間")
    record_id = _digest(["binance_confirmed_fill_v1", environment, order["s"], trade_id])
    identity = {"symbol": order["s"], "trade_id": trade_id, "at": _stamp(stamp),
        "side": order["S"], "position_side": order.get("ps", "BOTH"),
        "price": price, "quantity": quantity, "commission": _number(order.get("n")),
        "commission_asset": order.get("N"), "realized_gross": _number(order.get("rp")),
        "reduce_only": order.get("R") is True}
    with journal.connect() as db:
        prior = db.execute("SELECT payload FROM records WHERE id=?", (record_id,)).fetchone()
        if prior:
            if json.loads(prior[0]).get("confirmed_fill") != identity:
                raise ValueError("同一交易所成交識別的內容互相矛盾")
            return False
        # 成交當下只取已收盤資料；歷史補登缺少行情時明確顯示空白範圍。
        close_cutoff = _stamp(stamp - interval_duration(interval))
        candles = [json.loads(row[0]) for row in db.execute(
            "SELECT payload FROM bars WHERE timestamp<=? ORDER BY timestamp DESC LIMIT 96", (close_cutoff,))][::-1]
        entry = {"schema": VERSION, "environment": environment, "position_id": record_id,
            "symbol": "BTC/USDT", "interval": interval, "side": 1 if order["S"] == "BUY" else -1,
            "entry_at": _stamp(stamp), "entry_price": price, "entry_quantity": quantity,
            "initial_stop": None, "initial_target": None, "decision": None,
            "scaled_in": False, "model_input_complete": False, "input_candles": candles}
        record = {"schema": VERSION, "entry": entry, "outcome": None, "candles": candles,
            "confirmed_fill": identity, "chart_coverage_complete": bool(candles),
            "training_use": "confirmed_fill_only_not_a_complete_trade_label",
            "limitations": ["BUY/SELL 不等同開倉/平倉；未推定未知初始部位", "未分攤開倉成本、跨幣手續費與資金費率", "尚未證明完整往返交易及初始保護單，不提供虛構盈虧比"]}
        db.execute("INSERT INTO records(id,position_id,kind,timestamp,payload) VALUES (?,?,?,?,?)",
            (record_id, record_id, "confirmed_fill", _stamp(stamp), _json(record)))
    return True


def sync_confirmed_fills(repository, paths, environment, *, limit=200):
    """有界循環掃描耐久事件；回到起點可補到延遲提交的較小 ID，投影本身冪等。"""
    journal = TradeJournal(paths.environment_dir / "trade_journal")
    try:
        with journal.connect() as db:
            row = db.execute("SELECT payload FROM context WHERE key='exchange_scan'").fetchone()
        cursor = int(json.loads(row[0])) if row else 0
        events = repository.journal_exchange_events(environment, after_id=cursor, limit=limit)
        inserted = 0
        for event in events:
            inserted += project_confirmed_fill(journal, environment, event["payload"])
            cursor = event["id"]
        with journal.connect() as db:
            db.execute("INSERT OR REPLACE INTO context VALUES ('exchange_scan',?)", (_json(cursor if len(events) == limit else 0),))
        journal.render_pending(limit=4)
        write_json_atomic(journal.root / "sync_status.json", {"ok": True, "at": datetime.now(timezone.utc).isoformat(), "inserted": inserted})
        return inserted
    except Exception as error:
        logging.getLogger(__name__).warning("實盤成交日誌等待重播：%s", type(error).__name__)
        try:
            write_json_atomic(journal.root / "sync_status.json", {"ok": False, "error": type(error).__name__})
        except OSError:
            pass
        return 0


def cache_live_market(paths, frame):
    """供圖像使用的行情快取；不呼叫交易 API。"""
    try:
        journal = TradeJournal(paths.environment_dir / "trade_journal")
        for row in frame.tail(96).to_dict("records"):
            journal.market(row)
    except Exception as error:
        logging.getLogger(__name__).warning("實盤日誌行情快取失敗：%s", type(error).__name__)
