"""本機唯讀用途的成交日誌：決策與事後標籤隔離，PNG 失敗可重試。"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from threading import Lock

import pandas as pd

from ai_quant_trading.market_clock import interval_duration
from ai_quant_trading.persistence import write_json_atomic

LOGGER = logging.getLogger(__name__)
PLOT_LOCK = Lock()
VERSION = "trade_journal_v1"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _stamp(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("日誌時間必須明確包含時區")
    return timestamp.tz_convert("UTC").isoformat()


class TradeJournal:
    """SQLite 僅存觀察副本，不參與下單、資金計算或風控裁決。"""

    def __init__(self, root):
        self.root = Path(root)

    @contextmanager
    def connect(self):
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.root / "journal.sqlite3", timeout=.25)
        try:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS context (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS bars (timestamp TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY, position_id TEXT NOT NULL, kind TEXT NOT NULL,
                    timestamp TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    claimed_at REAL, error TEXT);
            """)
            with connection:
                yield connection
        finally:
            connection.close()

    def market(self, row):
        candle = {k: _number(row.get(k)) for k in ("open", "high", "low", "close", "volume")}
        if any(candle[k] is None or candle[k] <= 0 for k in ("open", "high", "low", "close")):
            raise ValueError("日誌缺少有效 OHLC")
        if candle["high"] < max(candle[k] for k in ("open", "close", "low")) or candle["low"] > min(candle[k] for k in ("open", "close", "high")):
            raise ValueError("日誌 OHLC 順序錯誤")
        candle["timestamp"] = _stamp(row["timestamp"])
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO bars VALUES (?, ?)", (candle["timestamp"], _json(candle)))
            # 只保留本功能的行情快取；每筆進場與平倉已另存不可變快照。
            db.execute("DELETE FROM bars WHERE timestamp NOT IN (SELECT timestamp FROM bars ORDER BY timestamp DESC LIMIT 2048)")

    def decision(self, state, row, prediction):
        stamp = _stamp(row["timestamp"])
        snapshot = prediction.get("journal_input_snapshot")
        if not isinstance(snapshot, dict):
            snapshot = None
        if snapshot is not None:
            values = snapshot.get("observation", [])
            if not values or any(_number(x) is None for x in values):
                raise ValueError("日誌 observation 不合法")
            if any(any(token in name.lower() for token in ("future_", "actual_", "label_"))
                   for name in snapshot.get("feature_columns", [])):
                raise ValueError("日誌拒絕疑似未來特徵")
        value = {"schema": VERSION, "bar_open_at": stamp,
            "known_at": _stamp(pd.Timestamp(stamp) + interval_duration(state.interval)),
            "symbol": state.symbol, "interval": state.interval, "exchange": state.exchange,
            "model_id": Path(state.model_dir).name, "model_input": snapshot,
            "model_target": _number(prediction.get("model_target_fraction")),
            "approved_target": _number(state.pending_target_fraction),
            "candidate_handoff": snapshot.get("candidate_handoff") if snapshot else None,
            "reason": state.pending_reason,
            "risk": {"stop_mode": state.risk_config.stop_loss_mode,
                     "max_risk_per_trade": state.risk_config.max_risk_per_trade,
                     "fee_rate": state.config.fee_rate, "slippage_rate": state.config.slippage_rate}}
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO context VALUES ('decision', ?)", (_json(value),))

    @staticmethod
    def position_id(state):
        return _digest([VERSION, state.symbol, state.entry_time, 1 if state.quantity > 0 else -1])

    def opened(self, state, row, *, added=False, fill_price=None, filled_quantity=None):
        position_id = self.position_id(state)
        stamp = _stamp(row["timestamp"])
        with self.connect() as db:
            old = db.execute("SELECT payload FROM entries WHERE id=?", (position_id,)).fetchone()
            if old:
                if added:
                    payload = json.loads(old[0])
                    payload["scaled_in"] = True
                    db.execute("UPDATE entries SET payload=? WHERE id=?", (_json(payload), position_id))
                    candles = [json.loads(r[0]) for r in db.execute(
                        "SELECT payload FROM bars WHERE timestamp < ? ORDER BY timestamp DESC LIMIT 96", (stamp,))][::-1]
                    # 加碼保留當次成交價，不把持倉平均成本誤當成交價。
                    record = {"schema": VERSION, "entry": payload, "outcome": None, "candles": candles,
                        "execution": {"kind": "scale_in", "at": stamp, "fill_price": _number(fill_price),
                                      "quantity": _number(filled_quantity), "position_quantity": abs(state.quantity)},
                        "training_use": "scaled_position_excluded", "chart_scope": "closed_bars_before_addition"}
                    record_id = _digest([position_id, record["execution"]])
                    db.execute("INSERT OR IGNORE INTO records (id,position_id,kind,timestamp,payload) VALUES (?,?,?,?,?)",
                        (record_id, position_id, "scale_in", stamp, _json(record)))
                return
            result = db.execute("SELECT payload FROM context WHERE key='decision'").fetchone()
            decision = json.loads(result[0]) if result else None
            if decision and (pd.Timestamp(decision["known_at"]) > pd.Timestamp(stamp)
                             or decision["bar_open_at"] != _stamp(state.pending_time)):
                decision = None
            candles = [json.loads(r[0]) for r in db.execute(
                "SELECT payload FROM bars WHERE timestamp < ? ORDER BY timestamp DESC LIMIT 96", (stamp,))][::-1]
            entry = {"schema": VERSION, "environment": "paper", "position_id": position_id,
                "symbol": state.symbol, "interval": state.interval, "side": 1 if state.quantity > 0 else -1,
                "entry_at": stamp, "entry_price": state.entry_price, "entry_quantity": abs(state.quantity),
                "initial_stop": state.stop_loss, "initial_target": state.take_profit,
                "leverage": state.position_leverage, "decision": decision, "scaled_in": bool(added),
                "input_candles": candles, "model_input_complete": bool(decision and decision["model_input"])}
            db.execute("INSERT INTO entries VALUES (?, ?)", (position_id, _json(entry)))
            record = {"schema": VERSION, "entry": entry, "outcome": None, "candles": candles,
                      "training_use": "entry_snapshot_only; no_outcome_yet", "chart_scope": "closed_bars_before_entry"}
            db.execute("INSERT OR IGNORE INTO records (id,position_id,kind,timestamp,payload) VALUES (?,?,?,?,?)",
                       (position_id, position_id, "entry", stamp, _json(record)))

    def closed(self, state, row, *, quantity, entry_fee, exit_fee, carry, gross, net, reason, fill_price, partial=False):
        position_id = self.position_id(state)
        stamp = _stamp(row["timestamp"])
        with self.connect() as db:
            result = db.execute("SELECT payload FROM entries WHERE id=?", (position_id,)).fetchone()
            entry = json.loads(result[0]) if result else {
                "schema": VERSION, "environment": "paper", "position_id": position_id, "symbol": state.symbol,
                "interval": state.interval, "side": 1 if state.quantity > 0 else -1,
                "entry_at": state.entry_time, "entry_price": state.entry_price,
                "initial_stop": None, "initial_target": None, "decision": None,
                "scaled_in": False, "input_candles": [], "model_input_complete": False}
            expected = gross - entry_fee - exit_fee - carry
            if not math.isclose(net, expected, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError("日誌與交易成本帳務不一致")
            initial_risk = (abs(entry["entry_price"] - entry["initial_stop"]) * quantity
                            if entry["initial_stop"] is not None and not entry["scaled_in"] else None)
            rr = (abs(entry["initial_target"] - entry["entry_price"]) / abs(entry["entry_price"] - entry["initial_stop"])
                  if initial_risk and entry["initial_target"] is not None else None)
            outcome = {"exit_bar_at": stamp, "exit_price": fill_price, "quantity": quantity,
                "entry_price_basis": state.entry_price, "gross_fill_pnl": gross, "entry_fee": entry_fee,
                "exit_fee": exit_fee, "carry_cost": carry, "net_pnl": net,
                "net_return_on_entry_notional": net / (state.entry_price * quantity),
                "planned_reward_risk": rr, "realized_net_r": net / initial_risk if initial_risk else None,
                "exit_reason": reason, "partial_close": partial,
                "intrabar_time_unknown": reason in {"stop_loss", "take_profit", "liquidation"},
                "label_available_at": _stamp(pd.Timestamp(stamp) + interval_duration(state.interval)),
                "cost_scope": "fill_prices_include_slippage; carry_is_simulated; tax_unverified"}
            candle_map = {r["timestamp"]: r for r in entry["input_candles"]}
            for r in db.execute("SELECT payload FROM bars WHERE timestamp>=? AND timestamp<=? ORDER BY timestamp",
                                (entry["entry_at"], stamp)):
                candle = json.loads(r[0])
                candle_map[candle["timestamp"]] = candle
            expected_times = pd.date_range(entry["entry_at"], stamp, freq=interval_duration(entry["interval"]))
            missing_bars = sum(t.isoformat() not in candle_map for t in expected_times)
            record = {"schema": VERSION, "entry": entry, "outcome": outcome,
                "candles": sorted(candle_map.values(), key=lambda r: r["timestamp"]),
                "chart_scope": "diagnostic_only_contains_outcome",
                "chart_missing_holding_bars": missing_bars,
                "chart_coverage_complete": missing_bars == 0,
                "training_use": "requires_offline_review_not_auto_training",
                "selection_bias": "executed_trades_only; no_counterfactual_or_rejected_candidates"}
            record_id = _digest([position_id, outcome])
            db.execute("INSERT OR IGNORE INTO records (id,position_id,kind,timestamp,payload) VALUES (?,?,?,?,?)",
                       (record_id, position_id, "partial_close" if partial else "close", stamp, _json(record)))

    def records(self, limit=200):
        if not (self.root / "journal.sqlite3").exists():
            return []
        with self.connect() as db:
            rows = db.execute("SELECT id,kind,timestamp,status,payload,error FROM records ORDER BY timestamp DESC,id LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "kind": r[1], "timestamp": r[2], "status": r[3], "record": json.loads(r[4]), "error": r[5]} for r in rows]

    def render_pending(self, limit=4):
        """交易帳務完成後才繪圖；短交易只產圖一次，失敗保留資料供下輪重試。"""
        count = 0
        for _ in range(limit):
            now = datetime.now(timezone.utc).timestamp()
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                item = db.execute("SELECT id,payload FROM records WHERE status='pending' OR (status='rendering' AND claimed_at<?) ORDER BY timestamp LIMIT 1", (now - 300,)).fetchone()
                if not item:
                    break
                db.execute("UPDATE records SET status='rendering',claimed_at=? WHERE id=?", (now, item[0]))
            record_id, payload = item[0], json.loads(item[1])
            try:
                write_json_atomic(self.root / f"{record_id}.json", payload)
                render_journal_png(payload, self.root / f"{record_id}.png")
                with self.connect() as db:
                    db.execute("UPDATE records SET status='complete',error=NULL WHERE id=?", (record_id,))
                count += 1
            except Exception as error:
                with self.connect() as db:
                    db.execute("UPDATE records SET status='pending',error=? WHERE id=?", (type(error).__name__, record_id))
                LOGGER.warning("交易日誌繪圖待重試：%s", type(error).__name__)
                break
        return count

    def export_training_review(self, output):
        """輸入與標籤分檔，保留群組、時間及來源，禁止把圖檔直接當現行模型輸入。"""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        inputs, labels, excluded = [], [], []
        for item in self.records(limit=1_000_000):
            r = item["record"]
            if r["outcome"] is None:
                continue
            entry, outcome = r["entry"], r["outcome"]
            decision = entry["decision"]
            if not decision or not entry["model_input_complete"] or entry["scaled_in"]:
                excluded.append({"id": item["id"], "reason": "missing_original_input_or_scaled_position"})
                continue
            if pd.Timestamp(decision["known_at"]) > pd.Timestamp(entry["entry_at"]):
                raise ValueError("禁止匯出晚於進場的模型輸入")
            inputs.append({"id": item["id"], "position_group": entry["position_id"], "decision": decision})
            labels.append({"id": item["id"], "position_group": entry["position_id"], "outcome": outcome})
        for name, values in (("inputs.jsonl", inputs), ("labels.jsonl", labels), ("excluded.jsonl", excluded)):
            with (output / name).open("x", encoding="utf-8") as stream:
                for value in values:
                    stream.write(_json(value) + "\n")
        write_json_atomic(output / "manifest.json", {"schema": VERSION, "rows": len(inputs), "excluded": len(excluded),
            "auto_train": False, "live_eligible": False, "policy": "group_by_position_then_time_purge_label_end",
            "limitations": ["成交選擇偏誤", "部分平倉共享同一群組", "圖片含未來結果不可作決策輸入",
                            "保存的是 RL 正規化 observation，非 Transformer 完整序列；需另外以凍結來源建立序列",
                            "仍需確認 checkpoint、正規化器與合法 OOS 證明，不能自動回灌 SAC"],
            "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                       for name in ("inputs.jsonl", "labels.jsonl", "excluded.jsonl")}})
        return {"rows": len(inputs), "excluded": len(excluded)}


def journal_call(paths, method, *args, **kwargs):
    """副本失敗不能阻斷減倉；只保存錯誤型別，不輸出交易憑證或任意物件。"""
    root = paths.account_dir / "trade_journal"
    try:
        from ai_quant_trading.paper_trading.transactions import capture_journal
        if capture_journal(method, args, kwargs):
            return None
        return getattr(TradeJournal(root), method)(*args, **kwargs)
    except Exception as error:
        LOGGER.warning("交易日誌 %s 失敗：%s", method, type(error).__name__)
        try:
            write_json_atomic(root / "last_error.json", {"operation": method, "error": type(error).__name__,
                "at": datetime.now(timezone.utc).isoformat(), "trading_blocked": False})
        except OSError:
            pass
        return None


def render_journal_png(record, target):
    """圖片只作診斷；進場圖片完全不包含進場後 K 棒。"""
    with PLOT_LOCK:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import Rectangle

        font = Path("C:/Windows/Fonts/msjh.ttc")
        entry, outcome = record["entry"], record["outcome"]
        style = {"axes.unicode_minus": False}
        if font.exists():
            style["font.family"] = FontProperties(fname=str(font)).get_name()
        with plt.rc_context(style):
            fig, ax = plt.subplots(figsize=(12, 6))
            try:
                all_candles = record["candles"]
                # 長持倉保留進場前後及最近行情，中間省略處明確畫出，避免進場點被裁掉。
                candles = all_candles if len(all_candles) <= 512 else all_candles[:128] + all_candles[-384:]
                times = [pd.Timestamp(c["timestamp"]) for c in candles]
                for i, c in enumerate(candles):
                    color = "#157a60" if c["close"] >= c["open"] else "#bd3549"
                    ax.vlines(i, c["low"], c["high"], color=color, lw=.8)
                    ax.add_patch(Rectangle((i-.3, min(c["open"], c["close"])), .6,
                        max(abs(c["close"]-c["open"]), .01), facecolor=color, edgecolor=color))
                def bar_index(stamp):
                    target_time = pd.Timestamp(stamp)
                    for i, t in enumerate(times):
                        if t == target_time:
                            return i
                        if t > target_time:
                            return i - .5
                    return len(times)
                for i in range(1, len(times)):
                    if times[i] - times[i - 1] > interval_duration(entry["interval"]):
                        ax.axvline(i - .5, color="#666666", ls=":", lw=1)
                x = bar_index(entry["entry_at"])
                y = entry["entry_price"]
                confirmed = record.get("confirmed_fill")
                ax.scatter([x], [y], marker="^" if entry["side"] > 0 else "v", s=100, color="#1976ad", zorder=4)
                point_label = f"{confirmed['side']} 成交" if confirmed else f"{'多' if entry['side'] > 0 else '空'} 進場"
                ax.annotate(f"{point_label} {y:,.2f}", (x, y), xytext=(-90, 30),
                            textcoords="offset points", arrowprops={"arrowstyle": "->"}, bbox={"facecolor": "white", "alpha": .9})
                for key, label, color in (("initial_stop", "初始停損", "#bd3549"), ("initial_target", "初始停利", "#157a60")):
                    if entry[key] is not None:
                        ax.axhline(entry[key], color=color, ls="--", lw=.8, label=f"{label} {entry[key]:,.2f}")
                title = f"{entry['symbol']} / {entry['interval']} / 模擬交易 / {'部分平倉' if outcome and outcome['partial_close'] else '平倉' if outcome else '進場快照'}"
                execution = record.get("execution")
                if execution and execution["fill_price"] is not None:
                    add_x, add_y = bar_index(execution["at"]), execution["fill_price"]
                    ax.scatter([add_x], [add_y], marker="+", s=110, color="#1976ad", zorder=5)
                    ax.annotate(f"加碼 {add_y:,.2f}", (add_x, add_y), xytext=(15, -25), textcoords="offset points")
                    title = title.replace("進場快照", "加碼快照")
                if record.get("chart_missing_holding_bars", 0) or len(record["candles"]) > 512:
                    title += " / 行情僅部分顯示"
                if outcome:
                    exit_x, exit_y = bar_index(outcome["exit_bar_at"]), outcome["exit_price"]
                    ax.scatter([exit_x], [exit_y], marker="x", s=90, color="#bd3549", zorder=5)
                    ax.annotate(f"出場 {exit_y:,.2f}", (exit_x, exit_y), xytext=(20, -35), textcoords="offset points",
                                arrowprops={"arrowstyle": "->"}, bbox={"facecolor": "white", "alpha": .9})
                    rr = outcome["planned_reward_risk"]
                    net_r = outcome["realized_net_r"]
                    detail = f"預定報酬:風險 {f'{rr:.2f}:1' if rr is not None else '未定義'} | 實現 {f'{net_r:+.3f}R' if net_r is not None else 'R 未定義'} | 淨損益 {outcome['net_pnl']:+.4f} | 淨報酬 {outcome['net_return_on_entry_notional']:+.3%}"
                else:
                    detail = "只顯示進場前已收盤行情；尚無出場與實現損益"
                if confirmed:
                    title = f"{entry['symbol']} / {entry['environment']} / 交易所確認成交"
                    detail = f"{confirmed['side']} {confirmed['quantity']} | 手續費 {confirmed['commission']} {confirmed['commission_asset']} | 非完整往返交易，R 與淨損益待對帳"
                ax.set_title(title + "\n" + detail, loc="left", fontsize=12)
                ticks = list(range(0, len(times), max(1, len(times)//7)))
                ax.set_xticks(ticks, [times[i].tz_convert("Asia/Taipei").strftime("%m/%d\n%H:%M") for i in ticks])
                ax.set_xlim(-1, max(len(times), x)+12)
                ax.set_ylabel("價格")
                ax.grid(alpha=.15)
                if entry["initial_stop"] is not None or entry["initial_target"] is not None:
                    ax.legend(loc="best", fontsize=9)
                fig.text(.08, .025, "台北 UTC+8。平倉圖不可作進場特徵；觸價棒內先後未知；灰色直線代表省略或缺漏的 K 棒。", fontsize=9)
                fig.tight_layout(rect=(0, .07, 1, 1))
                descriptor, temporary = tempfile.mkstemp(dir=target.parent, suffix=".png")
                os.close(descriptor)
                try:
                    fig.savefig(temporary, dpi=110)
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            finally:
                plt.close(fig)
