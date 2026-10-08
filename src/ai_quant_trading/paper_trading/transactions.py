"""逐棒耐久提交：SQLite 為待提交來源，CSV、帳戶及圖文日誌可冪等恢復。"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import base64
import csv
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sqlite3

import pandas as pd
from filelock import FileLock

from ai_quant_trading.persistence import write_json_atomic

CURRENT = ContextVar("paper_bar_transaction", default=None)


def _encode(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def snapshot(value):
    return json.loads(json.dumps(value, default=_encode), parse_constant=lambda _: None)


def capture_csv(path, columns, row):
    active = CURRENT.get()
    if active is None:
        return False
    path = Path(path).resolve()
    if path.parent != active["root"] or path.suffix != ".csv":
        raise ValueError("逐棒提交拒絕帳戶外的 CSV")
    active["rows"].append((path.name, columns, snapshot(row)))
    return True


def capture_journal(method, args, kwargs):
    active = CURRENT.get()
    if active is None:
        return False
    active["journal"].append(snapshot([method, args, kwargs]))
    return True


@contextmanager
def database(root):
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "paper_commits.sqlite3", timeout=5)
    try:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS commits (id INTEGER PRIMARY KEY, payload TEXT NOT NULL, applied INTEGER NOT NULL DEFAULT 0)")
        with db:
            yield db
    finally:
        db.close()


def _file_plans(active):
    grouped = {}
    for name, columns, row in active["rows"]:
        group = grouped.setdefault(name, {"columns": columns, "rows": []})
        if group["columns"] != columns:
            raise ValueError("同次提交的 CSV 欄位不可改變")
        group["rows"].append(row)
    plans = []
    for name, group in grouped.items():
        path = active["root"] / name
        offset = path.stat().st_size if path.exists() else 0
        tail = b""
        if offset:
            with path.open("rb") as stream:
                header = next(csv.reader([stream.readline().decode("utf-8").strip()]))
                if header != group["columns"]:
                    # 遷移舊欄位先另存副本；原檔保留，不在未提交狀態偷偷覆寫。
                    raise ValueError("舊 CSV 欄位需先遷移，請使用 migrate_account_csv_schema")
                stream.seek(max(0, offset - 4096))
                tail = stream.read()
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=group["columns"], extrasaction="ignore")
        if not offset:
            writer.writeheader()
        writer.writerows(group["rows"])
        plans.append({"name": name, "offset": offset, "tail_sha": hashlib.sha256(tail).hexdigest(),
                      "data": base64.b64encode(text.getvalue().encode("utf-8")).decode("ascii")})
    return plans


def _apply_file(root, plan):
    path = (root / plan["name"]).resolve()
    if path.parent != root.resolve() or path.suffix != ".csv":
        raise ValueError("提交紀錄包含非法路徑")
    data, offset = base64.b64decode(plan["data"], validate=True), plan["offset"]
    if not path.exists():
        path.touch()
    with path.open("r+b") as stream:
        stream.seek(0, 2)
        length = stream.tell()
        if length < offset or length > offset + len(data):
            raise ValueError("CSV 長度與提交紀錄不符，禁止覆寫外部修改")
        stream.seek(max(0, offset - 4096))
        tail = stream.read(min(4096, offset))
        if hashlib.sha256(tail).hexdigest() != plan["tail_sha"]:
            raise ValueError("CSV 前段已被修改，停止恢復")
        stream.seek(offset)
        written = stream.read()
        if not data.startswith(written):
            raise ValueError("CSV 尾端與待恢復資料不一致")
        stream.seek(0, 2)
        stream.write(data[len(written):])
        stream.flush()
        os.fsync(stream.fileno())


def recover(paths):
    from ai_quant_trading.paper_trading.state import PaperAccountState
    from ai_quant_trading.trading.journal import TradeJournal
    root = paths.account_dir.resolve()
    with database(root) as db:
        pending = db.execute("SELECT id,payload,applied FROM commits WHERE applied<2 ORDER BY id").fetchall()
    journal_failed = False
    for commit_id, encoded, applied in pending:
        payload = json.loads(encoded)
        if applied == 0:
            for plan in payload["files"]:
                _apply_file(root, plan)
            write_json_atomic(paths.account_json, payload["state"])
            with database(root) as db:
                db.execute("UPDATE commits SET applied=1 WHERE id=?", (commit_id,))
        if journal_failed:
            continue
        journal = TradeJournal(root / "trade_journal")
        try:
            for method, args, kwargs in payload["journal"]:
                if method not in {"market", "decision", "opened", "closed"}:
                    raise ValueError("未知日誌提交操作")
                if method != "market":
                    args[0] = PaperAccountState.from_dict(args[0])
                getattr(journal, method)(*args, **kwargs)
        except Exception as error:
            journal_failed = True
            logging.getLogger(__name__).warning("日誌等待耐久佇列重試：%s", type(error).__name__)
            try:
                write_json_atomic(root / "trade_journal" / "recovery_status.json", {
                    "pending": True, "commit_id": commit_id, "error": type(error).__name__,
                    "account_applied": True, "trading_blocked": False})
            except OSError:
                pass
            continue
        with database(root) as db:
            db.execute("UPDATE commits SET applied=2 WHERE id=?", (commit_id,))
            # 已投影的完整資料在日誌與 CSV；只保留最近 256 棒作本機修復審計。
            db.execute("DELETE FROM commits WHERE applied=2 AND id < ?", (commit_id - 256,))
    if pending and not journal_failed:
        try:
            write_json_atomic(root / "trade_journal" / "recovery_status.json", {"pending": False})
        except OSError:
            pass
    return len(pending)


@contextmanager
def bar_transaction(paths, state):
    active = {"root": paths.account_dir.resolve(), "rows": [], "journal": []}
    token = CURRENT.set(active)
    try:
        yield
        payload = {"files": _file_plans(active), "state": snapshot(state), "journal": active["journal"]}
        with database(active["root"]) as db:
            db.execute("INSERT INTO commits(payload) VALUES (?)", (json.dumps(payload, ensure_ascii=False, allow_nan=False),))
    finally:
        CURRENT.reset(token)
    recover(paths)


def recoverable_cycle(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        from ai_quant_trading.paper_trading.storage import account_paths
        paths = account_paths(kwargs["root_dir"], kwargs["account_id"])
        paths.account_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(paths.account_dir / "paper_commits.lock"), timeout=30):
            recover(paths)
            migrate_account_csv_schema(paths)
            return function(*args, **kwargs)
    return wrapped


def migrate_account_csv_schema(paths):
    """只擴充已知欄位，備份原檔；復原待辦必須先完成。"""
    from ai_quant_trading.paper_trading import storage
    import shutil
    import tempfile
    for name, columns in (("orders", storage.ORDER_COLUMNS), ("trades", storage.TRADE_COLUMNS),
                          ("predictions", storage.PREDICTION_COLUMNS), ("performance", storage.PERFORMANCE_COLUMNS),
                          ("positions", storage.POSITION_COLUMNS)):
        path = getattr(paths, name + "_csv")
        if not path.exists() or not path.stat().st_size:
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream))
        if header == columns:
            continue
        if not set(header).issubset(columns):
            raise ValueError("拒絕自動移除未知 CSV 欄位")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        backup = path.with_name(path.name + f".{digest[:16]}.schema_backup")
        if backup.exists():
            if hashlib.sha256(backup.read_bytes()).hexdigest() != digest:
                raise ValueError("欄位遷移備份內容不符，停止覆寫")
        else:
            shutil.copy2(path, backup)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                pd.read_csv(path).reindex(columns=columns).to_csv(stream, index=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
