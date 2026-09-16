"""不保存憑證的雜湊鏈交易稽核紀錄。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
from time import sleep, monotonic, time
from typing import Mapping
from uuid import uuid4


GENESIS_HASH = "0" * 64


@dataclass(frozen=True, slots=True)
class AuditVerification:
    """逐行驗證稽核事件及前後雜湊是否一致。"""

    valid: bool
    event_count: int
    last_hash: str
    reasons: tuple[str, ...]


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _event_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _last_record(path: Path) -> dict[str, object] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    last = ""
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = line
    return dict(json.loads(last)) if last else None


def _acquire_lock(lock_path: Path, timeout_seconds: float = 3.0) -> int:
    """以同目錄排他檔避免 Dashboard 與背景程序同時破壞雜湊順序。"""
    started = monotonic()
    while True:
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except PermissionError as exc:
            # Windows 在另一執行緒剛建立或刪除鎖檔時，可能短暫回傳
            # WinError 5/32；將它視為共享衝突，但仍以逾時保留失敗訊號。
            is_windows_sharing_conflict = os.name == "nt" and (
                getattr(exc, "winerror", None) in {5, 32}
                or exc.errno == errno.EACCES
            )
            if not is_windows_sharing_conflict:
                raise
            if monotonic() - started >= timeout_seconds:
                raise TimeoutError("稽核鎖檔持續遭 Windows 拒絕存取") from exc
            sleep(0.02)
        except FileExistsError:
            try:
                stale = time() - lock_path.stat().st_mtime > 30
            except OSError:
                stale = False
            if stale:
                lock_path.unlink(missing_ok=True)
                continue
            if monotonic() - started >= timeout_seconds:
                raise TimeoutError("稽核紀錄目前正由另一個程序寫入")
            sleep(0.02)


def append_audit_event(
    path: str | Path,
    event_type: str,
    details: Mapping[str, object],
    *,
    severity: str = "info",
    occurred_at: str | None = None,
) -> dict[str, object]:
    """追加一筆可驗證事件；details 禁止放入 API Key 或 Secret。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    descriptor = _acquire_lock(lock_path)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        previous = _last_record(target)
        unsigned: dict[str, object] = {
            "schema_version": 1,
            "event_id": uuid4().hex,
            "occurred_at": occurred_at or datetime.now(timezone.utc).isoformat(),
            "event_type": str(event_type),
            "severity": str(severity).lower(),
            "details": dict(details),
            "previous_hash": (
                str(previous.get("event_hash")) if previous is not None else GENESIS_HASH
            ),
        }
        event = {**unsigned, "event_hash": _event_hash(unsigned)}
        with target.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def verify_audit_log(path: str | Path) -> AuditVerification:
    """檢查 JSON 格式、事件本身雜湊與整條前序雜湊鏈。"""
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        return AuditVerification(True, 0, GENESIS_HASH, ())
    expected_previous = GENESIS_HASH
    count = 0
    reasons: list[str] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = dict(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError):
                reasons.append(f"第 {line_number} 行不是有效 JSON")
                break
            stored_hash = str(event.pop("event_hash", ""))
            if str(event.get("previous_hash")) != expected_previous:
                reasons.append(f"第 {line_number} 行的前序雜湊不一致")
                break
            calculated = _event_hash(event)
            if stored_hash != calculated:
                reasons.append(f"第 {line_number} 行內容可能已被修改")
                break
            expected_previous = stored_hash
            count += 1
    return AuditVerification(not reasons, count, expected_previous, tuple(reasons))
