"""具心跳、單一執行鎖、錯誤熔斷與停止檔的常駐排程器。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Literal


AutomationState = Literal[
    "idle",
    "running",
    "stopping",
    "stopped",
    "completed",
    "failed",
]


@dataclass(frozen=True, slots=True)
class AutomationPaths:
    """單一自動交易工作的控制與狀態檔。"""

    root_dir: Path
    status_json: Path
    stop_file: Path
    lock_file: Path


@dataclass(frozen=True, slots=True)
class AutomationConfig:
    """排程間隔與連續錯誤熔斷限制；max_cycles=0 代表持續執行。"""

    poll_seconds: float = 60.0
    max_cycles: int = 0
    max_consecutive_errors: int = 5

    def __post_init__(self) -> None:
        if self.poll_seconds < 0:
            raise ValueError("poll_seconds 不可小於 0")
        if self.max_cycles < 0:
            raise ValueError("max_cycles 不可小於 0")
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors 必須大於 0")


@dataclass(frozen=True, slots=True)
class AutomationStatus:
    """可由 UI 或另一個程序安全讀取的 runner 心跳。"""

    state: AutomationState = "idle"
    pid: int | None = None
    started_at: str | None = None
    heartbeat_at: str | None = None
    stopped_at: str | None = None
    attempted_cycles: int = 0
    successful_cycles: int = 0
    consecutive_errors: int = 0
    last_message: str = "尚未啟動"
    last_error: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AutomationStatus":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in allowed})


def automation_paths(root_dir: str | Path) -> AutomationPaths:
    target = Path(root_dir).resolve()
    return AutomationPaths(
        target,
        target / "status.json",
        target / "stop.requested",
        target / "runner.lock",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            mode="w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(min(0.02 * (2**attempt), 0.25))
    finally:
        temporary.unlink(missing_ok=True)


def read_automation_status(paths: AutomationPaths) -> AutomationStatus:
    if not paths.status_json.exists():
        return AutomationStatus()
    try:
        payload = json.loads(paths.status_json.read_text(encoding="utf-8"))
        status = AutomationStatus.from_dict(dict(payload))
        if (
            status.state in {"running", "stopping"}
            and status.pid
            and not _process_is_running(status.pid)
        ):
            values = asdict(status)
            values.update(
                {
                    "state": "failed",
                    "stopped_at": _utc_now(),
                    "last_message": "背景程序已中斷",
                    "last_error": "runner_process_not_found",
                }
            )
            return AutomationStatus(**values)
        return status
    except (OSError, ValueError, TypeError):
        return AutomationStatus(state="failed", last_message="狀態檔損壞", last_error="invalid_json")


def request_automation_stop(paths: AutomationPaths) -> None:
    """建立停止請求；runner 最晚會在下一次輪詢前停止。"""
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    paths.stop_file.write_text(_utc_now(), encoding="utf-8")
    status = read_automation_status(paths)
    if status.state == "running":
        values = asdict(status)
        values.update(
            {
                "state": "stopping",
                "heartbeat_at": _utc_now(),
                "last_message": "已送出安全停止請求，等待目前工作完成",
            }
        )
        _write_json_atomic(paths.status_json, values)


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # Windows 的 os.kill(pid, 0) 不是可靠的唯讀探測，改用受限查詢 handle。
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_lock(paths: AutomationPaths) -> None:
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    if paths.lock_file.exists():
        try:
            old_pid = int(paths.lock_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            old_pid = -1
        if _process_is_running(old_pid):
            raise RuntimeError(f"自動交易工作已在執行，PID={old_pid}")
        paths.lock_file.unlink(missing_ok=True)
    descriptor = os.open(paths.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))


def _status(
    *,
    state: AutomationState,
    started_at: str,
    attempted: int,
    successful: int,
    consecutive_errors: int,
    message: str,
    error: str | None,
) -> AutomationStatus:
    now = _utc_now()
    return AutomationStatus(
        state=state,
        pid=os.getpid(),
        started_at=started_at,
        heartbeat_at=now,
        stopped_at=now if state in {"stopped", "completed", "failed"} else None,
        attempted_cycles=attempted,
        successful_cycles=successful,
        consecutive_errors=consecutive_errors,
        last_message=message,
        last_error=error,
    )


def run_automation(
    cycle: Callable[[], object],
    paths: AutomationPaths,
    config: AutomationConfig | None = None,
    *,
    event_handler: Callable[[str, str], None] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> AutomationStatus:
    """持續呼叫單次交易流程；連續失敗達上限時自動熔斷。"""
    config = config or AutomationConfig()
    _acquire_lock(paths)
    paths.stop_file.unlink(missing_ok=True)
    started_at = _utc_now()
    attempted = 0
    successful = 0
    consecutive_errors = 0
    current = _status(
        state="running",
        started_at=started_at,
        attempted=0,
        successful=0,
        consecutive_errors=0,
        message="自動交易工作已啟動",
        error=None,
    )
    _write_json_atomic(paths.status_json, asdict(current))
    try:
        while True:
            if paths.stop_file.exists():
                current = _status(
                    state="stopped",
                    started_at=started_at,
                    attempted=attempted,
                    successful=successful,
                    consecutive_errors=consecutive_errors,
                    message="收到停止請求，已安全停止",
                    error=None,
                )
                break
            if config.max_cycles and attempted >= config.max_cycles:
                current = _status(
                    state="completed",
                    started_at=started_at,
                    attempted=attempted,
                    successful=successful,
                    consecutive_errors=consecutive_errors,
                    message="已完成指定輪數",
                    error=None,
                )
                break

            attempted += 1
            try:
                outcome = cycle()
                successful += 1
                consecutive_errors = 0
                message = str(getattr(outcome, "message", outcome or "本輪完成"))
                error = None
                if event_handler is not None:
                    event_handler("info", message)
            except Exception as exc:
                consecutive_errors += 1
                message = f"第 {attempted} 輪失敗"
                error = f"{type(exc).__name__}: {exc}"
                if event_handler is not None:
                    event_handler("error", error)
                if consecutive_errors >= config.max_consecutive_errors:
                    current = _status(
                        state="failed",
                        started_at=started_at,
                        attempted=attempted,
                        successful=successful,
                        consecutive_errors=consecutive_errors,
                        message="連續錯誤達上限，已自動熔斷",
                        error=error,
                    )
                    break

            current = _status(
                state="running",
                started_at=started_at,
                attempted=attempted,
                successful=successful,
                consecutive_errors=consecutive_errors,
                message=message,
                error=error,
            )
            _write_json_atomic(paths.status_json, asdict(current))
            if config.poll_seconds > 0:
                remaining = config.poll_seconds
                while remaining > 0 and not paths.stop_file.exists():
                    chunk = min(1.0, remaining)
                    sleeper(chunk)
                    remaining -= chunk
    finally:
        _write_json_atomic(paths.status_json, asdict(current))
        paths.lock_file.unlink(missing_ok=True)
    return current
